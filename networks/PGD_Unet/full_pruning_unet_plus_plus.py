from __future__ import annotations

import types
from typing import List, Mapping, Sequence, Tuple

import torch
from torch import nn

from networks.Basic_Model.unet_plus_plus import UNetPlusPlus2D
from networks.Basic_Model.Unet_restnet import DecoderBlock, FinalUpBlock, UNetResNet152
from networks.Basic_Model.common import DoubleConv2d
from utils.model_output import BaseSegmentationModel, extract_features, extract_logits


def _as_indices(values: Sequence[int] | None, fallback_count: int) -> List[int]:
    if values is None:
        return list(range(int(fallback_count)))
    return [int(index) for index in values]


def _index_tensor(indices: Sequence[int], *, device) -> torch.Tensor:
    return torch.as_tensor([int(index) for index in indices], dtype=torch.long, device=device)


def _copy_norm_subset(source_norm: nn.Module, target_norm: nn.Module, indices: Sequence[int]) -> None:
    index = _index_tensor(indices, device=next(source_norm.parameters()).device)
    for attr in ("weight", "bias", "running_mean", "running_var"):
        source_value = getattr(source_norm, attr, None)
        target_value = getattr(target_norm, attr, None)
        if source_value is None or target_value is None:
            continue
        target_value.data.copy_(source_value.data.index_select(0, index).to(target_value.device))
    if hasattr(source_norm, "num_batches_tracked") and hasattr(target_norm, "num_batches_tracked"):
        target_norm.num_batches_tracked.data.copy_(source_norm.num_batches_tracked.data.to(target_norm.num_batches_tracked.device))


def _reset_norm_identity(norm: nn.Module) -> None:
    if hasattr(norm, "weight") and norm.weight is not None:
        norm.weight.data.fill_(1.0)
    if hasattr(norm, "bias") and norm.bias is not None:
        norm.bias.data.zero_()
    if hasattr(norm, "running_mean") and norm.running_mean is not None:
        norm.running_mean.data.zero_()
    if hasattr(norm, "running_var") and norm.running_var is not None:
        norm.running_var.data.fill_(1.0)
    if hasattr(norm, "num_batches_tracked"):
        norm.num_batches_tracked.data.zero_()


def _make_conv2d_like(source: nn.Conv2d, in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=source.groups,
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
    )


def _make_projection_conv(source: nn.Conv2d | None, in_channels: int, out_channels: int, stride) -> nn.Conv2d:
    if source is not None:
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=source.kernel_size,
            stride=source.stride,
            padding=source.padding,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
        )
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def _make_batchnorm_like(source: nn.BatchNorm2d | None, num_features: int) -> nn.BatchNorm2d:
    if source is None:
        return nn.BatchNorm2d(num_features)
    return nn.BatchNorm2d(
        num_features=num_features,
        eps=source.eps,
        momentum=source.momentum,
        affine=source.affine,
        track_running_stats=source.track_running_stats,
    )


def _get_module(root: nn.Module, module_name: str) -> nn.Module:
    modules = dict(root.named_modules())
    if module_name not in modules:
        raise KeyError(f"Module not found: {module_name}")
    return modules[module_name]


def _first_conv_bn(sequence: nn.Module | None) -> tuple[nn.Conv2d | None, nn.BatchNorm2d | None]:
    if sequence is None:
        return None, None
    conv = None
    norm = None
    for module in sequence.modules():
        if module is sequence:
            continue
        if conv is None and isinstance(module, nn.Conv2d):
            conv = module
        elif norm is None and isinstance(module, nn.BatchNorm2d):
            norm = module
        if conv is not None and norm is not None:
            break
    return conv, norm


def _iter_full_prune_plan(blueprint: Mapping[str, object]) -> List[dict]:
    plan = list(blueprint.get("full_prune_plan", []) or [])
    if not plan:
        raise ValueError("Full-pruned ResNet blueprint must contain a non-empty full_prune_plan.")
    return [dict(row) for row in plan]


def _stage_output_count(blueprint: Mapping[str, object], stage_name: str, default_count: int) -> int:
    counts = dict(blueprint.get("stage_full_output_channel_config", {}) or {}).get(stage_name)
    if counts:
        return int(list(counts)[-1])
    plan = [row for row in _iter_full_prune_plan(blueprint) if str(row.get("stage_name")) == stage_name]
    if plan:
        return int(plan[-1].get("kept_output_channels", plan[-1].get("kept_internal_channels", default_count)))
    return int(default_count)


def _stage_output_indices(blueprint: Mapping[str, object], stage_name: str, default_count: int) -> List[int]:
    stage_indices = dict(blueprint.get("stage_output_kept_indices", {}) or {}).get(stage_name)
    if stage_indices:
        return _as_indices(stage_indices, default_count)
    plan = [row for row in _iter_full_prune_plan(blueprint) if str(row.get("stage_name")) == stage_name]
    if plan:
        return _as_indices(plan[-1].get("output_kept_channel_indices", plan[-1].get("kept_channel_indices")), default_count)
    return list(range(int(default_count)))


def _replace_bottleneck_full(block: nn.Module, row: Mapping[str, object]) -> None:
    if not all(hasattr(block, name) for name in ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3")):
        raise TypeError(f"Expected a ResNet bottleneck-like block, got {type(block)!r}.")
    if not all(isinstance(getattr(block, name), nn.Conv2d) for name in ("conv1", "conv2", "conv3")):
        raise TypeError("Expected bottleneck conv1/conv2/conv3 to be Conv2d layers.")
    if not all(isinstance(getattr(block, name), nn.BatchNorm2d) for name in ("bn1", "bn2", "bn3")):
        raise TypeError("Expected bottleneck bn1/bn2/bn3 to be BatchNorm2d layers.")

    input_indices = _as_indices(row.get("input_channel_indices"), block.conv1.in_channels)
    internal_indices = _as_indices(row.get("internal_kept_channel_indices", row.get("kept_channel_indices")), block.conv2.out_channels)
    output_indices = _as_indices(row.get("output_kept_channel_indices", row.get("kept_channel_indices")), block.conv3.out_channels)
    input_channels = len(input_indices)
    internal_channels = len(internal_indices)
    output_channels = len(output_indices)
    if min(input_channels, internal_channels, output_channels) <= 0:
        raise ValueError("Full bottleneck pruning requires non-empty input/internal/output channel sets.")

    original_downsample = block.downsample
    source_projection_conv, source_projection_bn = _first_conv_bn(original_downsample)
    stride = block.conv2.stride
    needs_projection = (
        original_downsample is not None
        or input_channels != output_channels
        or list(input_indices) != list(output_indices)
        or tuple(stride) != (1, 1)
    )

    block.conv1 = _make_conv2d_like(block.conv1, input_channels, internal_channels)
    block.bn1 = _make_batchnorm_like(block.bn1, internal_channels)
    block.conv2 = _make_conv2d_like(block.conv2, internal_channels, internal_channels)
    block.bn2 = _make_batchnorm_like(block.bn2, internal_channels)
    block.conv3 = _make_conv2d_like(block.conv3, internal_channels, output_channels)
    block.bn3 = _make_batchnorm_like(block.bn3, output_channels)
    if needs_projection:
        projection_conv = _make_projection_conv(source_projection_conv, input_channels, output_channels, stride)
        projection_bn = _make_batchnorm_like(source_projection_bn, output_channels)
        block.downsample = nn.Sequential(projection_conv, projection_bn)
    else:
        block.downsample = None


def _rebuild_decoder_for_stage_outputs(base_model: UNetResNet152, blueprint: Mapping[str, object]) -> None:
    stage1 = _stage_output_count(blueprint, "layer1", 256)
    stage2 = _stage_output_count(blueprint, "layer2", 512)
    stage3 = _stage_output_count(blueprint, "layer3", 1024)
    stage4 = _stage_output_count(blueprint, "layer4", 2048)
    base_model.center = DoubleConv2d(stage4, 2048, normalization="batchnorm")
    base_model.dec4 = DecoderBlock(2048, stage3, 512, normalization="batchnorm")
    base_model.dec3 = DecoderBlock(512, stage2, 256, normalization="batchnorm")
    base_model.dec2 = DecoderBlock(256, stage1, 128, normalization="batchnorm")
    base_model.dec1 = DecoderBlock(128, 64, 64, normalization="batchnorm")
    base_model.final_up = FinalUpBlock(64, 32, normalization="batchnorm")


def _copy_conv_subset(source: nn.Conv2d, target: nn.Conv2d, *, out_indices: Sequence[int] | None = None, in_indices: Sequence[int] | None = None) -> None:
    weight = source.weight.data
    if out_indices is not None:
        weight = weight.index_select(0, _index_tensor(out_indices, device=source.weight.device))
    if in_indices is not None:
        weight = weight.index_select(1, _index_tensor(in_indices, device=source.weight.device))
    target.weight.data.copy_(weight.to(target.weight.device))
    if source.bias is not None and target.bias is not None:
        bias = source.bias.data
        if out_indices is not None:
            bias = bias.index_select(0, _index_tensor(out_indices, device=source.bias.device))
        target.bias.data.copy_(bias.to(target.bias.device))


def _copy_exact_matching_base_weights(source_model: nn.Module, target_model: nn.Module) -> dict:
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    copied = []
    skipped = []
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None or source_tensor.shape != target_tensor.shape:
            skipped.append(key)
            continue
        target_state[key] = source_tensor.detach().clone()
        copied.append(key)
    target_model.load_state_dict(target_state, strict=False)
    return {
        "copied_tensor_keys": int(len(copied)),
        "skipped_tensor_keys": int(len(skipped)),
        "total_target_tensors": int(len(target_state)),
        "copy_ratio": float(len(copied) / max(1, len(target_state))),
        "copied_key_examples": copied[:20],
        "skipped_key_examples": skipped[:20],
    }


def _initialize_selector_projection(target_block: nn.Module, input_indices: Sequence[int], output_indices: Sequence[int]) -> None:
    if target_block.downsample is None:
        return
    target_conv, target_bn = _first_conv_bn(target_block.downsample)
    if target_conv is None:
        return
    target_conv.weight.data.zero_()
    if target_conv.bias is not None:
        target_conv.bias.data.zero_()
    input_position = {int(index): position for position, index in enumerate(input_indices)}
    for out_position, original_index in enumerate(output_indices):
        in_position = input_position.get(int(original_index))
        if in_position is not None:
            target_conv.weight.data[out_position, in_position, 0, 0] = 1.0
    if target_bn is not None:
        _reset_norm_identity(target_bn)


def _copy_pruned_bottleneck_full(source_block: nn.Module, target_block: nn.Module, row: Mapping[str, object]) -> dict:
    input_indices = _as_indices(row.get("input_channel_indices"), source_block.conv1.in_channels)
    internal_indices = _as_indices(row.get("internal_kept_channel_indices", row.get("kept_channel_indices")), source_block.conv2.out_channels)
    output_indices = _as_indices(row.get("output_kept_channel_indices", row.get("kept_channel_indices")), source_block.conv3.out_channels)

    _copy_conv_subset(source_block.conv1, target_block.conv1, out_indices=internal_indices, in_indices=input_indices)
    _copy_norm_subset(source_block.bn1, target_block.bn1, internal_indices)

    _copy_conv_subset(source_block.conv2, target_block.conv2, out_indices=internal_indices, in_indices=internal_indices)
    _copy_norm_subset(source_block.bn2, target_block.bn2, internal_indices)

    _copy_conv_subset(source_block.conv3, target_block.conv3, out_indices=output_indices, in_indices=internal_indices)
    _copy_norm_subset(source_block.bn3, target_block.bn3, output_indices)

    source_projection_conv, source_projection_bn = _first_conv_bn(source_block.downsample)
    target_projection_conv, target_projection_bn = _first_conv_bn(target_block.downsample)
    projection_mode = "none"
    if target_projection_conv is not None:
        if source_projection_conv is not None:
            _copy_conv_subset(source_projection_conv, target_projection_conv, out_indices=output_indices, in_indices=input_indices)
            if source_projection_bn is not None and target_projection_bn is not None:
                _copy_norm_subset(source_projection_bn, target_projection_bn, output_indices)
            projection_mode = "teacher_downsample_subset"
        else:
            _initialize_selector_projection(target_block, input_indices, output_indices)
            projection_mode = "inserted_channel_selector"

    return {
        "copied": True,
        "kept_input_channels": int(len(input_indices)),
        "kept_internal_channels": int(len(internal_indices)),
        "kept_output_channels": int(len(output_indices)),
        "original_input_channels": int(source_block.conv1.in_channels),
        "original_internal_channels": int(source_block.conv2.out_channels),
        "original_output_channels": int(source_block.conv3.out_channels),
        "projection_mode": projection_mode,
        "protected_components": ["spatial_stride", "residual_add_semantics"],
        "pruned_components": ["conv1_in", "conv1_out", "bn1", "conv2_in", "conv2_out", "bn2", "conv3_in", "conv3_out", "bn3", "downsample_out"],
    }


def _first_conv_in_doubleconv(module: nn.Module) -> nn.Conv2d | None:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return child
    return None


def _copy_first_conv_input_subset(source_module: nn.Module, target_module: nn.Module, input_indices: Sequence[int]) -> bool:
    source_conv = _first_conv_in_doubleconv(source_module)
    target_conv = _first_conv_in_doubleconv(target_module)
    if source_conv is None or target_conv is None:
        return False
    if source_conv.out_channels != target_conv.out_channels or source_conv.kernel_size != target_conv.kernel_size:
        return False
    _copy_conv_subset(source_conv, target_conv, in_indices=input_indices)
    return True


def _copy_decoder_input_subsets(source_model: UNetResNet152, target_model: UNetResNet152, blueprint: Mapping[str, object]) -> dict:
    stage1_indices = _stage_output_indices(blueprint, "layer1", 256)
    stage2_indices = _stage_output_indices(blueprint, "layer2", 512)
    stage3_indices = _stage_output_indices(blueprint, "layer3", 1024)
    stage4_indices = _stage_output_indices(blueprint, "layer4", 2048)
    rows = []
    rows.append({"module": "center", "copied": _copy_first_conv_input_subset(source_model.center, target_model.center, stage4_indices)})

    decoder_specs = (
        ("dec4.conv", source_model.dec4.conv, target_model.dec4.conv, 512, stage3_indices),
        ("dec3.conv", source_model.dec3.conv, target_model.dec3.conv, 256, stage2_indices),
        ("dec2.conv", source_model.dec2.conv, target_model.dec2.conv, 128, stage1_indices),
    )
    for name, source_module, target_module, up_channels, skip_indices in decoder_specs:
        input_indices = list(range(int(up_channels))) + [int(up_channels) + int(index) for index in skip_indices]
        rows.append({"module": name, "copied": _copy_first_conv_input_subset(source_module, target_module, input_indices)})
    return {"decoder_subset_rows": rows}


def _blueprint_uses_unet_plus_plus(blueprint: Mapping[str, object]) -> bool:
    return str(blueprint.get("teacher_model", "")).lower() == "unet_plus_plus"


def _stage_alias(stage_name: str) -> str:
    return str(stage_name).split(".")[-1]


def _patch_unet_plus_plus_encoder_expanders(base_model: UNetPlusPlus2D, blueprint: Mapping[str, object]) -> None:
    encoder = base_model.model.encoder
    default_stage_channels = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
    expanders = nn.ModuleDict()
    stage_indices = dict(blueprint.get("stage_output_kept_indices", {}) or {})
    for stage_name, original_channels in default_stage_channels.items():
        matched_key = next((key for key in stage_indices if _stage_alias(key) == stage_name), stage_name)
        kept_indices = _stage_output_indices(blueprint, matched_key, original_channels)
        kept_channels = len(kept_indices)
        if kept_channels == original_channels and kept_indices == list(range(original_channels)):
            expanders[stage_name] = nn.Identity()
            continue
        projection = nn.Conv2d(kept_channels, original_channels, kernel_size=1, bias=False)
        projection.weight.data.zero_()
        for compact_index, original_index in enumerate(kept_indices):
            projection.weight.data[int(original_index), int(compact_index), 0, 0] = 1.0
        expanders[stage_name] = projection
    encoder.pgd_stage_expanders = expanders

    def forward_with_pruned_stages(self, x):
        features = [x]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(self.pgd_stage_expanders["layer1"](x))
        x = self.layer2(x)
        features.append(self.pgd_stage_expanders["layer2"](x))
        x = self.layer3(x)
        features.append(self.pgd_stage_expanders["layer3"](x))
        x = self.layer4(x)
        features.append(self.pgd_stage_expanders["layer4"](x))
        depth = int(getattr(self, "_depth", len(features) - 1))
        return features[: depth + 1]

    encoder.forward = types.MethodType(forward_with_pruned_stages, encoder)


class FullPrunedResNetUNet(BaseSegmentationModel):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 2,
        blueprint: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.uses_unet_plus_plus = _blueprint_uses_unet_plus_plus(blueprint)
        if self.uses_unet_plus_plus:
            self.base_model = UNetPlusPlus2D(
                in_channels=in_channels,
                num_classes=num_classes,
                encoder_pretrained=False,
            )
        else:
            self.base_model = UNetResNet152(
                in_channels=in_channels,
                num_classes=num_classes,
                encoder_pretrained=False,
            )
        self.prune_method = str(blueprint.get("prune_method", "full_static"))
        self.full_prune_plan = _iter_full_prune_plan(blueprint)
        self.channel_config = tuple(int(row.get("kept_output_channels", row.get("kept_internal_channels", 0))) for row in self.full_prune_plan)
        self.stage_full_channel_config = {
            str(stage): [int(value) for value in values]
            for stage, values in dict(blueprint.get("stage_full_channel_config", {}) or {}).items()
        }
        self.stage_full_output_channel_config = {
            str(stage): [int(value) for value in values]
            for stage, values in dict(blueprint.get("stage_full_output_channel_config", {}) or {}).items()
        }
        self.stage_output_kept_indices = {
            str(stage): [int(value) for value in values]
            for stage, values in dict(blueprint.get("stage_output_kept_indices", {}) or {}).items()
        }

        for row in self.full_prune_plan:
            block = _get_module(self.base_model, str(row["block_name"]))
            _replace_bottleneck_full(block, row)

        self.model_name = "full_pruning_unet_plus_plus" if self.uses_unet_plus_plus else "full_pruning_resnet_unet"
        self.backbone_name = "resnet152_unet_plus_plus_full_pruned" if self.uses_unet_plus_plus else "resnet152_full_pruned"
        self.student_name = f"{self.prune_method}_student"
        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            channel_config=list(self.channel_config),
            stage_full_channel_config=self.stage_full_channel_config,
            stage_full_output_channel_config=self.stage_full_output_channel_config,
            stage_output_kept_indices=self.stage_output_kept_indices,
            pruning_method=self.prune_method,
            decoder_architecture="unet_plus_plus" if self.uses_unet_plus_plus else "unet",
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        base_output = self.base_model(x, return_features=True)
        features = extract_features(base_output)
        logits = extract_logits(base_output)
        output = self.build_output(
            logits,
            features=features,
            aux={
                "channel_config": list(self.channel_config),
                "stage_full_channel_config": self.stage_full_channel_config,
                "stage_full_output_channel_config": self.stage_full_output_channel_config,
                "stage_output_kept_indices": self.stage_output_kept_indices,
            },
        )
        if return_features:
            return output
        return output

    def get_gate_tensors(self) -> List[torch.Tensor]:
        return []

    def get_gate_modules(self) -> List[nn.Module]:
        return []

    def set_gate_trainable(self, trainable: bool) -> None:
        return None

    def force_gates_open(self, open_probability: float = 0.999) -> None:
        return None


def build_full_pruning_resnet_unet(
    *,
    in_channels: int,
    num_classes: int,
    blueprint: Mapping[str, object],
) -> FullPrunedResNetUNet:
    return FullPrunedResNetUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        blueprint=blueprint,
    )


def build_full_pruning_resnet_unet_from_teacher(
    teacher_model: nn.Module,
    *,
    in_channels: int,
    num_classes: int,
    blueprint: Mapping[str, object],
) -> Tuple[FullPrunedResNetUNet, dict]:
    student = build_full_pruning_resnet_unet(
        in_channels=in_channels,
        num_classes=num_classes,
        blueprint=blueprint,
    )
    exact_copy = _copy_exact_matching_base_weights(teacher_model, student.base_model)
    teacher_modules = dict(teacher_model.named_modules())
    student_modules = dict(student.base_model.named_modules())
    rows = []
    copied_blocks = 0
    for row in student.full_prune_plan:
        block_name = str(row["block_name"])
        source_block = teacher_modules.get(block_name)
        target_block = student_modules.get(block_name)
        transfer_row = {
            "block_name": block_name,
            "status": "skipped",
            "kept_internal_channels": int(row.get("kept_internal_channels", row.get("kept_channel_indices", []).__len__())),
            "kept_output_channels": int(row.get("kept_output_channels", row.get("kept_channel_indices", []).__len__())),
            "protected_components": ["spatial_stride", "residual_add_semantics"],
            "pruned_components": [
                "conv1_in",
                "conv1_out",
                "bn1",
                "conv2_in",
                "conv2_out",
                "bn2",
                "conv3_in",
                "conv3_out",
                "bn3",
                "downsample_out",
            ],
        }
        if source_block is None or target_block is None:
            transfer_row["status"] = "block_not_found"
        else:
            result = _copy_pruned_bottleneck_full(source_block, target_block, row)
            transfer_row.update(result)
            transfer_row["status"] = "full_subset_reused"
            copied_blocks += 1
        rows.append(transfer_row)

    if student.uses_unet_plus_plus:
        _patch_unet_plus_plus_encoder_expanders(student.base_model, blueprint)
        decoder_subset_transfer = _copy_decoder_input_subsets(teacher_model, student.base_model, blueprint)
    else:
        decoder_subset_transfer = _copy_decoder_input_subsets(teacher_model, student.base_model, blueprint)

    return student, {
        "strategy": f"resnet_bottleneck_{student.prune_method}_pruning",
        "source": "teacher_unet_plus_plus" if student.uses_unet_plus_plus else "teacher_unet_resnet152",
        "student_architecture": student.model_name,
        "copied_blocks": int(copied_blocks),
        "requested_blocks": int(len(rows)),
        "block_transfer_ratio": float(copied_blocks / max(1, len(rows))),
        "exact_matching_full_weight_copy": exact_copy,
        "stage_full_channel_config": student.stage_full_channel_config,
        "stage_full_output_channel_config": student.stage_full_output_channel_config,
        "decoder_subset_transfer": decoder_subset_transfer,
        "rows": rows,
    }
