from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple

import torch
from torch import nn

from networks.Basic_Model.Unet_restnet import UNetResNet152
from utils.model_output import BaseSegmentationModel, extract_features, extract_logits


def _copy_norm_subset(source_norm: nn.Module, target_norm: nn.Module, indices: Sequence[int]) -> None:
    index = torch.as_tensor(indices, dtype=torch.long, device=next(source_norm.parameters()).device)
    for attr in ("weight", "bias", "running_mean", "running_var"):
        source_value = getattr(source_norm, attr, None)
        target_value = getattr(target_norm, attr, None)
        if source_value is None or target_value is None:
            continue
        target_value.data.copy_(source_value.data.index_select(0, index).to(target_value.device))
    if hasattr(source_norm, "num_batches_tracked") and hasattr(target_norm, "num_batches_tracked"):
        target_norm.num_batches_tracked.data.copy_(source_norm.num_batches_tracked.data.to(target_norm.num_batches_tracked.device))


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


def _make_batchnorm_like(source: nn.BatchNorm2d, num_features: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(
        num_features=num_features,
        eps=source.eps,
        momentum=source.momentum,
        affine=source.affine,
        track_running_stats=source.track_running_stats,
    )


def _replace_bottleneck_middle(block: nn.Module, keep_channels: int) -> None:
    if not all(hasattr(block, name) for name in ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3")):
        raise TypeError(f"Expected a ResNet bottleneck-like block, got {type(block)!r}.")
    if not isinstance(block.conv2, nn.Conv2d) or not isinstance(block.conv3, nn.Conv2d):
        raise TypeError("Expected bottleneck conv2/conv3 to be Conv2d layers.")
    if not isinstance(block.bn2, nn.BatchNorm2d):
        raise TypeError("Expected bottleneck bn2 to be BatchNorm2d.")

    keep_channels = int(keep_channels)
    if keep_channels <= 0:
        raise ValueError("keep_channels must be > 0 for middle bottleneck pruning.")
    if keep_channels > int(block.conv2.out_channels):
        raise ValueError("keep_channels cannot exceed the original conv2 out_channels.")

    block.conv2 = _make_conv2d_like(block.conv2, int(block.conv2.in_channels), keep_channels)
    block.bn2 = _make_batchnorm_like(block.bn2, keep_channels)
    block.conv3 = _make_conv2d_like(block.conv3, keep_channels, int(block.conv3.out_channels))


def _get_module(root: nn.Module, module_name: str) -> nn.Module:
    modules = dict(root.named_modules())
    if module_name not in modules:
        raise KeyError(f"Module not found: {module_name}")
    return modules[module_name]


def _iter_middle_prune_plan(blueprint: Mapping[str, object]) -> List[dict]:
    plan = list(blueprint.get("middle_prune_plan", []) or [])
    if not plan:
        raise ValueError("middle_static blueprint must contain a non-empty middle_prune_plan.")
    return [dict(row) for row in plan]


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


def _copy_pruned_bottleneck_middle(source_block: nn.Module, target_block: nn.Module, kept_indices: Sequence[int]) -> dict:
    kept_indices = [int(index) for index in kept_indices]
    index = torch.as_tensor(kept_indices, dtype=torch.long, device=source_block.conv2.weight.device)

    target_block.conv2.weight.data.copy_(source_block.conv2.weight.data.index_select(0, index).to(target_block.conv2.weight.device))
    if source_block.conv2.bias is not None and target_block.conv2.bias is not None:
        target_block.conv2.bias.data.copy_(source_block.conv2.bias.data.index_select(0, index).to(target_block.conv2.bias.device))

    _copy_norm_subset(source_block.bn2, target_block.bn2, kept_indices)

    target_block.conv3.weight.data.copy_(source_block.conv3.weight.data.index_select(1, index).to(target_block.conv3.weight.device))
    if source_block.conv3.bias is not None and target_block.conv3.bias is not None:
        target_block.conv3.bias.data.copy_(source_block.conv3.bias.data.to(target_block.conv3.bias.device))

    return {
        "copied": True,
        "kept_middle_channels": int(len(kept_indices)),
        "original_middle_channels": int(source_block.conv2.out_channels),
        "protected_components": ["conv1", "bn1", "conv3_out", "bn3", "downsample"],
        "pruned_components": ["conv2_out", "bn2", "conv3_in"],
    }


class MiddlePrunedResNetUNet(BaseSegmentationModel):
    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 2,
        blueprint: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.base_model = UNetResNet152(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_pretrained=False,
        )
        self.middle_prune_plan = _iter_middle_prune_plan(blueprint)
        self.channel_config = tuple(int(row["kept_middle_channels"]) for row in self.middle_prune_plan)
        self.stage_middle_channel_config = {
            str(stage): [int(value) for value in values]
            for stage, values in dict(blueprint.get("stage_middle_channel_config", {}) or {}).items()
        }

        for row in self.middle_prune_plan:
            block = _get_module(self.base_model, str(row["block_name"]))
            _replace_bottleneck_middle(block, int(row["kept_middle_channels"]))

        self.model_name = "middle_pruned_resnet_unet"
        self.backbone_name = "resnet152_middle_pruned"
        self.student_name = "middle_static_student"
        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            channel_config=list(self.channel_config),
            stage_middle_channel_config=self.stage_middle_channel_config,
            pruning_method="middle_static",
            protected_boundary_layers=True,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        base_output = self.base_model(x, return_features=True)
        output = self.build_output(
            extract_logits(base_output),
            features=extract_features(base_output),
            aux={
                "channel_config": list(self.channel_config),
                "stage_middle_channel_config": self.stage_middle_channel_config,
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


def build_middle_pruned_resnet_unet(
    *,
    in_channels: int,
    num_classes: int,
    blueprint: Mapping[str, object],
) -> MiddlePrunedResNetUNet:
    return MiddlePrunedResNetUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        blueprint=blueprint,
    )


def build_middle_pruned_resnet_unet_from_teacher(
    teacher_model: nn.Module,
    *,
    in_channels: int,
    num_classes: int,
    blueprint: Mapping[str, object],
) -> Tuple[MiddlePrunedResNetUNet, dict]:
    student = build_middle_pruned_resnet_unet(
        in_channels=in_channels,
        num_classes=num_classes,
        blueprint=blueprint,
    )
    exact_copy = _copy_exact_matching_base_weights(teacher_model, student.base_model)
    teacher_modules = dict(teacher_model.named_modules())
    student_modules = dict(student.base_model.named_modules())

    rows = []
    copied_blocks = 0
    for row in student.middle_prune_plan:
        block_name = str(row["block_name"])
        source_block = teacher_modules.get(block_name)
        target_block = student_modules.get(block_name)
        transfer_row = {
            "block_name": block_name,
            "middle_layer_name": row.get("middle_layer_name"),
            "status": "skipped",
            "kept_middle_channels": int(row["kept_middle_channels"]),
            "original_middle_channels": int(row["original_middle_channels"]),
            "protected_components": ["conv1", "bn1", "conv3_out", "bn3", "downsample"],
            "pruned_components": ["conv2_out", "bn2", "conv3_in"],
        }
        if source_block is None or target_block is None:
            transfer_row["status"] = "block_not_found"
        else:
            result = _copy_pruned_bottleneck_middle(source_block, target_block, row["kept_channel_indices"])
            transfer_row.update(result)
            transfer_row["status"] = "middle_subset_reused_boundary_full"
            copied_blocks += 1
        rows.append(transfer_row)

    return student, {
        "strategy": "resnet_bottleneck_middle_static_pruning",
        "source": "teacher_unet_resnet152",
        "copied_blocks": int(copied_blocks),
        "requested_blocks": int(len(rows)),
        "block_transfer_ratio": float(copied_blocks / max(1, len(rows))),
        "exact_matching_full_weight_copy": exact_copy,
        "stage_middle_channel_config": student.stage_middle_channel_config,
        "rows": rows,
    }
