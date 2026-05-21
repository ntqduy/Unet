from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from networks.PGD_Unet.pruning_algorithms.Kneedle_Otsu_GMM import prune_one_layer
from networks.PGD_Unet.pruning_algorithms.pruning_smart import (
    is_full_resnet_pruning,
    is_middle_static_pruning,
    is_middle_resnet_pruning,
    uses_static_prune_ratio,
)
from utils.channel_analysis import find_primary_channel_layer


DEFAULT_BLUEPRINT_MODULES = ("stem", "down1", "down2", "down3", "down4")
PRUNE_METHODS = (
    "static",
    "kneedle",
    "otsu",
    "gmm",
    "middle_static",
    "middle_kneedle",
    "middle_otsu",
    "middle_gmm",
    "full_static",
    "full_kneedle",
    "full_otsu",
    "full_gmm",
)
MODULE_GROUP_CANDIDATES = (
    ("model.encoder.conv1", "model.encoder.layer1", "model.encoder.layer2", "model.encoder.layer3", "model.encoder.layer4"),
    ("stem", "down1", "down2", "down3", "down4"),
    ("stem", "layer1", "layer2", "layer3", "layer4"),
    ("enc1", "enc2", "enc3", "enc4", "bottleneck"),
    ("encoder.block_one", "encoder.block_two", "encoder.block_three", "encoder.block_four", "encoder.block_five"),
)
RESNET_STAGE_MODULES = ("layer1", "layer2", "layer3", "layer4")


def _is_resnet_stage_module(module_name: str) -> bool:
    return str(module_name).split(".")[-1] in RESNET_STAGE_MODULES


def _resolve_module_importance(model: nn.Module, module_name: str, module: nn.Module) -> torch.Tensor:
    bn_candidates = (
        f"{module_name}.bn",
        module_name.replace("conv", "bn"),
        f"{module_name}.norm",
    )
    for candidate_name, candidate_module in model.named_modules():
        if candidate_name in bn_candidates and isinstance(candidate_module, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            if hasattr(candidate_module, "weight") and candidate_module.weight is not None:
                return candidate_module.weight.detach().abs()
    return module.weight.detach().abs().sum(dim=(1, 2, 3))


def _conv_input_importance(module: nn.Conv2d) -> torch.Tensor:
    return module.weight.detach().abs().sum(dim=(0, 2, 3))


def _normalize_importance(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().float()
    max_value = values.max() if values.numel() else values.new_tensor(0.0)
    if float(max_value.item()) <= 0.0:
        return torch.zeros_like(values)
    return values / max_value.clamp_min(1e-12)


def _bottleneck_internal_importance(model: nn.Module, block_name: str, block: nn.Module) -> tuple[torch.Tensor, Dict[str, List[float]]]:
    conv1_out = _resolve_module_importance(model, f"{block_name}.conv1", block.conv1)
    conv2_out = _resolve_module_importance(model, f"{block_name}.conv2", block.conv2)
    conv2_in = _conv_input_importance(block.conv2)
    conv3_in = _conv_input_importance(block.conv3)
    sources = {
        "conv1_out": conv1_out,
        "conv2_in": conv2_in,
        "conv2_out": conv2_out,
        "conv3_in": conv3_in,
    }
    lengths = {int(value.numel()) for value in sources.values()}
    if len(lengths) != 1:
        raise ValueError(f"Internal bottleneck importance sources have mismatched lengths for {block_name}: {sorted(lengths)}")
    combined = torch.stack([_normalize_importance(value) for value in sources.values()]).mean(dim=0)
    source_values = {name: [float(item) for item in value.detach().cpu().tolist()] for name, value in sources.items()}
    source_values["combined_internal"] = [float(item) for item in combined.detach().cpu().tolist()]
    return combined, source_values


def _bottleneck_full_channel_importance(
    model: nn.Module,
    block_name: str,
    block: nn.Module,
    next_input_importance: Optional[torch.Tensor] = None,
    next_input_source: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List[float]]]:
    conv1_out = _resolve_module_importance(model, f"{block_name}.conv1", block.conv1)
    conv2_in = _conv_input_importance(block.conv2)
    conv2_out = _resolve_module_importance(model, f"{block_name}.conv2", block.conv2)
    conv3_in = _conv_input_importance(block.conv3)
    conv3_out = _resolve_module_importance(model, f"{block_name}.conv3", block.conv3)
    if next_input_importance is not None and int(next_input_importance.numel()) != int(conv3_out.numel()):
        raise ValueError(
            f"conv3 output and next input importance lengths mismatch for {block_name}: "
            f"{int(conv3_out.numel())} vs {int(next_input_importance.numel())}"
        )
    conv3_next_in = next_input_importance if next_input_importance is not None else conv3_out

    if int(conv1_out.numel()) != int(conv2_in.numel()):
        raise ValueError(
            f"conv1 output and conv2 input importance lengths mismatch for {block_name}: "
            f"{int(conv1_out.numel())} vs {int(conv2_in.numel())}"
        )
    if int(conv2_out.numel()) != int(conv3_in.numel()):
        raise ValueError(
            f"conv2 output and conv3 input importance lengths mismatch for {block_name}: "
            f"{int(conv2_out.numel())} vs {int(conv3_in.numel())}"
        )

    conv1_to_conv2 = torch.stack((_normalize_importance(conv1_out), _normalize_importance(conv2_in))).mean(dim=0)
    conv2_to_conv3 = torch.stack((_normalize_importance(conv2_out), _normalize_importance(conv3_in))).mean(dim=0)
    conv3_to_next = torch.stack((_normalize_importance(conv3_out), _normalize_importance(conv3_next_in))).mean(dim=0)
    source_values = {
        "conv1_out": [float(item) for item in conv1_out.detach().cpu().tolist()],
        "conv2_in": [float(item) for item in conv2_in.detach().cpu().tolist()],
        "conv2_out": [float(item) for item in conv2_out.detach().cpu().tolist()],
        "conv3_in": [float(item) for item in conv3_in.detach().cpu().tolist()],
        "conv3_out": [float(item) for item in conv3_out.detach().cpu().tolist()],
        "conv3_next_in": [float(item) for item in conv3_next_in.detach().cpu().tolist()],
        "conv3_next_in_source": next_input_source or "conv3_out_fallback",
        "combined_conv1_to_conv2": [float(item) for item in conv1_to_conv2.detach().cpu().tolist()],
        "combined_conv2_to_conv3": [float(item) for item in conv2_to_conv3.detach().cpu().tolist()],
        "combined_conv3_to_next": [float(item) for item in conv3_to_next.detach().cpu().tolist()],
    }
    return conv1_to_conv2, conv2_to_conv3, conv3_to_next, source_values


def _bottleneck_middle_importance(model: nn.Module, block_name: str, block: nn.Module) -> tuple[torch.Tensor, Dict[str, List[float]]]:
    conv2_out = _resolve_module_importance(model, f"{block_name}.conv2", block.conv2)
    conv3_in = _conv_input_importance(block.conv3)
    if int(conv2_out.numel()) != int(conv3_in.numel()):
        raise ValueError(
            f"conv2 output and conv3 input importance lengths mismatch for {block_name}: "
            f"{int(conv2_out.numel())} vs {int(conv3_in.numel())}"
        )
    combined = torch.stack((_normalize_importance(conv2_out), _normalize_importance(conv3_in))).mean(dim=0)
    source_values = {
        "conv2_out": [float(item) for item in conv2_out.detach().cpu().tolist()],
        "conv3_in": [float(item) for item in conv3_in.detach().cpu().tolist()],
        "combined_conv2_to_conv3": [float(item) for item in combined.detach().cpu().tolist()],
    }
    return combined, source_values


def _first_conv_input_importance(module: Optional[nn.Module]) -> Optional[torch.Tensor]:
    if module is None:
        return None
    if isinstance(module, nn.Conv2d):
        return _conv_input_importance(module)
    for child in module.modules():
        if child is module:
            continue
        if isinstance(child, nn.Conv2d):
            return _conv_input_importance(child)
    return None


def _resolve_target_modules(teacher_model: nn.Module, target_modules: Optional[Sequence[str]]) -> Sequence[str]:
    if target_modules:
        return tuple(target_modules)

    named_modules = dict(teacher_model.named_modules())
    for candidate_group in MODULE_GROUP_CANDIDATES:
        if all(candidate in named_modules for candidate in candidate_group):
            return candidate_group
    return DEFAULT_BLUEPRINT_MODULES


def _tensor_stats(values: Sequence[float]) -> Dict[str, float]:
    tensor = torch.tensor(list(values), dtype=torch.float32)
    if tensor.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0,
    }


def _is_resnet_bottleneck_block(module: nn.Module) -> bool:
    return all(hasattr(module, name) for name in ("conv1", "bn1", "conv2", "bn2", "conv3", "bn3")) and isinstance(getattr(module, "conv2"), nn.Conv2d)


def _extract_middle_resnet_blueprint(
    teacher_model: nn.Module,
    *,
    prune_method: str,
    prune_ratio: float,
    static_prune_ratio: Optional[float],
    target_modules: Sequence[str],
    minimum_channels: int,
    dynamic_min_keep_ratio: float,
    otsu_bins: int,
    gmm_random_state: int,
) -> Dict[str, object]:
    named_modules = dict(teacher_model.named_modules())
    stage_names = [name for name in target_modules if _is_resnet_stage_module(name) and name in named_modules]
    if not stage_names:
        raise RuntimeError(f"{prune_method} requires ResNet stage modules layer1/layer2/layer3/layer4.")

    ratio_based_static = uses_static_prune_ratio(prune_method)
    selection_policy = (
        "topk_static_ratio_on_resnet_bottleneck_middle_conv"
        if ratio_based_static
        else f"{prune_method}_threshold_on_resnet_bottleneck_middle_conv"
    )
    effective_minimum_channels = 1 if ratio_based_static else int(minimum_channels)
    effective_min_keep_ratio = 0.0 if ratio_based_static else float(dynamic_min_keep_ratio)

    per_module: List[Dict[str, object]] = []
    teacher_channel_analysis: List[Dict[str, object]] = []
    teacher_vs_student_rows: List[Dict[str, object]] = []
    channel_level_detail: List[Dict[str, object]] = []
    middle_prune_plan: List[Dict[str, object]] = []
    stage_middle_channel_config: Dict[str, List[int]] = {}

    for stage_name in stage_names:
        stage = named_modules[stage_name]
        stage_middle_channel_config[stage_name] = []
        for child_name, block in stage.named_children():
            block_name = f"{stage_name}.{child_name}"
            if not _is_resnet_bottleneck_block(block):
                continue
            plot_index = len(middle_prune_plan) + 1

            middle_layer_name = f"{block_name}.conv2"
            importance, middle_importance_sources = _bottleneck_middle_importance(teacher_model, block_name, block)
            importance_np = importance.detach().cpu().numpy()
            prune_kwargs = {
                "scores": importance_np,
                "layer_name": middle_layer_name,
                "method": prune_method,
                "otsu_bins": otsu_bins,
                "gmm_random_state": gmm_random_state,
            }
            if ratio_based_static:
                prune_kwargs["static_prune_ratio"] = static_prune_ratio
            else:
                prune_kwargs["min_keep_ratio"] = dynamic_min_keep_ratio
                prune_kwargs["min_keep_channels"] = minimum_channels
            prune_result = prune_one_layer(
                **prune_kwargs,
            )
            ranked_indices = torch.argsort(importance, descending=True)
            rank_lookup = {int(index): rank + 1 for rank, index in enumerate(ranked_indices.tolist())}
            kept_indices = sorted(int(index) for index in prune_result.keep_indices.tolist())
            kept_index_set = set(kept_indices)
            pruned_indices = sorted(index for index in range(int(importance.numel())) if index not in kept_index_set)
            importance_values = [float(value) for value in importance.detach().cpu().tolist()]
            stats = _tensor_stats(importance_values)
            keep_channels = int(prune_result.num_keep)
            original_middle_channels = int(block.conv2.out_channels)
            actual_prune_ratio = float(prune_result.prune_ratio)

            stage_middle_channel_config[stage_name].append(keep_channels)
            middle_prune_plan.append(
                {
                    "stage_name": stage_name,
                    "block_name": block_name,
                    "pruning_group": "s5_s8_middle_conv2_block",
                    "plot_index": plot_index,
                    "plot_role": "conv2_output_to_conv3_input",
                    "middle_layer_name": middle_layer_name,
                    "first_layer_name": f"{block_name}.conv1",
                    "last_layer_name": f"{block_name}.conv3",
                    "original_middle_channels": original_middle_channels,
                    "kept_middle_channels": keep_channels,
                    "kept_channel_indices": kept_indices,
                    "pruned_channel_indices": pruned_indices,
                    "selection_policy": selection_policy,
                    "pruning_threshold": float(prune_result.threshold),
                    "prune_method": prune_method,
                    "protected_boundary_layers": [f"{block_name}.conv1", f"{block_name}.conv3"],
                    "importance_source": "mean_normalized_conv2_out_and_conv3_in",
                    "importance_sources": middle_importance_sources,
                }
            )

            teacher_channel_analysis.append(
                {
                    "layer_name": block_name,
                    "pruning_group": "s5_s8_middle_conv2_block",
                    "plot_index": plot_index,
                    "plot_role": "conv2_output_to_conv3_input",
                    "pruning_layer_name": middle_layer_name,
                    "channel_role": "conv2_output_to_conv3_input",
                    "module_type": type(block).__name__,
                    "channel_layer_type": type(block.conv2).__name__,
                    "in_channels": int(block.conv2.in_channels),
                    "teacher_out_channels": original_middle_channels,
                    "kernel_size": list(block.conv2.kernel_size),
                    "weight_shape": [int(value) for value in block.conv2.weight.shape],
                    "importance_source": "mean_normalized_conv2_out_and_conv3_in",
                    "importance_sources": middle_importance_sources,
                    "importance_values": importance_values,
                    "ranked_channel_indices_desc": [int(index) for index in ranked_indices.tolist()],
                    "kept_channel_indices": kept_indices,
                    "pruned_channel_indices": pruned_indices,
                    "importance_min": stats["min"],
                    "importance_max": stats["max"],
                    "importance_mean": stats["mean"],
                    "importance_std": stats["std"],
                    "prune_method": prune_method,
                    "selection_policy": selection_policy,
                    "pruning_threshold": float(prune_result.threshold),
                    "requested_static_prune_ratio": float(static_prune_ratio) if ratio_based_static else None,
                    "block_middle_pruning": True,
                    "block_middle_layer_found": True,
                    "protected_boundary_layers": [f"{block_name}.conv1", f"{block_name}.conv3"],
                    "candidate_channel_layers": [f"{block_name}.conv1", middle_layer_name, f"{block_name}.conv3"],
                }
            )

            row = {
                "layer_name": block_name,
                "module_name": block_name,
                "pruning_group": "s5_s8_middle_conv2_block",
                "plot_index": plot_index,
                "plot_role": "conv2_output_to_conv3_input",
                "pruning_layer_name": middle_layer_name,
                "channel_role": "conv2_output_to_conv3_input",
                "module_type": type(block).__name__,
                "channel_layer_type": type(block.conv2).__name__,
                "in_channels": int(block.conv2.in_channels),
                "source_out_channels": original_middle_channels,
                "teacher_out_channels": original_middle_channels,
                "student_out_channels": keep_channels,
                "kept_channels": keep_channels,
                "pruned_channels": int(len(pruned_indices)),
                "actual_prune_ratio": actual_prune_ratio,
                "criterion": "mean_normalized_conv2_out_and_conv3_in",
                "prune_method": prune_method,
                "selection_policy": selection_policy,
                "pruning_threshold": float(prune_result.threshold),
                "requested_static_prune_ratio": float(static_prune_ratio) if ratio_based_static else None,
                "block_middle_pruning": True,
                "block_middle_layer_found": True,
                "protected_boundary_layers": [f"{block_name}.conv1", f"{block_name}.conv3"],
                "candidate_channel_layers": [f"{block_name}.conv1", middle_layer_name, f"{block_name}.conv3"],
                "importance_min": stats["min"],
                "importance_max": stats["max"],
                "importance_mean": stats["mean"],
                "importance_std": stats["std"],
                "importance_source": "mean_normalized_conv2_out_and_conv3_in",
                "importance_sources": middle_importance_sources,
                "kept_channel_indices": kept_indices,
                "pruned_channel_indices": pruned_indices,
                "ranked_channel_indices_desc": [int(index) for index in ranked_indices.tolist()],
            }
            per_module.append(row)
            teacher_vs_student_rows.append(
                {
                    "layer_name": block_name,
                    "pruning_group": "s5_s8_middle_conv2_block",
                    "plot_index": plot_index,
                    "plot_role": "conv2_output_to_conv3_input",
                    "pruning_layer_name": middle_layer_name,
                    "teacher_out_channels": original_middle_channels,
                    "student_out_channels": keep_channels,
                    "channels_pruned": int(len(pruned_indices)),
                    "actual_prune_ratio": actual_prune_ratio,
                    "prune_method": prune_method,
                    "selection_policy": selection_policy,
                    "pruning_threshold": float(prune_result.threshold),
                }
            )
            for channel_index, importance_value in enumerate(importance_values):
                channel_level_detail.append(
                    {
                        "layer_name": block_name,
                        "pruning_group": "s5_s8_middle_conv2_block",
                        "plot_index": plot_index,
                        "plot_role": "conv2_output_to_conv3_input",
                        "pruning_layer_name": middle_layer_name,
                        "channel_role": "conv2_output_to_conv3_input",
                        "channel_index": int(channel_index),
                        "importance": float(importance_value),
                        "conv2_out_importance": middle_importance_sources["conv2_out"][channel_index],
                        "conv3_in_importance": middle_importance_sources["conv3_in"][channel_index],
                        "rank_desc": int(rank_lookup[channel_index]),
                        "decision": "keep" if channel_index in kept_index_set else "prune",
                        "prune_method": prune_method,
                        "selection_policy": selection_policy,
                        "pruning_threshold": float(prune_result.threshold),
                    }
                )

    if not middle_prune_plan:
        raise RuntimeError(f"Could not find ResNet bottleneck blocks for {prune_method} pruning.")

    total_before = int(sum(row["teacher_out_channels"] for row in teacher_vs_student_rows))
    total_after = int(sum(row["student_out_channels"] for row in teacher_vs_student_rows))
    total_pruned = int(total_before - total_after)
    changed_layers = int(sum(1 for row in teacher_vs_student_rows if row["channels_pruned"] > 0))
    global_pruning_summary = {
        "num_layers_analyzed": int(len(teacher_vs_student_rows)),
        "total_channels_before": total_before,
        "total_channels_after": total_after,
        "total_channels_pruned": total_pruned,
        "global_prune_ratio": float(total_pruned / max(1, total_before)),
        "num_layers_changed": changed_layers,
        "num_layers_unchanged": int(len(teacher_vs_student_rows) - changed_layers),
        "protected_boundary_policy": "conv1_out_and_conv3_out_are_kept_full; conv2_out and conv3_in are pruned with one shared combined mask",
    }

    return {
        "channel_config": tuple(int(row["kept_middle_channels"]) for row in middle_prune_plan),
        "stage_middle_channel_config": {stage: list(values) for stage, values in stage_middle_channel_config.items()},
        "middle_prune_plan": middle_prune_plan,
        "student_architecture": "middle_pruned_resnet_unet",
        "prune_method": prune_method,
        "prune_ratio": float(prune_ratio) if ratio_based_static else None,
        "static_prune_ratio": float(static_prune_ratio) if ratio_based_static else None,
        "block_middle_pruning": True,
        "target_modules": list(target_modules),
        "criterion": "mean_normalized_conv2_out_and_conv3_in",
        "minimum_channels": effective_minimum_channels,
        "dynamic_min_keep_ratio": effective_min_keep_ratio,
        "otsu_bins": int(otsu_bins),
        "gmm_random_state": int(gmm_random_state),
        "modules": per_module,
        "teacher_channel_analysis": teacher_channel_analysis,
        "pruning_summary_rows": per_module,
        "teacher_vs_student_rows": teacher_vs_student_rows,
        "channel_level_detail": channel_level_detail,
        "global_pruning_summary": global_pruning_summary,
    }


def _extract_full_resnet_blueprint(
    teacher_model: nn.Module,
    *,
    prune_method: str,
    prune_ratio: float,
    static_prune_ratio: Optional[float],
    target_modules: Sequence[str],
    minimum_channels: int,
    dynamic_min_keep_ratio: float,
    otsu_bins: int,
    gmm_random_state: int,
) -> Dict[str, object]:
    named_modules = dict(teacher_model.named_modules())
    stage_names = [name for name in target_modules if _is_resnet_stage_module(name) and name in named_modules]
    if not stage_names:
        raise RuntimeError(f"{prune_method} requires ResNet stage modules layer1/layer2/layer3/layer4.")

    ratio_based_static = uses_static_prune_ratio(prune_method)
    selection_policy = (
        "topk_static_ratio_on_resnet_bottleneck_full_block"
        if ratio_based_static
        else f"{prune_method}_threshold_on_resnet_bottleneck_full_block"
    )
    effective_minimum_channels = 1 if ratio_based_static else int(minimum_channels)
    effective_min_keep_ratio = 0.0 if ratio_based_static else float(dynamic_min_keep_ratio)

    per_module: List[Dict[str, object]] = []
    teacher_channel_analysis: List[Dict[str, object]] = []
    teacher_vs_student_rows: List[Dict[str, object]] = []
    channel_level_detail: List[Dict[str, object]] = []
    full_prune_plan: List[Dict[str, object]] = []
    stage_full_channel_config: Dict[str, List[int]] = {}
    stage_full_conv2_channel_config: Dict[str, List[int]] = {}
    stage_full_output_channel_config: Dict[str, List[int]] = {}
    stage_output_kept_indices: Dict[str, List[int]] = {}

    def run_channel_prune(scores: torch.Tensor, layer_name: str):
        prune_kwargs = {
            "scores": scores.detach().cpu().numpy(),
            "layer_name": layer_name,
            "method": prune_method,
            "otsu_bins": otsu_bins,
            "gmm_random_state": gmm_random_state,
        }
        if ratio_based_static:
            prune_kwargs["static_prune_ratio"] = static_prune_ratio
        else:
            prune_kwargs["min_keep_ratio"] = dynamic_min_keep_ratio
            prune_kwargs["min_keep_channels"] = minimum_channels
        prune_result = prune_one_layer(**prune_kwargs)
        ranked_indices = torch.argsort(scores, descending=True)
        kept_indices = sorted(int(index) for index in prune_result.keep_indices.tolist())
        kept_index_set = set(kept_indices)
        pruned_indices = sorted(index for index in range(int(scores.numel())) if index not in kept_index_set)
        values = [float(value) for value in scores.detach().cpu().tolist()]
        rank_lookup = {int(index): rank + 1 for rank, index in enumerate(ranked_indices.tolist())}
        return {
            "result": prune_result,
            "ranked_indices": [int(index) for index in ranked_indices.tolist()],
            "rank_lookup": rank_lookup,
            "kept_indices": kept_indices,
            "kept_index_set": kept_index_set,
            "pruned_indices": pruned_indices,
            "values": values,
            "stats": _tensor_stats(values),
        }

    ordered_blocks: List[tuple[str, nn.Module]] = []
    for stage_name in stage_names:
        stage = named_modules[stage_name]
        for child_name, block in stage.named_children():
            if _is_resnet_bottleneck_block(block):
                ordered_blocks.append((f"{stage_name}.{child_name}", block))
    next_input_by_block: Dict[str, tuple[Optional[torch.Tensor], str]] = {}
    for index, (block_name, block) in enumerate(ordered_blocks):
        if index + 1 < len(ordered_blocks):
            next_name, next_block = ordered_blocks[index + 1]
            next_input_by_block[block_name] = (_conv_input_importance(next_block.conv1), f"{next_name}.conv1_input")
        else:
            next_input = _first_conv_input_importance(named_modules.get("center"))
            next_input_by_block[block_name] = (next_input, "center_first_conv_input" if next_input is not None else "conv3_out_fallback")

    previous_stage_output_indices: Optional[List[int]] = None
    for stage_name in stage_names:
        stage = named_modules[stage_name]
        stage_full_channel_config[stage_name] = []
        stage_full_conv2_channel_config[stage_name] = []
        stage_full_output_channel_config[stage_name] = []
        current_input_indices = list(previous_stage_output_indices) if previous_stage_output_indices is not None else None
        for child_name, block in stage.named_children():
            block_name = f"{stage_name}.{child_name}"
            if not _is_resnet_bottleneck_block(block):
                continue
            plot_index = len(full_prune_plan) + 1

            if current_input_indices is None:
                current_input_indices = list(range(int(block.conv1.in_channels)))
            input_indices = list(current_input_indices)
            full_layer_name = f"{block_name}.full_block"
            internal_layer_name = f"{block_name}.internal_width"
            conv1_layer_name = f"{block_name}.conv1_to_conv2"
            conv2_layer_name = f"{block_name}.conv2_to_conv3"
            output_layer_name = f"{block_name}.conv3_output"
            next_input_importance, next_input_source = next_input_by_block.get(block_name, (None, "conv3_out_fallback"))
            conv1_importance, conv2_importance, output_importance, internal_importance_sources = _bottleneck_full_channel_importance(
                teacher_model,
                block_name,
                block,
                next_input_importance=next_input_importance,
                next_input_source=next_input_source,
            )
            internal_importance = torch.cat(
                (
                    _normalize_importance(conv1_importance),
                    _normalize_importance(conv2_importance),
                )
            )
            conv1_prune = run_channel_prune(conv1_importance, conv1_layer_name)
            conv2_prune = run_channel_prune(conv2_importance, conv2_layer_name)
            internal_prune = run_channel_prune(internal_importance, internal_layer_name)
            output_prune = run_channel_prune(output_importance, output_layer_name)

            conv1_keep_indices = conv1_prune["kept_indices"]
            conv1_pruned_indices = conv1_prune["pruned_indices"]
            conv2_keep_indices = conv2_prune["kept_indices"]
            conv2_pruned_indices = conv2_prune["pruned_indices"]
            internal_keep_indices = conv1_keep_indices
            internal_pruned_indices = conv1_pruned_indices
            output_keep_indices = output_prune["kept_indices"]
            output_selection_policy = selection_policy
            output_pruning_threshold = float(output_prune["result"].threshold)
            residual_safe_output_for_identity = bool(
                block.downsample is None
                and tuple(block.conv2.stride) == (1, 1)
                and len(input_indices) <= int(block.conv3.out_channels)
            )
            if residual_safe_output_for_identity:
                # Sequential block pruning: the current block may prune its
                # output further, but an identity residual can only keep
                # channels that are still present in its input skip tensor.
                candidate_output_indices = sorted(int(index) for index in input_indices)
                candidate_scores = output_importance.detach().cpu()[candidate_output_indices]
                constrained_output_prune = run_channel_prune(candidate_scores, f"{output_layer_name}.input_subset")
                output_keep_indices = sorted(candidate_output_indices[int(index)] for index in constrained_output_prune["kept_indices"])
                output_selection_policy = f"{selection_policy}_sequential_output_subset_of_input"
                output_pruning_threshold = float(constrained_output_prune["result"].threshold)
            output_kept_index_set = set(output_keep_indices)
            output_pruned_indices = sorted(index for index in range(int(output_importance.numel())) if index not in output_kept_index_set)
            keep_channels = int(len(conv1_keep_indices))
            keep_conv2_channels = int(len(conv2_keep_indices))
            keep_output_channels = int(len(output_keep_indices))
            original_internal_channels = int(block.conv2.out_channels)
            original_output_channels = int(block.conv3.out_channels)
            internal_actual_prune_ratio = float(len(conv1_pruned_indices) / max(1, original_internal_channels))
            conv2_actual_prune_ratio = float(len(conv2_pruned_indices) / max(1, original_internal_channels))
            output_actual_prune_ratio = float(len(output_pruned_indices) / max(1, original_output_channels))

            stage_full_channel_config[stage_name].append(keep_channels)
            stage_full_conv2_channel_config[stage_name].append(keep_conv2_channels)
            stage_full_output_channel_config[stage_name].append(keep_output_channels)
            plan_row = {
                "stage_name": stage_name,
                "block_name": block_name,
                "pruning_group": "s9_s12_full_block",
                "plot_index": plot_index,
                "plot_role": "conv3_output",
                "full_layer_name": full_layer_name,
                "first_layer_name": f"{block_name}.conv1",
                "middle_layer_name": f"{block_name}.conv2",
                "last_layer_name": f"{block_name}.conv3",
                "input_channel_indices": input_indices,
                "original_input_channels": int(block.conv1.in_channels),
                "kept_input_channels": int(len(input_indices)),
                "original_internal_channels": original_internal_channels,
                "kept_internal_channels": keep_channels,
                "internal_kept_channel_indices": internal_keep_indices,
                "internal_pruned_channel_indices": internal_pruned_indices,
                "conv1_kept_channel_indices": conv1_keep_indices,
                "conv1_pruned_channel_indices": conv1_pruned_indices,
                "conv1_pruning_threshold": float(conv1_prune["result"].threshold),
                "conv1_actual_prune_ratio": internal_actual_prune_ratio,
                "conv2_kept_channel_indices": conv2_keep_indices,
                "conv2_pruned_channel_indices": conv2_pruned_indices,
                "conv2_pruning_threshold": float(conv2_prune["result"].threshold),
                "conv2_actual_prune_ratio": conv2_actual_prune_ratio,
                "kept_conv2_channels": keep_conv2_channels,
                "internal_pruning_threshold": float(conv1_prune["result"].threshold),
                "internal_actual_prune_ratio": internal_actual_prune_ratio,
                "internal_importance_source": "conv1_out_plus_conv2_in__conv2_out_plus_conv3_in__conv3_out_plus_next_input_from_teacher",
                "internal_importance_sources": internal_importance_sources,
                "original_output_channels": original_output_channels,
                "kept_output_channels": keep_output_channels,
                "output_kept_channel_indices": output_keep_indices,
                "output_pruned_channel_indices": output_pruned_indices,
                "output_pruning_threshold": output_pruning_threshold,
                "output_actual_prune_ratio": output_actual_prune_ratio,
                "kept_channel_indices": output_keep_indices,
                "pruned_channel_indices": output_pruned_indices,
                "selection_policy": output_selection_policy,
                "pruning_threshold": output_pruning_threshold,
                "prune_method": prune_method,
                "residual_projection_required": bool(
                    block.downsample is not None
                    or len(input_indices) != keep_output_channels
                    or input_indices != output_keep_indices
                    or tuple(block.conv2.stride) != (1, 1)
                ),
                "protected_boundary_layers": ["spatial_stride", "residual_add_semantics"],
                "pruned_components": ["conv1_in", "conv1_out", "bn1", "conv2_in", "conv2_out", "bn2", "conv3_in", "conv3_out", "bn3", "downsample_out"],
            }
            full_prune_plan.append(plan_row)

            row = {
                "layer_name": block_name,
                "module_name": block_name,
                "pruning_group": "s9_s12_full_block",
                "plot_index": plot_index,
                "plot_role": "conv3_output",
                "pruning_layer_name": full_layer_name,
                "channel_role": "conv3_output",
                "module_type": type(block).__name__,
                "channel_layer_type": "BottleneckFullOutput",
                "in_channels": int(block.conv1.in_channels),
                "source_out_channels": original_output_channels,
                "teacher_out_channels": original_output_channels,
                "student_out_channels": keep_output_channels,
                "kept_channels": keep_output_channels,
                "pruned_channels": int(len(output_pruned_indices)),
                "actual_prune_ratio": output_actual_prune_ratio,
                "internal_teacher_channels": original_internal_channels,
                "internal_student_channels": keep_channels,
                "conv2_student_channels": keep_conv2_channels,
                "internal_prune_ratio": internal_actual_prune_ratio,
                "conv2_prune_ratio": conv2_actual_prune_ratio,
                "criterion": "mean_normalized_output_and_next_input_per_pruned_edge",
                "prune_method": prune_method,
                "selection_policy": output_selection_policy,
                "pruning_threshold": output_pruning_threshold,
                "internal_pruning_threshold": float(conv1_prune["result"].threshold),
                "conv1_pruning_threshold": float(conv1_prune["result"].threshold),
                "conv2_pruning_threshold": float(conv2_prune["result"].threshold),
                "requested_static_prune_ratio": float(static_prune_ratio) if ratio_based_static else None,
                "block_full_pruning": True,
                "prunes_conv3_output": True,
                "protected_boundary_layers": ["spatial_stride", "residual_add_semantics"],
                "candidate_channel_layers": [f"{block_name}.conv1", f"{block_name}.conv2", f"{block_name}.conv3"],
                "importance_min": output_prune["stats"]["min"],
                "importance_max": output_prune["stats"]["max"],
                "importance_mean": output_prune["stats"]["mean"],
                "importance_std": output_prune["stats"]["std"],
                "internal_importance_min": conv1_prune["stats"]["min"],
                "internal_importance_max": conv1_prune["stats"]["max"],
                "internal_importance_mean": conv1_prune["stats"]["mean"],
                "internal_importance_std": conv1_prune["stats"]["std"],
                "conv2_importance_min": conv2_prune["stats"]["min"],
                "conv2_importance_max": conv2_prune["stats"]["max"],
                "conv2_importance_mean": conv2_prune["stats"]["mean"],
                "conv2_importance_std": conv2_prune["stats"]["std"],
                "internal_importance_source": "conv1_out_plus_conv2_in__conv2_out_plus_conv3_in__conv3_out_plus_next_input_from_teacher",
                "internal_importance_sources": internal_importance_sources,
                "input_channel_indices": input_indices,
                "internal_kept_channel_indices": internal_keep_indices,
                "conv1_kept_channel_indices": conv1_keep_indices,
                "conv2_kept_channel_indices": conv2_keep_indices,
                "output_kept_channel_indices": output_keep_indices,
                "kept_channel_indices": output_keep_indices,
                "pruned_channel_indices": output_pruned_indices,
                "ranked_channel_indices_desc": output_prune["ranked_indices"],
            }
            per_module.append(row)
            teacher_channel_analysis.append({**row, "importance_source": "mean_normalized_conv3_out_and_next_input", "importance_values": output_prune["values"]})
            teacher_channel_analysis.append(
                {
                    **row,
                    "pruning_layer_name": internal_layer_name,
                    "channel_role": "internal_width",
                    "importance_source": "mean_normalized_conv1_out_and_conv2_in",
                    "importance_values": conv1_prune["values"],
                    "ranked_channel_indices_desc": conv1_prune["ranked_indices"],
                    "kept_channel_indices": conv1_keep_indices,
                    "pruned_channel_indices": conv1_pruned_indices,
                    "pruning_threshold": float(conv1_prune["result"].threshold),
                    "importance_min": conv1_prune["stats"]["min"],
                    "importance_max": conv1_prune["stats"]["max"],
                    "importance_mean": conv1_prune["stats"]["mean"],
                    "importance_std": conv1_prune["stats"]["std"],
                }
            )
            teacher_channel_analysis.append(
                {
                    **row,
                    "pruning_layer_name": conv2_layer_name,
                    "channel_role": "conv2_output",
                    "importance_source": "mean_normalized_conv2_out_and_conv3_in",
                    "importance_values": conv2_prune["values"],
                    "ranked_channel_indices_desc": conv2_prune["ranked_indices"],
                    "kept_channel_indices": conv2_keep_indices,
                    "pruned_channel_indices": conv2_pruned_indices,
                    "pruning_threshold": float(conv2_prune["result"].threshold),
                    "importance_min": conv2_prune["stats"]["min"],
                    "importance_max": conv2_prune["stats"]["max"],
                    "importance_mean": conv2_prune["stats"]["mean"],
                    "importance_std": conv2_prune["stats"]["std"],
                }
            )
            teacher_vs_student_rows.append(
                {
                    "layer_name": block_name,
                    "pruning_group": "s9_s12_full_block",
                    "plot_index": plot_index,
                    "plot_role": "conv3_output",
                    "pruning_layer_name": full_layer_name,
                    "teacher_out_channels": original_output_channels,
                    "student_out_channels": keep_output_channels,
                    "channels_pruned": int(len(output_pruned_indices)),
                    "actual_prune_ratio": output_actual_prune_ratio,
                    "teacher_internal_channels": original_internal_channels,
                    "student_internal_channels": keep_channels,
                    "student_conv2_channels": keep_conv2_channels,
                    "internal_prune_ratio": internal_actual_prune_ratio,
                    "conv2_prune_ratio": conv2_actual_prune_ratio,
                    "prune_method": prune_method,
                    "selection_policy": output_selection_policy,
                    "pruning_threshold": output_pruning_threshold,
                    "internal_pruning_threshold": float(conv1_prune["result"].threshold),
                    "conv2_pruning_threshold": float(conv2_prune["result"].threshold),
                }
            )
            for channel_index, importance_value in enumerate(output_prune["values"]):
                channel_level_detail.append(
                    {
                        "layer_name": block_name,
                        "pruning_group": "s9_s12_full_block",
                        "plot_index": plot_index,
                        "plot_role": "conv3_output",
                        "pruning_layer_name": output_layer_name,
                        "channel_role": "conv3_output",
                        "channel_index": int(channel_index),
                        "importance": float(importance_value),
                        "rank_desc": int(output_prune["rank_lookup"][channel_index]),
                        "decision": "keep" if channel_index in output_kept_index_set else "prune",
                        "prune_method": prune_method,
                        "selection_policy": output_selection_policy,
                        "pruning_threshold": output_pruning_threshold,
                        "conv3_out_importance": internal_importance_sources["conv3_out"][channel_index],
                        "conv3_next_in_importance": internal_importance_sources["conv3_next_in"][channel_index],
                        "conv3_next_in_source": internal_importance_sources.get("conv3_next_in_source", "conv3_out_fallback"),
                    }
                )
            for channel_index, importance_value in enumerate(internal_prune["values"]):
                channel_level_detail.append(
                    {
                        "layer_name": block_name,
                        "pruning_group": "s9_s12_full_block",
                        "plot_index": plot_index,
                        "plot_role": "internal_combined_summary",
                        "pruning_layer_name": internal_layer_name,
                        "channel_role": "internal_combined_summary",
                        "channel_index": int(channel_index),
                        "importance": float(importance_value),
                        "rank_desc": int(internal_prune["rank_lookup"][channel_index]),
                        "decision": "summary",
                        "prune_method": prune_method,
                        "selection_policy": selection_policy,
                        "pruning_threshold": float(internal_prune["result"].threshold),
                    }
                )
            for channel_index, importance_value in enumerate(conv1_prune["values"]):
                channel_level_detail.append(
                    {
                        "layer_name": block_name,
                        "pruning_group": "s9_s12_full_block",
                        "plot_index": plot_index,
                        "plot_role": "conv1_output",
                        "pruning_layer_name": conv1_layer_name,
                        "channel_role": "conv1_output",
                        "channel_index": int(channel_index),
                        "importance": float(importance_value),
                        "rank_desc": int(conv1_prune["rank_lookup"][channel_index]),
                        "decision": "keep" if channel_index in conv1_prune["kept_index_set"] else "prune",
                        "prune_method": prune_method,
                        "selection_policy": selection_policy,
                        "pruning_threshold": float(conv1_prune["result"].threshold),
                        "conv1_out_importance": internal_importance_sources["conv1_out"][channel_index],
                        "conv2_in_importance": internal_importance_sources["conv2_in"][channel_index],
                    }
                )
            for channel_index, importance_value in enumerate(conv2_prune["values"]):
                channel_level_detail.append(
                    {
                        "layer_name": block_name,
                        "pruning_group": "s9_s12_full_block",
                        "plot_index": plot_index,
                        "plot_role": "conv2_output",
                        "pruning_layer_name": conv2_layer_name,
                        "channel_role": "conv2_output",
                        "channel_index": int(channel_index),
                        "importance": float(importance_value),
                        "rank_desc": int(conv2_prune["rank_lookup"][channel_index]),
                        "decision": "keep" if channel_index in conv2_prune["kept_index_set"] else "prune",
                        "prune_method": prune_method,
                        "selection_policy": selection_policy,
                        "pruning_threshold": float(conv2_prune["result"].threshold),
                        "conv2_out_importance": internal_importance_sources["conv2_out"][channel_index],
                        "conv3_in_importance": internal_importance_sources["conv3_in"][channel_index],
                    }
                )
            current_input_indices = output_keep_indices
        if current_input_indices is not None:
            stage_output_kept_indices[stage_name] = list(current_input_indices)
            previous_stage_output_indices = list(current_input_indices)

    if not full_prune_plan:
        raise RuntimeError(f"Could not find ResNet bottleneck blocks for {prune_method} pruning.")

    total_before = int(sum(row["teacher_out_channels"] for row in teacher_vs_student_rows))
    total_after = int(sum(row["student_out_channels"] for row in teacher_vs_student_rows))
    internal_total_before = int(sum(row["teacher_internal_channels"] for row in teacher_vs_student_rows))
    internal_total_after = int(sum(row["student_internal_channels"] for row in teacher_vs_student_rows))
    conv2_total_after = int(sum(row.get("student_conv2_channels", row["student_internal_channels"]) for row in teacher_vs_student_rows))
    total_pruned = int(total_before - total_after)
    internal_total_pruned = int(internal_total_before - internal_total_after)
    conv2_total_pruned = int(internal_total_before - conv2_total_after)
    combined_total_before = int(total_before + internal_total_before + internal_total_before)
    combined_total_after = int(total_after + internal_total_after + conv2_total_after)
    combined_total_pruned = int(total_pruned + internal_total_pruned + conv2_total_pruned)
    changed_layers = int(sum(1 for row in teacher_vs_student_rows if row["channels_pruned"] > 0))
    global_pruning_summary = {
        "num_layers_analyzed": int(len(teacher_vs_student_rows)),
        "total_channels_before": total_before,
        "total_channels_after": total_after,
        "total_channels_pruned": total_pruned,
        "global_prune_ratio": float(total_pruned / max(1, total_before)),
        "output_channels_before": total_before,
        "output_channels_after": total_after,
        "output_channels_pruned": total_pruned,
        "output_global_prune_ratio": float(total_pruned / max(1, total_before)),
        "internal_channels_before": internal_total_before,
        "internal_channels_after": internal_total_after,
        "internal_channels_pruned": internal_total_pruned,
        "internal_global_prune_ratio": float(internal_total_pruned / max(1, internal_total_before)),
        "conv2_channels_before": internal_total_before,
        "conv2_channels_after": conv2_total_after,
        "conv2_channels_pruned": conv2_total_pruned,
        "conv2_global_prune_ratio": float(conv2_total_pruned / max(1, internal_total_before)),
        "combined_channels_before": combined_total_before,
        "combined_channels_after": combined_total_after,
        "combined_channels_pruned": combined_total_pruned,
        "combined_global_prune_ratio": float(combined_total_pruned / max(1, combined_total_before)),
        "num_layers_changed": changed_layers,
        "num_layers_unchanged": int(len(teacher_vs_student_rows) - changed_layers),
        "protected_boundary_policy": "conv3 output is pruned; residual projections are inserted or subset-copied whenever block input/output channels differ",
    }

    return {
        "channel_config": tuple(int(row["kept_output_channels"]) for row in full_prune_plan),
        "stage_full_channel_config": {stage: list(values) for stage, values in stage_full_channel_config.items()},
        "stage_full_conv2_channel_config": {stage: list(values) for stage, values in stage_full_conv2_channel_config.items()},
        "stage_full_output_channel_config": {stage: list(values) for stage, values in stage_full_output_channel_config.items()},
        "stage_output_kept_indices": {stage: list(values) for stage, values in stage_output_kept_indices.items()},
        "full_prune_plan": full_prune_plan,
        "student_architecture": "full_pruning_resnet_unet",
        "prune_method": prune_method,
        "prune_ratio": float(prune_ratio) if ratio_based_static else None,
        "static_prune_ratio": float(static_prune_ratio) if ratio_based_static else None,
        "block_full_pruning": True,
        "prunes_conv3_output": True,
        "target_modules": list(target_modules),
        "criterion": "mean_normalized_output_and_next_input_per_pruned_edge",
        "minimum_channels": effective_minimum_channels,
        "dynamic_min_keep_ratio": effective_min_keep_ratio,
        "otsu_bins": int(otsu_bins),
        "gmm_random_state": int(gmm_random_state),
        "modules": per_module,
        "teacher_channel_analysis": teacher_channel_analysis,
        "pruning_summary_rows": per_module,
        "teacher_vs_student_rows": teacher_vs_student_rows,
        "channel_level_detail": channel_level_detail,
        "global_pruning_summary": global_pruning_summary,
    }


def extract_pruned_blueprint(
    teacher_model: nn.Module,
    *,
    prune_ratio: float = 0.5,
    prune_method: str = "static",
    static_prune_ratio: Optional[float] = None,
    target_modules: Optional[Sequence[str]] = None,
    minimum_channels: int = 16,
    dynamic_min_keep_ratio: float = 0.4,
    otsu_bins: int = 256,
    gmm_random_state: int = 42,
) -> Dict[str, object]:
    prune_method = str(prune_method).lower()
    if prune_method not in PRUNE_METHODS:
        raise ValueError(f"Unsupported prune_method: {prune_method}. Expected one of {PRUNE_METHODS}.")

    ratio_based_static = uses_static_prune_ratio(prune_method)

    if ratio_based_static:
        if static_prune_ratio is None:
            static_prune_ratio = prune_ratio
        if not 0.0 <= float(static_prune_ratio) < 1.0:
            raise ValueError("static_prune_ratio must be in [0, 1).")
        prune_ratio = float(static_prune_ratio)
    elif prune_ratio is not None and not 0.0 <= float(prune_ratio) < 1.0:
        raise ValueError("prune_ratio must be in [0, 1).")

    target_modules = tuple(_resolve_target_modules(teacher_model, target_modules))
    if is_middle_resnet_pruning(prune_method):
        return _extract_middle_resnet_blueprint(
            teacher_model,
            prune_method=prune_method,
            prune_ratio=prune_ratio,
            static_prune_ratio=float(static_prune_ratio) if ratio_based_static else None,
            target_modules=target_modules,
            minimum_channels=minimum_channels,
            dynamic_min_keep_ratio=dynamic_min_keep_ratio,
            otsu_bins=otsu_bins,
            gmm_random_state=gmm_random_state,
        )
    if is_full_resnet_pruning(prune_method):
        return _extract_full_resnet_blueprint(
            teacher_model,
            prune_method=prune_method,
            prune_ratio=prune_ratio,
            static_prune_ratio=float(static_prune_ratio) if ratio_based_static else None,
            target_modules=target_modules,
            minimum_channels=minimum_channels,
            dynamic_min_keep_ratio=dynamic_min_keep_ratio,
            otsu_bins=otsu_bins,
            gmm_random_state=gmm_random_state,
        )

    blueprint_channels: List[int] = []
    per_module: List[Dict[str, object]] = []
    teacher_channel_analysis: List[Dict[str, object]] = []
    teacher_vs_student_rows: List[Dict[str, object]] = []
    channel_level_detail: List[Dict[str, object]] = []

    named_modules = dict(teacher_model.named_modules())
    for module_index, module_name in enumerate(target_modules, start=1):
        module = named_modules.get(module_name)
        if module is None:
            continue

        conv_layer = find_primary_channel_layer(module)
        if conv_layer is None or conv_layer.out_channels <= minimum_channels:
            continue

        pruning_layer = conv_layer
        pruning_layer_name = module_name
        importance = _resolve_module_importance(teacher_model, pruning_layer_name, pruning_layer)
        importance_np = importance.detach().cpu().numpy()
        prune_result = prune_one_layer(
            importance_np,
            layer_name=pruning_layer_name,
            method=prune_method,
            min_keep_ratio=dynamic_min_keep_ratio,
            min_keep_channels=None if ratio_based_static else minimum_channels,
            static_prune_ratio=static_prune_ratio,
            otsu_bins=otsu_bins,
            gmm_random_state=gmm_random_state,
        )
        ranked_indices = torch.argsort(importance, descending=True)
        rank_lookup = {int(index): rank + 1 for rank, index in enumerate(ranked_indices.tolist())}
        keep_channels = int(prune_result.num_keep)
        kept_indices = sorted(int(index) for index in prune_result.keep_indices.tolist())
        kept_index_set = set(kept_indices)
        pruned_indices = sorted(index for index in range(int(importance.numel())) if index not in kept_index_set)
        importance_values = [float(value) for value in importance.detach().cpu().tolist()]
        stats = _tensor_stats(importance_values)
        actual_prune_ratio = float(prune_result.prune_ratio)
        blueprint_channels.append(keep_channels)
        selection_policy = "topk_static_ratio" if prune_method == "static" else f"{prune_method}_threshold"
        importance_source = "bn_weight_or_l1"

        teacher_channel_analysis.append(
            {
                "layer_name": module_name,
                "pruning_group": "s1_s4_blueprint_stage",
                "plot_index": module_index,
                "plot_role": "stage_output",
                "pruning_layer_name": pruning_layer_name,
                "channel_role": "stage_output",
                "module_type": type(module).__name__,
                "channel_layer_type": type(pruning_layer).__name__,
                "in_channels": int(getattr(pruning_layer, "in_channels", 0)),
                "teacher_out_channels": int(pruning_layer.out_channels),
                "kernel_size": list(pruning_layer.kernel_size) if hasattr(pruning_layer, "kernel_size") else None,
                "weight_shape": [int(value) for value in pruning_layer.weight.shape],
                "importance_source": importance_source,
                "importance_values": importance_values,
                "ranked_channel_indices_desc": [int(index) for index in ranked_indices.tolist()],
                "kept_channel_indices": kept_indices,
                "pruned_channel_indices": pruned_indices,
                "importance_min": stats["min"],
                "importance_max": stats["max"],
                "importance_mean": stats["mean"],
                "importance_std": stats["std"],
                "prune_method": prune_method,
                "selection_policy": selection_policy,
                "pruning_threshold": float(prune_result.threshold),
                "requested_static_prune_ratio": float(static_prune_ratio) if ratio_based_static else None,
                "block_middle_pruning": False,
                "block_middle_layer_found": False,
                "protected_boundary_layers": [],
                "candidate_channel_layers": [],
            }
        )

        per_module.append(
            {
                "layer_name": module_name,
                "module_name": module_name,
                "pruning_group": "s1_s4_blueprint_stage",
                "plot_index": module_index,
                "plot_role": "stage_output",
                "pruning_layer_name": pruning_layer_name,
                "channel_role": "stage_output",
                "module_type": type(module).__name__,
                "channel_layer_type": type(pruning_layer).__name__,
                "in_channels": int(getattr(pruning_layer, "in_channels", 0)),
                "source_out_channels": int(pruning_layer.out_channels),
                "teacher_out_channels": int(pruning_layer.out_channels),
                "student_out_channels": int(keep_channels),
                "kept_channels": int(keep_channels),
                "pruned_channels": int(len(pruned_indices)),
                "actual_prune_ratio": actual_prune_ratio,
                "criterion": importance_source,
                "prune_method": prune_method,
                "selection_policy": selection_policy,
                "pruning_threshold": float(prune_result.threshold),
                "requested_static_prune_ratio": float(static_prune_ratio) if ratio_based_static else None,
                "block_middle_pruning": False,
                "block_middle_layer_found": False,
                "protected_boundary_layers": [],
                "candidate_channel_layers": [],
                "importance_min": stats["min"],
                "importance_max": stats["max"],
                "importance_mean": stats["mean"],
                "importance_std": stats["std"],
                "kept_channel_indices": kept_indices,
                "pruned_channel_indices": pruned_indices,
                "ranked_channel_indices_desc": [int(index) for index in ranked_indices.tolist()],
            }
        )
        teacher_vs_student_rows.append(
            {
                "layer_name": module_name,
                "pruning_group": "s1_s4_blueprint_stage",
                "plot_index": module_index,
                "plot_role": "stage_output",
                "pruning_layer_name": pruning_layer_name,
                "teacher_out_channels": int(pruning_layer.out_channels),
                "student_out_channels": int(keep_channels),
                "channels_pruned": int(len(pruned_indices)),
                "actual_prune_ratio": actual_prune_ratio,
                "prune_method": prune_method,
                "selection_policy": selection_policy,
                "pruning_threshold": float(prune_result.threshold),
            }
        )
        for channel_index, importance_value in enumerate(importance_values):
            channel_level_detail.append(
                {
                    "layer_name": module_name,
                    "pruning_group": "s1_s4_blueprint_stage",
                    "plot_index": module_index,
                    "plot_role": "stage_output",
                    "pruning_layer_name": pruning_layer_name,
                    "channel_role": "stage_output",
                    "channel_index": int(channel_index),
                    "importance": float(importance_value),
                    "rank_desc": int(rank_lookup[channel_index]),
                    "decision": "keep" if channel_index in kept_indices else "prune",
                    "prune_method": prune_method,
                    "selection_policy": selection_policy,
                    "pruning_threshold": float(prune_result.threshold),
                }
            )

    if len(blueprint_channels) != len(DEFAULT_BLUEPRINT_MODULES):
        fallback = []
        for module in teacher_model.modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > minimum_channels:
                importance = module.weight.detach().abs().sum(dim=(1, 2, 3)).cpu().numpy()
                result = prune_one_layer(
                    importance,
                    layer_name=f"fallback_conv_{len(fallback)}",
                    method=prune_method,
                    min_keep_ratio=dynamic_min_keep_ratio,
                    min_keep_channels=None if ratio_based_static else minimum_channels,
                    static_prune_ratio=static_prune_ratio,
                    otsu_bins=otsu_bins,
                    gmm_random_state=gmm_random_state,
                )
                fallback.append(int(result.num_keep))
            if len(fallback) == len(DEFAULT_BLUEPRINT_MODULES):
                break
        blueprint_channels = fallback or blueprint_channels

    if len(blueprint_channels) != len(DEFAULT_BLUEPRINT_MODULES):
        raise RuntimeError(
            "Could not derive a 5-stage pruning blueprint from the teacher. "
            "Provide explicit target_modules or align the teacher architecture with the pruning pipeline."
        )

    total_before = int(sum(row["teacher_out_channels"] for row in teacher_vs_student_rows))
    total_after = int(sum(row["student_out_channels"] for row in teacher_vs_student_rows))
    total_pruned = int(total_before - total_after)
    changed_layers = int(sum(1 for row in teacher_vs_student_rows if row["channels_pruned"] > 0))
    global_pruning_summary = {
        "num_layers_analyzed": int(len(teacher_vs_student_rows)),
        "total_channels_before": total_before,
        "total_channels_after": total_after,
        "total_channels_pruned": total_pruned,
        "global_prune_ratio": float(total_pruned / max(1, total_before)),
        "num_layers_changed": changed_layers,
        "num_layers_unchanged": int(len(teacher_vs_student_rows) - changed_layers),
    }

    return {
        "channel_config": tuple(int(channel) for channel in blueprint_channels[: len(DEFAULT_BLUEPRINT_MODULES)]),
        "prune_method": prune_method,
        "prune_ratio": float(prune_ratio) if ratio_based_static else None,
        "static_prune_ratio": float(static_prune_ratio) if ratio_based_static else None,
        "block_middle_pruning": bool(is_middle_static_pruning(prune_method)),
        "target_modules": list(target_modules),
        "criterion": "bn_weight_or_l1",
        "minimum_channels": int(minimum_channels),
        "dynamic_min_keep_ratio": float(dynamic_min_keep_ratio),
        "otsu_bins": int(otsu_bins),
        "gmm_random_state": int(gmm_random_state),
        "modules": per_module,
        "teacher_channel_analysis": teacher_channel_analysis,
        "pruning_summary_rows": per_module,
        "teacher_vs_student_rows": teacher_vs_student_rows,
        "channel_level_detail": channel_level_detail,
        "global_pruning_summary": global_pruning_summary,
    }


def save_blueprint_artifact(blueprint: Dict[str, object], output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    blueprint_path = output_dir / "blueprint.json"
    with blueprint_path.open("w", encoding="utf-8") as file:
        json.dump(blueprint, file, indent=2)
    return blueprint_path


def load_blueprint_artifact(blueprint_path: Path | str) -> Dict[str, object]:
    blueprint_path = Path(blueprint_path)
    with blueprint_path.open("r", encoding="utf-8") as file:
        blueprint = json.load(file)
    if "channel_config" in blueprint:
        blueprint["channel_config"] = tuple(int(channel) for channel in blueprint["channel_config"])
    return blueprint
