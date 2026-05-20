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

            middle_layer_name = f"{block_name}.conv2"
            importance = _resolve_module_importance(teacher_model, middle_layer_name, block.conv2)
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
                }
            )

            teacher_channel_analysis.append(
                {
                    "layer_name": block_name,
                    "pruning_layer_name": middle_layer_name,
                    "module_type": type(block).__name__,
                    "channel_layer_type": type(block.conv2).__name__,
                    "in_channels": int(block.conv2.in_channels),
                    "teacher_out_channels": original_middle_channels,
                    "kernel_size": list(block.conv2.kernel_size),
                    "weight_shape": [int(value) for value in block.conv2.weight.shape],
                    "importance_source": "bn2_weight_or_conv2_l1",
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
                "pruning_layer_name": middle_layer_name,
                "module_type": type(block).__name__,
                "channel_layer_type": type(block.conv2).__name__,
                "in_channels": int(block.conv2.in_channels),
                "source_out_channels": original_middle_channels,
                "teacher_out_channels": original_middle_channels,
                "student_out_channels": keep_channels,
                "kept_channels": keep_channels,
                "pruned_channels": int(len(pruned_indices)),
                "actual_prune_ratio": actual_prune_ratio,
                "criterion": "bn2_weight_or_conv2_l1",
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
                "kept_channel_indices": kept_indices,
                "pruned_channel_indices": pruned_indices,
                "ranked_channel_indices_desc": [int(index) for index in ranked_indices.tolist()],
            }
            per_module.append(row)
            teacher_vs_student_rows.append(
                {
                    "layer_name": block_name,
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
                        "pruning_layer_name": middle_layer_name,
                        "channel_index": int(channel_index),
                        "importance": float(importance_value),
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
        "protected_boundary_policy": "conv1_out_and_conv3_out_are_kept_full; only conv2_out/bn2 and conv3_in are pruned",
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
        "criterion": "bn2_weight_or_conv2_l1",
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

    previous_stage_output_indices: Optional[List[int]] = None
    for stage_name in stage_names:
        stage = named_modules[stage_name]
        stage_full_channel_config[stage_name] = []
        stage_full_output_channel_config[stage_name] = []
        current_input_indices = list(previous_stage_output_indices) if previous_stage_output_indices is not None else None
        for child_name, block in stage.named_children():
            block_name = f"{stage_name}.{child_name}"
            if not _is_resnet_bottleneck_block(block):
                continue

            if current_input_indices is None:
                current_input_indices = list(range(int(block.conv1.in_channels)))
            input_indices = list(current_input_indices)
            full_layer_name = f"{block_name}.full_block"
            internal_layer_name = f"{block_name}.internal_width"
            output_layer_name = f"{block_name}.conv3_output"
            internal_importance = _resolve_module_importance(teacher_model, f"{block_name}.conv2", block.conv2)
            output_importance = _resolve_module_importance(teacher_model, f"{block_name}.conv3", block.conv3)
            internal_prune = run_channel_prune(internal_importance, internal_layer_name)
            output_prune = run_channel_prune(output_importance, output_layer_name)

            internal_keep_indices = internal_prune["kept_indices"]
            internal_pruned_indices = internal_prune["pruned_indices"]
            output_keep_indices = output_prune["kept_indices"]
            output_pruned_indices = output_prune["pruned_indices"]
            keep_channels = int(len(internal_keep_indices))
            keep_output_channels = int(len(output_keep_indices))
            original_internal_channels = int(block.conv2.out_channels)
            original_output_channels = int(block.conv3.out_channels)
            internal_actual_prune_ratio = float(internal_prune["result"].prune_ratio)
            output_actual_prune_ratio = float(output_prune["result"].prune_ratio)

            stage_full_channel_config[stage_name].append(keep_channels)
            stage_full_output_channel_config[stage_name].append(keep_output_channels)
            plan_row = {
                "stage_name": stage_name,
                "block_name": block_name,
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
                "internal_pruning_threshold": float(internal_prune["result"].threshold),
                "internal_actual_prune_ratio": internal_actual_prune_ratio,
                "original_output_channels": original_output_channels,
                "kept_output_channels": keep_output_channels,
                "output_kept_channel_indices": output_keep_indices,
                "output_pruned_channel_indices": output_pruned_indices,
                "output_pruning_threshold": float(output_prune["result"].threshold),
                "output_actual_prune_ratio": output_actual_prune_ratio,
                "kept_channel_indices": output_keep_indices,
                "pruned_channel_indices": output_pruned_indices,
                "selection_policy": selection_policy,
                "pruning_threshold": float(output_prune["result"].threshold),
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
                "pruning_layer_name": full_layer_name,
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
                "internal_prune_ratio": internal_actual_prune_ratio,
                "criterion": "bn2/bn3_weight_or_conv_l1",
                "prune_method": prune_method,
                "selection_policy": selection_policy,
                "pruning_threshold": float(output_prune["result"].threshold),
                "internal_pruning_threshold": float(internal_prune["result"].threshold),
                "requested_static_prune_ratio": float(static_prune_ratio) if ratio_based_static else None,
                "block_full_pruning": True,
                "prunes_conv3_output": True,
                "protected_boundary_layers": ["spatial_stride", "residual_add_semantics"],
                "candidate_channel_layers": [f"{block_name}.conv1", f"{block_name}.conv2", f"{block_name}.conv3"],
                "importance_min": output_prune["stats"]["min"],
                "importance_max": output_prune["stats"]["max"],
                "importance_mean": output_prune["stats"]["mean"],
                "importance_std": output_prune["stats"]["std"],
                "internal_importance_min": internal_prune["stats"]["min"],
                "internal_importance_max": internal_prune["stats"]["max"],
                "internal_importance_mean": internal_prune["stats"]["mean"],
                "internal_importance_std": internal_prune["stats"]["std"],
                "input_channel_indices": input_indices,
                "internal_kept_channel_indices": internal_keep_indices,
                "output_kept_channel_indices": output_keep_indices,
                "kept_channel_indices": output_keep_indices,
                "pruned_channel_indices": output_pruned_indices,
                "ranked_channel_indices_desc": output_prune["ranked_indices"],
            }
            per_module.append(row)
            teacher_channel_analysis.append({**row, "importance_source": "bn3_weight_or_conv3_l1", "importance_values": output_prune["values"]})
            teacher_vs_student_rows.append(
                {
                    "layer_name": block_name,
                    "pruning_layer_name": full_layer_name,
                    "teacher_out_channels": original_output_channels,
                    "student_out_channels": keep_output_channels,
                    "channels_pruned": int(len(output_pruned_indices)),
                    "actual_prune_ratio": output_actual_prune_ratio,
                    "teacher_internal_channels": original_internal_channels,
                    "student_internal_channels": keep_channels,
                    "internal_prune_ratio": internal_actual_prune_ratio,
                    "prune_method": prune_method,
                    "selection_policy": selection_policy,
                    "pruning_threshold": float(output_prune["result"].threshold),
                    "internal_pruning_threshold": float(internal_prune["result"].threshold),
                }
            )
            for channel_index, importance_value in enumerate(output_prune["values"]):
                channel_level_detail.append(
                    {
                        "layer_name": block_name,
                        "pruning_layer_name": output_layer_name,
                        "channel_role": "conv3_output",
                        "channel_index": int(channel_index),
                        "importance": float(importance_value),
                        "rank_desc": int(output_prune["rank_lookup"][channel_index]),
                        "decision": "keep" if channel_index in output_prune["kept_index_set"] else "prune",
                        "prune_method": prune_method,
                        "selection_policy": selection_policy,
                        "pruning_threshold": float(output_prune["result"].threshold),
                    }
                )
            for channel_index, importance_value in enumerate(internal_prune["values"]):
                channel_level_detail.append(
                    {
                        "layer_name": block_name,
                        "pruning_layer_name": internal_layer_name,
                        "channel_role": "internal_width",
                        "channel_index": int(channel_index),
                        "importance": float(importance_value),
                        "rank_desc": int(internal_prune["rank_lookup"][channel_index]),
                        "decision": "keep" if channel_index in internal_prune["kept_index_set"] else "prune",
                        "prune_method": prune_method,
                        "selection_policy": selection_policy,
                        "pruning_threshold": float(internal_prune["result"].threshold),
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
    total_pruned = int(total_before - total_after)
    changed_layers = int(sum(1 for row in teacher_vs_student_rows if row["channels_pruned"] > 0))
    global_pruning_summary = {
        "num_layers_analyzed": int(len(teacher_vs_student_rows)),
        "total_channels_before": total_before,
        "total_channels_after": total_after,
        "total_channels_pruned": total_pruned,
        "global_prune_ratio": float(total_pruned / max(1, total_before)),
        "internal_channels_before": internal_total_before,
        "internal_channels_after": internal_total_after,
        "internal_channels_pruned": int(internal_total_before - internal_total_after),
        "internal_global_prune_ratio": float((internal_total_before - internal_total_after) / max(1, internal_total_before)),
        "num_layers_changed": changed_layers,
        "num_layers_unchanged": int(len(teacher_vs_student_rows) - changed_layers),
        "protected_boundary_policy": "conv3 output is pruned; residual projections are inserted or subset-copied whenever block input/output channels differ",
    }

    return {
        "channel_config": tuple(int(row["kept_output_channels"]) for row in full_prune_plan),
        "stage_full_channel_config": {stage: list(values) for stage, values in stage_full_channel_config.items()},
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
        "criterion": "bn2/bn3_weight_or_conv_l1",
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
    for module_name in target_modules:
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
                "pruning_layer_name": pruning_layer_name,
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
                "pruning_layer_name": pruning_layer_name,
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
                    "pruning_layer_name": pruning_layer_name,
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
