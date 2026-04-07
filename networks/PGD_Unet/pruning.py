from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from utils.channel_analysis import find_primary_channel_layer


DEFAULT_BLUEPRINT_MODULES = ("stem", "down1", "down2", "down3", "down4")
MODULE_GROUP_CANDIDATES = (
    ("stem", "down1", "down2", "down3", "down4"),
    ("stem", "layer1", "layer2", "layer3", "layer4"),
    ("enc1", "enc2", "enc3", "enc4", "bottleneck"),
    ("encoder.block_one", "encoder.block_two", "encoder.block_three", "encoder.block_four", "encoder.block_five"),
)


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


def extract_pruned_blueprint(
    teacher_model: nn.Module,
    *,
    prune_ratio: float = 0.5,
    target_modules: Optional[Sequence[str]] = None,
    minimum_channels: int = 16,
) -> Dict[str, object]:
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError("prune_ratio must be in [0, 1).")

    target_modules = tuple(_resolve_target_modules(teacher_model, target_modules))
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

        importance = _resolve_module_importance(teacher_model, module_name, conv_layer)
        ranked_indices = torch.argsort(importance, descending=True)
        rank_lookup = {int(index): rank + 1 for rank, index in enumerate(ranked_indices.tolist())}
        keep_channels = max(int(importance.numel() * (1 - prune_ratio)), minimum_channels)
        keep_channels = min(keep_channels, int(importance.numel()))
        kept_indices = sorted(int(index) for index in ranked_indices[:keep_channels].tolist())
        pruned_indices = sorted(int(index) for index in ranked_indices[keep_channels:].tolist())
        importance_values = [float(value) for value in importance.detach().cpu().tolist()]
        stats = _tensor_stats(importance_values)
        actual_prune_ratio = float(len(pruned_indices) / max(1, int(importance.numel())))
        blueprint_channels.append(keep_channels)

        teacher_channel_analysis.append(
            {
                "layer_name": module_name,
                "module_type": type(module).__name__,
                "channel_layer_type": type(conv_layer).__name__,
                "in_channels": int(getattr(conv_layer, "in_channels", 0)),
                "teacher_out_channels": int(conv_layer.out_channels),
                "kernel_size": list(conv_layer.kernel_size) if hasattr(conv_layer, "kernel_size") else None,
                "weight_shape": [int(value) for value in conv_layer.weight.shape],
                "importance_source": "bn_weight_or_l1",
                "importance_values": importance_values,
                "ranked_channel_indices_desc": [int(index) for index in ranked_indices.tolist()],
                "kept_channel_indices": kept_indices,
                "pruned_channel_indices": pruned_indices,
                "importance_min": stats["min"],
                "importance_max": stats["max"],
                "importance_mean": stats["mean"],
                "importance_std": stats["std"],
            }
        )

        per_module.append(
            {
                "layer_name": module_name,
                "module_name": module_name,
                "module_type": type(module).__name__,
                "channel_layer_type": type(conv_layer).__name__,
                "in_channels": int(getattr(conv_layer, "in_channels", 0)),
                "source_out_channels": int(conv_layer.out_channels),
                "teacher_out_channels": int(conv_layer.out_channels),
                "student_out_channels": int(keep_channels),
                "kept_channels": int(keep_channels),
                "pruned_channels": int(len(pruned_indices)),
                "actual_prune_ratio": actual_prune_ratio,
                "criterion": "bn_weight_or_l1",
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
                "teacher_out_channels": int(conv_layer.out_channels),
                "student_out_channels": int(keep_channels),
                "channels_pruned": int(len(pruned_indices)),
                "actual_prune_ratio": actual_prune_ratio,
            }
        )
        for channel_index, importance_value in enumerate(importance_values):
            channel_level_detail.append(
                {
                    "layer_name": module_name,
                    "channel_index": int(channel_index),
                    "importance": float(importance_value),
                    "rank_desc": int(rank_lookup[channel_index]),
                    "decision": "keep" if channel_index in kept_indices else "prune",
                }
            )

    if len(blueprint_channels) != len(DEFAULT_BLUEPRINT_MODULES):
        fallback = []
        for module in teacher_model.modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > minimum_channels:
                fallback.append(max(int(module.out_channels * (1 - prune_ratio)), minimum_channels))
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
        "prune_ratio": float(prune_ratio),
        "target_modules": list(target_modules),
        "criterion": "bn_weight_or_l1",
        "minimum_channels": int(minimum_channels),
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
