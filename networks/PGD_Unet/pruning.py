from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


DEFAULT_BLUEPRINT_MODULES = ("stem", "down1", "down2", "down3", "down4")


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


def extract_pruned_blueprint(
    teacher_model: nn.Module,
    *,
    prune_ratio: float = 0.5,
    target_modules: Optional[Sequence[str]] = None,
    minimum_channels: int = 16,
) -> Dict[str, object]:
    if not 0.0 <= prune_ratio < 1.0:
        raise ValueError("prune_ratio must be in [0, 1).")

    target_modules = tuple(target_modules or DEFAULT_BLUEPRINT_MODULES)
    blueprint_channels: List[int] = []
    per_module: List[Dict[str, object]] = []

    named_modules = dict(teacher_model.named_modules())
    for module_name in target_modules:
        module = named_modules.get(module_name)
        if module is None:
            continue

        conv_layer = None
        if isinstance(module, nn.Conv2d):
            conv_layer = module
        else:
            for child in module.modules():
                if isinstance(child, nn.Conv2d):
                    conv_layer = child
                    break
        if conv_layer is None or conv_layer.out_channels <= minimum_channels:
            continue

        importance = _resolve_module_importance(teacher_model, module_name, conv_layer)
        keep_channels = max(int(importance.numel() * (1 - prune_ratio)), minimum_channels)
        blueprint_channels.append(keep_channels)
        per_module.append(
            {
                "module_name": module_name,
                "source_out_channels": int(conv_layer.out_channels),
                "kept_channels": int(keep_channels),
                "criterion": "bn_weight_or_l1",
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

    return {
        "channel_config": tuple(int(channel) for channel in blueprint_channels[: len(DEFAULT_BLUEPRINT_MODULES)]),
        "prune_ratio": float(prune_ratio),
        "target_modules": list(target_modules),
        "criterion": "bn_weight_or_l1",
        "minimum_channels": int(minimum_channels),
        "modules": per_module,
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
