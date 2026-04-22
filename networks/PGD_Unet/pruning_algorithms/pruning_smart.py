from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch.nn as nn


STATIC_RATIO_PRUNE_METHODS = {"static", "middle_static"}
MIDDLE_STATIC_PRUNE_METHOD = "middle_static"
CHANNEL_LAYER_TYPES = (nn.Conv2d, nn.ConvTranspose2d)
PROJECTION_LAYER_TOKENS = ("downsample", "skip", "shortcut", "projection", "proj")


@dataclass(frozen=True)
class MiddleLayerSelection:
    layer_name: str
    layer: nn.Module
    selection_policy: str
    protects_block_boundaries: bool
    has_middle_layer: bool
    candidate_layer_names: Tuple[str, ...]
    boundary_layer_names: Tuple[str, ...]


def uses_static_prune_ratio(prune_method: str) -> bool:
    return str(prune_method).lower() in STATIC_RATIO_PRUNE_METHODS


def is_middle_static_pruning(prune_method: str) -> bool:
    return str(prune_method).lower() == MIDDLE_STATIC_PRUNE_METHOD


def _is_projection_layer(layer_name: str) -> bool:
    lowered = layer_name.lower()
    return any(token in lowered for token in PROJECTION_LAYER_TOKENS)


def _named_channel_layers(module: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers: List[Tuple[str, nn.Module]] = []
    for name, child in module.named_modules():
        if child is module:
            continue
        if isinstance(child, CHANNEL_LAYER_TYPES):
            layers.append((name, child))
    return layers


def _select_middle_from_layers(layers: List[Tuple[str, nn.Module]]) -> Tuple[str, nn.Module] | None:
    main_layers = [(name, layer) for name, layer in layers if not _is_projection_layer(name)]
    if len(main_layers) < 3:
        return None
    interior_layers = main_layers[1:-1]
    if not interior_layers:
        return None
    return interior_layers[len(interior_layers) // 2]


def _select_middle_from_child_block(module: nn.Module) -> Tuple[str, nn.Module, List[str]] | None:
    for child_name, child in module.named_children():
        child_layers = _named_channel_layers(child)
        child_middle = _select_middle_from_layers(child_layers)
        if child_middle is None:
            continue
        relative_name, layer = child_middle
        candidate_names = [f"{child_name}.{name}" for name, _ in child_layers]
        return f"{child_name}.{relative_name}", layer, candidate_names
    return None


def select_block_middle_channel_layer(
    module_name: str,
    module: nn.Module,
    fallback_layer: nn.Module,
) -> MiddleLayerSelection:
    """
    Select the interior channel layer used by S5.

    S5 protects the first and last conv-like layers of each block. It prunes by
    fixed ratio using an interior layer when one exists; otherwise it keeps the
    stage unchanged so boundary weights can still be copied safely.
    """
    child_middle = _select_middle_from_child_block(module)
    if child_middle is not None:
        relative_name, layer, candidate_names = child_middle
        main_candidates = [name for name in candidate_names if not _is_projection_layer(name)]
        boundary_names = tuple(name for name in (main_candidates[:1] + main_candidates[-1:]) if name)
        return MiddleLayerSelection(
            layer_name=f"{module_name}.{relative_name}" if module_name else relative_name,
            layer=layer,
            selection_policy="topk_static_ratio_on_first_child_block_middle_layer",
            protects_block_boundaries=True,
            has_middle_layer=True,
            candidate_layer_names=tuple(candidate_names),
            boundary_layer_names=boundary_names,
        )

    module_layers = _named_channel_layers(module)
    module_middle = _select_middle_from_layers(module_layers)
    if module_middle is not None:
        relative_name, layer = module_middle
        candidate_names = [name for name, _ in module_layers]
        main_candidates = [name for name in candidate_names if not _is_projection_layer(name)]
        boundary_names = tuple(name for name in (main_candidates[:1] + main_candidates[-1:]) if name)
        return MiddleLayerSelection(
            layer_name=f"{module_name}.{relative_name}" if module_name else relative_name,
            layer=layer,
            selection_policy="topk_static_ratio_on_block_middle_layer",
            protects_block_boundaries=True,
            has_middle_layer=True,
            candidate_layer_names=tuple(candidate_names),
            boundary_layer_names=boundary_names,
        )

    fallback_name = module_name
    return MiddleLayerSelection(
        layer_name=fallback_name,
        layer=fallback_layer,
        selection_policy="keep_all_no_block_middle_layer",
        protects_block_boundaries=True,
        has_middle_layer=False,
        candidate_layer_names=tuple(name for name, _ in module_layers),
        boundary_layer_names=tuple(name for name, _ in module_layers),
    )
