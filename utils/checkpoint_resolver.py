from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from utils.checkpoints import clone_checkpoint_file, load_checkpoint, resolve_phase_checkpoint
from utils.experiment import (
    build_basic_run_dir,
    project_relative_path,
)


COMPATIBILITY_KEYS = (
    "dataset",
    "model_name",
    "num_classes",
    "in_channels",
    "patch_size",
    "encoder_pretrained",
    "teacher_model",
    "channel_config",
    "student_variant",
)


def _normalize_sequence(value: Any) -> Optional[tuple]:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def extract_checkpoint_signature(payload: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(payload.get("config") or {})
    model_info = dict(payload.get("model_info") or {})
    architecture_config = dict(model_info.get("architecture_config") or {})
    extra_state = dict(payload.get("extra_state") or {})

    model_name = (
        model_info.get("model_name")
        or config.get("model")
        or config.get("teacher_model")
    )
    teacher_model = (
        model_info.get("teacher_model")
        or config.get("teacher_model")
        or extra_state.get("teacher_model")
    )

    channel_config = (
        architecture_config.get("channel_config")
        or extra_state.get("blueprint", {}).get("channel_config")
    )
    student_variant_info = model_info.get("student_variant")
    if isinstance(student_variant_info, dict):
        student_variant = student_variant_info.get("variant_name")
    else:
        student_variant = student_variant_info or config.get("student_variant")

    return {
        "dataset": config.get("dataset"),
        "model_name": model_name,
        "num_classes": config.get("num_classes"),
        "in_channels": config.get("in_channels"),
        "patch_size": _normalize_sequence(config.get("patch_size")),
        "encoder_pretrained": config.get("encoder_pretrained"),
        "teacher_model": teacher_model,
        "channel_config": _normalize_sequence(channel_config),
        "student_variant": student_variant,
        "phase": payload.get("phase"),
        "branch": model_info.get("branch"),
    }


def build_expected_signature(
    *,
    dataset: str,
    model_name: str,
    num_classes: int,
    in_channels: int,
    patch_size: Sequence[int],
    encoder_pretrained: Optional[int | bool] = None,
    teacher_model: Optional[str] = None,
    channel_config: Optional[Sequence[int]] = None,
    student_variant: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "model_name": model_name,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "patch_size": tuple(patch_size),
        "encoder_pretrained": encoder_pretrained,
        "teacher_model": teacher_model,
        "channel_config": tuple(channel_config) if channel_config is not None else None,
        "student_variant": student_variant,
    }


def evaluate_checkpoint_compatibility(payload: Dict[str, Any], expected_signature: Dict[str, Any]) -> Dict[str, Any]:
    actual_signature = extract_checkpoint_signature(payload)
    mismatches = []
    for key in COMPATIBILITY_KEYS:
        expected_value = expected_signature.get(key)
        if expected_value in (None, ""):
            continue
        actual_value = actual_signature.get(key)
        if _normalize_scalar(actual_value) != _normalize_scalar(expected_value):
            mismatches.append(
                {
                    "key": key,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    return {
        "compatible": len(mismatches) == 0,
        "mismatches": mismatches,
        "actual_signature": actual_signature,
        "expected_signature": dict(expected_signature),
    }


def find_compatible_checkpoint(
    candidate_paths: Iterable[Path | str],
    *,
    expected_signature: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    for candidate in candidate_paths:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if not candidate_path.is_file():
            continue
        try:
            payload = load_checkpoint(candidate_path)
        except Exception:
            continue
        compatibility = evaluate_checkpoint_compatibility(payload, expected_signature)
        if compatibility["compatible"]:
            return {
                "checkpoint_path": candidate_path.resolve(),
                "payload": payload,
                "compatibility": compatibility,
            }
    return None


def resolve_run_checkpoint(
    run_dir: Path | str,
    *,
    expected_signature: Dict[str, Any],
    preferred_order: Sequence[str] = ("best", "last"),
) -> Optional[Dict[str, Any]]:
    candidate_paths = [resolve_phase_checkpoint(run_dir, preferred) for preferred in preferred_order]
    return find_compatible_checkpoint(candidate_paths, expected_signature=expected_signature)


def resolve_basic_checkpoint(
    *,
    project_root: Path | str,
    output_root: Path | str | None,
    model_name: str,
    dataset: str,
    expected_signature: Dict[str, Any],
    preferred_order: Sequence[str] = ("best", "last"),
) -> Optional[Dict[str, Any]]:
    basic_run_dir = build_basic_run_dir(
        project_root=project_root,
        dataset=dataset,
        model_name=model_name,
        output_root=output_root,
    )
    candidate_run_dirs = (
        basic_run_dir,
        basic_run_dir / "basic",
    )
    for candidate_run_dir in candidate_run_dirs:
        match = resolve_run_checkpoint(candidate_run_dir, expected_signature=expected_signature, preferred_order=preferred_order)
        if match is not None:
            match["run_dir"] = candidate_run_dir
            return match
    return None


def register_reused_checkpoint(
    *,
    source_checkpoint_path: Path | str,
    target_run_dir: Path | str,
    project_root: Path | str,
    source_branch: str,
    source_run_dir: Path | str | None = None,
    alias_names: Sequence[str] = ("best", "last"),
    payload_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target_run_dir = Path(target_run_dir)
    source_checkpoint_path = Path(source_checkpoint_path)
    source_checkpoint_relative = project_relative_path(source_checkpoint_path, project_root)
    source_run_relative = project_relative_path(source_run_dir, project_root) if source_run_dir else ""

    payload = load_checkpoint(source_checkpoint_path)
    payload = dict(payload)
    payload.setdefault("extra_state", {})
    payload["extra_state"] = dict(payload["extra_state"])
    payload["extra_state"]["checkpoint_origin"] = {
        "source_branch": source_branch,
        "source_checkpoint_path": source_checkpoint_relative,
        "source_run_dir": source_run_relative,
    }
    if payload_updates:
        for key, value in payload_updates.items():
            payload[key] = value

    copied = {}
    for alias_name in alias_names:
        checkpoint_path = target_run_dir / "checkpoints" / f"{alias_name}.pth"
        metadata = clone_checkpoint_file(
            source_checkpoint_path,
            checkpoint_path,
            project_root=project_root,
            payload_overrides=payload,
            metadata_extra_fields={
                "checkpoint_origin": payload["extra_state"]["checkpoint_origin"],
                "registered_from_existing_checkpoint": True,
            },
        )
        copied[alias_name] = {
            "checkpoint_path": checkpoint_path,
            "metadata": metadata,
        }
    return copied
