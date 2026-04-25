from __future__ import annotations

import json
import shutil
import csv
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from utils.experiment import project_relative_path
from utils.model_output import extract_model_info


PROFILING_STATE_KEYS = {"total_ops", "total_params"}


def ensure_checkpoint_layout(run_dir: Path | str) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    layout = {
        "run_dir": run_dir,
        "checkpoint_dir": run_dir / "checkpoints",
        "metadata_dir": run_dir / "checkpoints" / "metadata",
    }
    for path in layout.values():
        if path != run_dir:
            path.mkdir(parents=True, exist_ok=True)
    return layout


def _strip_profiling_state_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_state_dict: Dict[str, Any] = {}
    for key, value in state_dict.items():
        suffix = key.rsplit(".", 1)[-1]
        if suffix in PROFILING_STATE_KEYS:
            continue
        cleaned_state_dict[key] = value
    return cleaned_state_dict


def _remove_profiling_buffers_from_model(model) -> None:
    for module in model.modules():
        buffer_store = getattr(module, "_buffers", None)
        if isinstance(buffer_store, dict):
            for key in PROFILING_STATE_KEYS:
                buffer_store.pop(key, None)
        non_persistent = getattr(module, "_non_persistent_buffers_set", None)
        if hasattr(non_persistent, "discard"):
            for key in PROFILING_STATE_KEYS:
                non_persistent.discard(key)


def build_checkpoint_payload(
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    best_metric: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    phase: Optional[str] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    extracted_model_info = extract_model_info(model)
    merged_model_info = dict(extracted_model_info)
    if model_info:
        merged_model_info.update(model_info)
    payload = {
        "model_state_dict": _strip_profiling_state_keys(dict(model.state_dict())),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "metrics": metrics or {},
        "config": config or {},
        "model_info": merged_model_info,
        "phase": phase,
        "extra_state": extra_state or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    return payload


def build_checkpoint_metadata(
    payload: Dict[str, Any],
    checkpoint_path: Path | str,
    *,
    project_root: Path | str | None = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path_value = project_relative_path(checkpoint_path, project_root) if project_root is not None else str(checkpoint_path)
    metadata = {
        "checkpoint_path": checkpoint_path_value,
        "checkpoint_name": checkpoint_path.name,
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "best_metric": payload.get("best_metric"),
        "metrics": payload.get("metrics", {}),
        "config": payload.get("config", {}),
        "model_info": payload.get("model_info", {}),
        "phase": payload.get("phase"),
        "extra_state": payload.get("extra_state", {}),
    }
    if extra_fields:
        metadata.update(extra_fields)
    return metadata


def write_checkpoint_metadata(
    metadata_path: Path | str,
    payload: Dict[str, Any],
    checkpoint_path: Path | str,
    *,
    project_root: Path | str | None = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata_path = Path(metadata_path)
    metadata = build_checkpoint_metadata(
        payload,
        checkpoint_path,
        project_root=project_root,
        extra_fields=extra_fields,
    )
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metadata


def _write_metadata(metadata_path: Path, payload: Dict[str, Any], checkpoint_path: Path, project_root: Path | str | None = None) -> Dict[str, Any]:
    return write_checkpoint_metadata(metadata_path, payload, checkpoint_path, project_root=project_root)


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if value == "":
            return '""'
        if any(char in value for char in ":#\n\r\t") or value.strip() != value:
            return json.dumps(value, ensure_ascii=False)
        return value
    return json.dumps(value, ensure_ascii=False)


def _write_yaml_value(lines: list[str], key: str, value: Any, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, dict):
        lines.append(f"{prefix}{key}:")
        for child_key, child_value in value.items():
            _write_yaml_value(lines, str(child_key), child_value, indent + 2)
    elif isinstance(value, (list, tuple)):
        lines.append(f"{prefix}{key}:")
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}  -")
                for child_key, child_value in item.items():
                    _write_yaml_value(lines, str(child_key), child_value, indent + 4)
            else:
                lines.append(f"{prefix}  - {_yaml_scalar(item)}")
    else:
        lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")


def _write_config_yaml(checkpoint_dir: Path, config: Dict[str, Any]) -> None:
    lines = []
    for key, value in sorted(dict(config or {}).items()):
        _write_yaml_value(lines, str(key), value)
    (checkpoint_dir / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(child_value) for key, child_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_metrics_json(checkpoint_dir: Path, payload: Dict[str, Any], checkpoint_path: Path, *, project_root: Path | str | None = None) -> None:
    metrics_payload = {
        "checkpoint_path": project_relative_path(checkpoint_path, project_root) if project_root is not None else str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "best_metric": payload.get("best_metric"),
        "metrics": payload.get("metrics", {}),
        "phase": payload.get("phase"),
    }
    with (checkpoint_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(_json_safe(metrics_payload), file, indent=2, ensure_ascii=False)


def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in dict(metrics or {}).items():
        flattened[str(key)] = value if isinstance(value, (str, int, float, bool)) or value is None else json.dumps(_json_safe(value), ensure_ascii=False)
    return flattened


def _append_train_log(checkpoint_dir: Path, payload: Dict[str, Any], tag: str, checkpoint_path: Path, *, is_best: bool, project_root: Path | str | None = None) -> None:
    log_path = checkpoint_dir / "train_log.csv"
    metric_values = _flatten_metrics(payload.get("metrics", {}))
    row = {
        "tag": tag,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_path": project_relative_path(checkpoint_path, project_root) if project_root is not None else str(checkpoint_path),
        "phase": payload.get("phase"),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "is_best": int(bool(is_best)),
        "best_metric": payload.get("best_metric"),
        **metric_values,
    }
    fieldnames = list(row.keys())
    existing_rows = []
    if log_path.is_file():
        with log_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            existing_rows = list(reader)
            for fieldname in reader.fieldnames or []:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)
            for existing_row in existing_rows:
                for key in existing_row.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow(existing_row)
        writer.writerow(row)


def _write_checkpoint_sidecars(
    checkpoint_dir: Path,
    payload: Dict[str, Any],
    tag: str,
    checkpoint_path: Path,
    *,
    is_best: bool,
    project_root: Path | str | None = None,
) -> None:
    _write_config_yaml(checkpoint_dir, payload.get("config", {}))
    _write_metrics_json(checkpoint_dir, payload, checkpoint_path, project_root=project_root)
    _append_train_log(checkpoint_dir, payload, tag, checkpoint_path, is_best=is_best, project_root=project_root)


def save_checkpoint_payload_atomic(payload: Any, checkpoint_path: Path | str) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_checkpoint_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    try:
        torch.save(payload, temp_checkpoint_path)
        temp_checkpoint_path.replace(checkpoint_path)
    finally:
        if temp_checkpoint_path.exists():
            temp_checkpoint_path.unlink()
    return checkpoint_path


def save_checkpoint(
    run_dir: Path | str,
    tag: str,
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    best_metric: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    phase: Optional[str] = None,
    extra_state: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
    save_tagged_checkpoint: bool = False,
    save_last_checkpoint: bool = True,
    include_optimizer_state: bool = False,
    project_root: Path | str | None = None,
) -> Optional[Path]:
    layout = ensure_checkpoint_layout(run_dir)
    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer if include_optimizer_state else None,
        scheduler=scheduler if include_optimizer_state else None,
        scaler=scaler if include_optimizer_state else None,
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        metrics=metrics,
        config=config,
        model_info=model_info,
        phase=phase,
        extra_state=extra_state,
    )

    return_path = None
    if save_last_checkpoint:
        last_checkpoint_path = layout["checkpoint_dir"] / "last.pth"
        save_checkpoint_payload_atomic(payload, last_checkpoint_path)
        _write_metadata(layout["metadata_dir"] / "last.json", payload, last_checkpoint_path, project_root=project_root)
        _write_checkpoint_sidecars(layout["checkpoint_dir"], payload, tag, last_checkpoint_path, is_best=is_best, project_root=project_root)
        return_path = last_checkpoint_path

    if save_tagged_checkpoint and tag not in {"last", "best"}:
        checkpoint_path = layout["checkpoint_dir"] / f"{tag}.pth"
        save_checkpoint_payload_atomic(payload, checkpoint_path)
        _write_metadata(layout["metadata_dir"] / f"{tag}.json", payload, checkpoint_path, project_root=project_root)
        _write_checkpoint_sidecars(layout["checkpoint_dir"], payload, tag, checkpoint_path, is_best=is_best, project_root=project_root)
        return_path = checkpoint_path

    if is_best:
        best_checkpoint_path = layout["checkpoint_dir"] / "best.pth"
        save_checkpoint_payload_atomic(payload, best_checkpoint_path)
        _write_metadata(layout["metadata_dir"] / "best.json", payload, best_checkpoint_path, project_root=project_root)
        _write_checkpoint_sidecars(layout["checkpoint_dir"], payload, tag, best_checkpoint_path, is_best=True, project_root=project_root)
        return_path = best_checkpoint_path

    return return_path


def load_checkpoint(checkpoint_path: Path | str, *, device: Optional[torch.device] = None) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    if checkpoint_path.stat().st_size == 0:
        raise EOFError(f"Checkpoint file is empty: {checkpoint_path}")
    map_location = device if device is not None else "cpu"
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except EOFError as error:
        raise EOFError(f"Checkpoint file is incomplete or corrupted: {checkpoint_path}") from error


def load_checkpoint_into_model(checkpoint_path: Path | str, model, *, device: Optional[torch.device] = None, strict: bool = True) -> Dict[str, Any]:
    payload = load_checkpoint(checkpoint_path, device=device)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    state_dict = _strip_profiling_state_keys(dict(state_dict))
    _remove_profiling_buffers_from_model(model)
    model.load_state_dict(state_dict, strict=strict)
    return payload if isinstance(payload, dict) else {"model_state_dict": state_dict}


def resolve_phase_checkpoint(run_dir: Path | str, preferred: str = "best") -> Optional[Path]:
    checkpoint_path = Path(run_dir) / "checkpoints" / f"{preferred}.pth"
    if checkpoint_path.is_file():
        return checkpoint_path
    return None


def clone_checkpoint_file(
    source_checkpoint_path: Path | str,
    target_checkpoint_path: Path | str,
    *,
    project_root: Path | str | None = None,
    payload_overrides: Optional[Dict[str, Any]] = None,
    metadata_extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    source_checkpoint_path = Path(source_checkpoint_path)
    target_checkpoint_path = Path(target_checkpoint_path)
    target_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = load_checkpoint(source_checkpoint_path)
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["model_state_dict"] = _strip_profiling_state_keys(dict(payload.get("model_state_dict", {})))
        if payload_overrides:
            for key, value in payload_overrides.items():
                payload[key] = value
        save_checkpoint_payload_atomic(payload, target_checkpoint_path)
    else:
        shutil.copy2(source_checkpoint_path, target_checkpoint_path)
        payload = {"model_state_dict": {}}

    metadata_dir = target_checkpoint_path.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{target_checkpoint_path.stem}.json"
    metadata = write_checkpoint_metadata(
        metadata_path,
        payload,
        target_checkpoint_path,
        project_root=project_root,
        extra_fields=metadata_extra_fields,
    )
    _write_checkpoint_sidecars(
        target_checkpoint_path.parent,
        payload,
        target_checkpoint_path.stem,
        target_checkpoint_path,
        is_best=target_checkpoint_path.stem == "best",
        project_root=project_root,
    )
    return metadata
