from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from utils.model_output import extract_model_info


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
        "model_state_dict": model.state_dict(),
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


def _write_metadata(metadata_path: Path, payload: Dict[str, Any], checkpoint_path: Path) -> Dict[str, Any]:
    metadata = {
        "checkpoint_path": str(checkpoint_path.resolve()),
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
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metadata


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
) -> Path:
    layout = ensure_checkpoint_layout(run_dir)
    payload = build_checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        metrics=metrics,
        config=config,
        model_info=model_info,
        phase=phase,
        extra_state=extra_state,
    )
    checkpoint_path = layout["checkpoint_dir"] / f"{tag}.pth"
    torch.save(payload, checkpoint_path)
    _write_metadata(layout["metadata_dir"] / f"{tag}.json", payload, checkpoint_path)

    last_checkpoint_path = layout["checkpoint_dir"] / "last.pth"
    if checkpoint_path != last_checkpoint_path:
        shutil.copyfile(checkpoint_path, last_checkpoint_path)
        _write_metadata(layout["metadata_dir"] / "last.json", payload, last_checkpoint_path)

    if is_best:
        best_checkpoint_path = layout["checkpoint_dir"] / "best.pth"
        if checkpoint_path != best_checkpoint_path:
            shutil.copyfile(checkpoint_path, best_checkpoint_path)
        _write_metadata(layout["metadata_dir"] / "best.json", payload, best_checkpoint_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path: Path | str, *, device: Optional[torch.device] = None) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    map_location = device if device is not None else "cpu"
    return torch.load(checkpoint_path, map_location=map_location)


def load_checkpoint_into_model(checkpoint_path: Path | str, model, *, device: Optional[torch.device] = None, strict: bool = True) -> Dict[str, Any]:
    payload = load_checkpoint(checkpoint_path, device=device)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=strict)
    return payload if isinstance(payload, dict) else {"model_state_dict": state_dict}


def resolve_phase_checkpoint(run_dir: Path | str, preferred: str = "best") -> Optional[Path]:
    checkpoint_path = Path(run_dir) / "checkpoints" / f"{preferred}.pth"
    if checkpoint_path.is_file():
        return checkpoint_path
    return None
