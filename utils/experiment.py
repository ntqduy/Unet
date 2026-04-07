from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from utils.model_output import extract_model_info


DEFAULT_OUTPUT_ROOT_NAME = "outputs"


def sanitize_tag(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._-") or "unknown"


def resolve_output_root(project_root: Path | str, output_root: Path | str | None = None) -> Path:
    if output_root:
        return Path(output_root).expanduser()
    return Path(project_root) / DEFAULT_OUTPUT_ROOT_NAME


def normalize_path_string(path_value: Path | str | None) -> str:
    if path_value in (None, ""):
        return ""
    return Path(path_value).as_posix()


def project_relative_path(path_value: Path | str | None, project_root: Path | str) -> str:
    if path_value in (None, ""):
        return ""
    project_root_path = Path(project_root).expanduser().resolve()
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        return normalize_path_string(path)
    try:
        return normalize_path_string(path.resolve().relative_to(project_root_path))
    except ValueError:
        return normalize_path_string(path)


def build_run_dir(
    *,
    project_root: Path | str,
    experiment: str,
    dataset: str,
    model_name: str,
    phase: str = "main",
    variant: Optional[str] = None,
    branch: Optional[str] = None,
    output_root: Path | str | None = None,
) -> Path:
    output_root_path = resolve_output_root(project_root, output_root)
    phase_tag = sanitize_tag(phase)
    dataset_tag = sanitize_tag(dataset)
    model_tag = sanitize_tag(model_name)
    variant_tag = sanitize_tag(variant) if variant else None

    if model_tag == "pdg_unet" and phase_tag.startswith("_"):
        run_dir = output_root_path / model_tag / phase_tag
        if variant_tag:
            run_dir = run_dir / variant_tag
        run_dir = run_dir / dataset_tag
    else:
        run_dir = output_root_path / model_tag / dataset_tag / phase_tag
        if variant_tag:
            run_dir = run_dir / variant_tag
    return run_dir


def ensure_run_layout(run_dir: Path | str) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    layout = {
        "run_dir": run_dir,
        "artifacts_dir": run_dir / "artifacts",
        "metrics_dir": run_dir / "metrics",
        "reports_dir": run_dir / "reports",
        "configs_dir": run_dir / "configs",
        "evaluation_dir": run_dir / "evaluations",
        "visualization_dir": run_dir / "artifacts" / "visualizations",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def write_run_config(run_dir: Path | str, config: Dict) -> Path:
    layout = ensure_run_layout(run_dir)
    config_path = layout["configs_dir"] / "run_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    with (layout["configs_dir"] / "hyperparameters.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    return config_path


def write_model_config(run_dir: Path | str, model_or_info) -> Path:
    model_config_path = ensure_run_layout(run_dir)["configs_dir"] / "model_config.json"
    model_info = extract_model_info(model_or_info) if not isinstance(model_or_info, dict) else model_or_info
    with model_config_path.open("w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=2)
    return model_config_path
