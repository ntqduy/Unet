from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from utils.model_output import extract_model_info


def sanitize_tag(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._-") or "unknown"


def build_run_dir(
    *,
    project_root: Path | str,
    branch: str,
    experiment: str,
    dataset: str,
    model_name: str,
    phase: str = "main",
    variant: Optional[str] = None,
) -> Path:
    run_dir = (
        Path(project_root)
        / "logs"
        / "runs"
        / sanitize_tag(branch)
        / sanitize_tag(experiment)
        / sanitize_tag(dataset)
        / sanitize_tag(model_name)
        / sanitize_tag(phase)
    )
    if variant:
        run_dir = run_dir / sanitize_tag(variant)
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
    config_path = ensure_run_layout(run_dir)["configs_dir"] / "run_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    return config_path


def write_model_config(run_dir: Path | str, model_or_info) -> Path:
    model_config_path = ensure_run_layout(run_dir)["configs_dir"] / "model_config.json"
    model_info = extract_model_info(model_or_info) if not isinstance(model_or_info, dict) else model_or_info
    with model_config_path.open("w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=2)
    return model_config_path
