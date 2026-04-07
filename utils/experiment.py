from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional


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
