from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dataloaders.dataset import list_available_datasets
from networks.net_factory import list_models
from utils.experiment import build_basic_run_dir


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_DATASET_ROOTS = {
    "cvc_clinicdb": "data/CVC-ClinicDB",
    "cvc": "data/CVC-ClinicDB",
    "kvasir_seg": "data/Kvasir-SEG",
    "kvasir": "data/Kvasir-SEG",
    "etis": "data/ETIS",
    "etis_larib": "data/ETIS",
    "cvc_colondb": "data/CVC-ColonDB",
    "cvc_colon_db": "data/CVC-ColonDB",
    "cvc_300": "data/CVC-300",
    "cvc300": "data/CVC-300",
}


def _load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError as error:
        raise RuntimeError("YAML config requires PyYAML. Use JSON or install PyYAML.") from error


def _dataset_root(dataset: str, config: Dict[str, Any]) -> str:
    roots = dict(config.get("dataset_roots", {}))
    return str(roots.get(dataset) or DEFAULT_DATASET_ROOTS.get(dataset) or f"data/{dataset}")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_run(dataset: str, model: str, output_root: str) -> List[Dict[str, Any]]:
    run_dir = build_basic_run_dir(
        project_root=PROJECT_ROOT,
        dataset=dataset,
        model_name=model,
        output_root=output_root or None,
    )
    rows = _read_csv_rows(run_dir / "metrics" / "basic_metrics.csv")
    for row in rows:
        row.setdefault("dataset", dataset)
        row.setdefault("model_name", model)
        row["run_dir"] = str(run_dir.relative_to(PROJECT_ROOT) if run_dir.is_absolute() and PROJECT_ROOT in run_dir.parents else run_dir)
    return rows


def _build_command(args: argparse.Namespace, dataset: str, model: str, root_path: str, passthrough: List[str]) -> List[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train_basic_model.py"),
        "--dataset",
        dataset,
        "--root_path",
        root_path,
        "--model",
        model,
        "--output_root",
        args.output_root,
        "--batch_size",
        str(args.batch_size),
        "--base_lr",
        str(args.base_lr),
        "--num_workers",
        str(args.num_workers),
        "--gpu",
        str(args.device).replace("cuda:", ""),
        "--encoder_pretrained",
        str(args.encoder_pretrained),
    ]
    if args.epochs is not None:
        command.extend(["--max_epochs", str(args.epochs)])
    if args.force_retrain:
        command.extend(["--force_retrain", "1"])
    command.extend(passthrough)
    return command


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Run existing basic-model trainer across datasets/models.")
    parser.add_argument("--datasets", nargs="+", default=["cvc_300", "cvc_clinicdb", "kvasir_seg", "etis", "cvc_colondb"], choices=list_available_datasets())
    parser.add_argument("--models", nargs="+", default=["unet", "resunet", "vnet", "unetr", "unet_resnet152"], choices=list_models())
    parser.add_argument("--config", type=str, default="", help="Optional JSON/YAML config with dataset_roots.")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="0", help="GPU id, cuda:0, or cpu.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--encoder-pretrained", type=int, default=1, help="Use 1 for pretrained ResNet152 encoder when model=unet_resnet152.")
    parser.add_argument("--force-retrain", action="store_true", help="Ignore compatible existing checkpoints and train again.")
    parser.add_argument("--summary-csv", type=str, default="", help="Optional path for aggregate summary CSV.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = _load_config(args.config)
    summary_rows: List[Dict[str, Any]] = []

    env = os.environ.copy()
    if str(args.device).lower() == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    for dataset in args.datasets:
        root_path = _dataset_root(dataset, config)
        if not Path(root_path).exists():
            logging.warning("Dataset root does not exist yet: %s", root_path)
        for model in args.models:
            command = _build_command(args, dataset, model, root_path, passthrough)
            logging.info("Run basic model | dataset=%s | model=%s", dataset, model)
            result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
            if result.returncode != 0:
                logging.warning("Run failed | dataset=%s | model=%s | code=%s", dataset, model, result.returncode)
                if args.stop_on_error:
                    return int(result.returncode)
                continue
            summary_rows.extend(_summarize_run(dataset, model, args.output_root))

    summary_path = Path(args.summary_csv) if args.summary_csv else Path(args.output_root) / "basic_models_summary.csv"
    if summary_rows:
        _write_csv(summary_path, summary_rows)
        logging.info("Saved aggregate summary: %s", summary_path)
    else:
        logging.warning("No basic metrics were found to summarize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
