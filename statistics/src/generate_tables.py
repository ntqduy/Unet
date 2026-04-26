from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
try:
    import pandas as pd
except ImportError as error:  # pragma: no cover - dependency guard
    raise SystemExit("Missing dependency: pandas. Install project requirements with `pip install -r requirements.txt`.") from error


TABLE1_COLUMNS = ["Model", "Dice", "IoU", "HD95", "Params", "FLOPs", "FPS", "Inf (s)"]
TABLE2_COLUMNS = ["Method", "Dice", "IoU", "HD95", "Params", "FLOPs", "FPS", "Inf (s)", "Search Time (s)"]
TABLE3_COLUMNS = TABLE2_COLUMNS
TABLE4_COLUMNS = ["Component", "Dice", "IoU", "HD95", "Params", "FLOPs", "FPS", "Inf (s)", "Search Time (s)"]
TABLE5_COLUMNS = ["Method", "Pruning Time (s)", "Search Time (s)", "Training Time (s)", "Inference Time (s)", "Total Time (s)"]
MEAN_STD_COLUMNS = [
    "Method",
    "Mean Dice",
    "Std Dice",
    "Mean IoU",
    "Std IoU",
    "Mean HD95",
    "Std HD95",
    "Mean Params",
    "Std Params",
    "Mean FLOPs",
    "Std FLOPs",
    "Mean FPS",
    "Std FPS",
    "Mean Inf (s)",
    "Std Inf (s)",
    "Mean Search Time (s)",
    "Std Search Time (s)",
]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        logging.warning("Cannot read JSON: %s | %s", path, error)
        return {}


def _safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: Any, default: int = 0) -> int:
    number = _safe_float(value)
    if np.isnan(number):
        return default
    return int(number)


def _infer_dataset(path: Path, row: Dict[str, Any], outputs_root: Path) -> str:
    dataset = str(row.get("dataset") or "").strip()
    if dataset:
        return dataset
    try:
        rel = path.relative_to(outputs_root)
        parts = rel.parts
    except ValueError:
        parts = path.parts
    if len(parts) >= 3 and parts[0] == "pgd_unet":
        return parts[1]
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def _nearest_run_config(path: Path) -> Dict[str, Any]:
    for parent in [path.parent, *path.parents]:
        config_path = parent / "configs" / "run_config.json"
        if config_path.is_file():
            return _read_json(config_path)
    return {}


def _read_metrics_rows(outputs_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    patterns = ["basic_metrics.csv", "teacher_metrics.csv", "pruning_metrics.csv", "student_metrics.csv", "pipeline_metrics.csv", "metrics_summary.csv"]
    seen = set()
    for pattern in patterns:
        logging.info("Scanning metrics pattern: %s/%s", outputs_root, pattern)
        for csv_path in outputs_root.rglob(pattern):
            if csv_path in seen:
                continue
            seen.add(csv_path)
            logging.info("Reading metrics CSV: %s", csv_path)
            try:
                frame = pd.read_csv(csv_path)
            except Exception as error:
                logging.warning("Cannot read metrics CSV: %s | %s", csv_path, error)
                continue
            config = _nearest_run_config(csv_path)
            for _, row_series in frame.iterrows():
                row = row_series.to_dict()
                row["source_path"] = str(csv_path)
                row["dataset"] = _infer_dataset(csv_path, row, outputs_root)
                for key, value in config.items():
                    row.setdefault(key, value)
                    row.setdefault(f"config_{key}", value)
                rows.append(row)
    if not rows:
        logging.warning("No metrics rows found under %s", outputs_root)
    return pd.DataFrame(rows)


def _read_timing_rows(outputs_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    logging.info("Scanning timing summaries under: %s", outputs_root)
    for csv_path in outputs_root.rglob("timing_summary.csv"):
        logging.info("Reading timing CSV: %s", csv_path)
        try:
            frame = pd.read_csv(csv_path)
        except Exception as error:
            logging.warning("Cannot read timing CSV: %s | %s", csv_path, error)
            continue
        for _, row_series in frame.iterrows():
            row = row_series.to_dict()
            row["source_path"] = str(csv_path)
            row["dataset"] = _infer_dataset(csv_path, row, outputs_root)
            rows.append(row)
    for json_path in outputs_root.rglob("pruning_search_time.json"):
        logging.info("Reading pruning search-time JSON: %s", json_path)
        payload = _read_json(json_path)
        if not payload:
            continue
        rows.append(
            {
                "dataset": _infer_dataset(json_path, payload, outputs_root),
                "method": payload.get("prune_method") or payload.get("prune_strategy") or "pruning",
                "phase": "pruning_search",
                "search_time_seconds": payload.get("search_time_seconds", np.nan),
                "pruning_time_seconds": np.nan,
                "training_time_seconds": 0.0,
                "inference_time_seconds": np.nan,
                "total_time_seconds": payload.get("search_time_seconds", np.nan),
                "source_path": str(json_path),
            }
        )
    return pd.DataFrame(rows)


def _test_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "split" not in frame.columns:
        return frame.copy()
    test = frame[frame["split"].fillna("").astype(str).str.lower().eq("test")]
    return test.copy() if not test.empty else frame.copy()


def _metric_row(row: Dict[str, Any], name_key: str, name: str) -> Dict[str, Any]:
    return {
        name_key: name,
        "Dice": _safe_float(row.get("dice")),
        "IoU": _safe_float(row.get("iou")),
        "HD95": _safe_float(row.get("hd95")),
        "Params": _safe_float(row.get("params")),
        "FLOPs": _safe_float(row.get("flops")),
        "FPS": _safe_float(row.get("fps")),
        "Inf (s)": _safe_float(row.get("inference_time_seconds", row.get("Inf (s)"))),
        "Search Time (s)": _safe_float(row.get("search_time_seconds", row.get("Search Time (s)"))),
    }


def _loss_method(row: Dict[str, Any]) -> str:
    display_name = str(row.get("loss_method") or row.get("config_loss_method") or "").strip()
    if display_name and display_name.lower() != "nan":
        return display_name
    loss_tag = str(row.get("loss_tag") or row.get("config_loss_tag") or "").lower()
    if loss_tag and loss_tag != "nan":
        return loss_tag.replace("loss_", "").replace("_", " + ")
    kd = _safe_int(row.get("use_kd_output", row.get("config_use_kd_output", 0)))
    feat = _safe_int(row.get("use_feature_distill", row.get("config_use_feature_distill", 0)))
    aux = _safe_int(row.get("use_aux_loss", row.get("config_use_aux_loss", 0)))
    parts = ["Proposed"]
    if kd:
        parts.append("KD")
    if feat:
        parts.append("Feature")
    if aux:
        parts.append("Aux")
    return " + ".join(parts) if len(parts) > 1 else "Hybrid"


def _component(row: Dict[str, Any]) -> str:
    method = str(row.get("prune_method") or row.get("config_prune_method") or "").lower()
    adaptive = method in {"kneedle", "otsu", "gmm", "middle_kneedle", "middle_otsu", "middle_gmm"}
    block = method.startswith("middle_")
    kd = bool(_safe_int(row.get("use_kd_output", row.get("config_use_kd_output", 0))))
    distill = kd or _safe_float(row.get("lambda_distill", row.get("config_lambda_distill", 0))) > 0
    if not adaptive and not block and not distill:
        return "Static"
    parts = []
    if adaptive:
        parts.append("Adaptive")
    if block:
        parts.append("Block")
    if distill:
        parts.append("Distillation")
    return " + ".join(parts) if parts else "Static"


def _dedupe_best(rows: Iterable[Dict[str, Any]], key_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if "Dice" in frame.columns:
        frame = frame.sort_values("Dice", ascending=False, na_position="last")
    return frame.drop_duplicates(subset=[key_name], keep="first").reset_index(drop=True)


def _tables_for_dataset(dataset: str, metrics: pd.DataFrame, timing: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    dataset_metrics = _test_rows(metrics[metrics["dataset"].astype(str).eq(dataset)]) if not metrics.empty else pd.DataFrame()
    table1_rows = []
    table2_rows = []
    table3_rows = []
    table4_rows = []

    for _, row_series in dataset_metrics.iterrows():
        row = row_series.to_dict()
        phase = str(row.get("phase") or "").lower()
        model_name = str(row.get("model_name") or row.get("model") or "unknown")
        prune_method = str(row.get("prune_method") or row.get("config_prune_method") or "").lower()
        if phase == "basic" or (not prune_method and "pgd_unet" not in str(row.get("source_path", ""))):
            table1_rows.append({key: value for key, value in _metric_row(row, "Model", model_name).items() if key in TABLE1_COLUMNS})
        if prune_method:
            method_name = prune_method if phase != "student" else f"{prune_method} + KD"
            table2_rows.append(_metric_row(row, "Method", method_name))
            if phase == "student":
                table3_rows.append(_metric_row(row, "Method", _loss_method(row)))
                table4_rows.append(_metric_row(row, "Component", _component(row)))

    dataset_timing = timing[timing["dataset"].astype(str).eq(dataset)] if not timing.empty else pd.DataFrame()
    table5_rows = []
    for _, row_series in dataset_timing.iterrows():
        row = row_series.to_dict()
        method = row.get("method") or row.get("phase") or "unknown"
        table5_rows.append(
            {
                "Method": method,
                "Pruning Time (s)": _safe_float(row.get("pruning_time_seconds")),
                "Search Time (s)": _safe_float(row.get("search_time_seconds")),
                "Training Time (s)": _safe_float(row.get("training_time_seconds")),
                "Inference Time (s)": _safe_float(row.get("inference_time_seconds")),
                "Total Time (s)": _safe_float(row.get("total_time_seconds")),
            }
        )

    return {
        "table1_baseline.csv": _dedupe_best(table1_rows, "Model").reindex(columns=TABLE1_COLUMNS),
        "table2_pruning.csv": _dedupe_best(table2_rows, "Method").reindex(columns=TABLE2_COLUMNS),
        "table3_loss.csv": _dedupe_best(table3_rows, "Method").reindex(columns=TABLE3_COLUMNS),
        "table4_ablation.csv": _dedupe_best(table4_rows, "Component").reindex(columns=TABLE4_COLUMNS),
        "table5_computational_cost.csv": _dedupe_best(table5_rows, "Method").reindex(columns=TABLE5_COLUMNS),
    }


def _mean_std_table(all_tables: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame.rename(columns={"Model": "Method", "Component": "Method"}) for frame in all_tables if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=MEAN_STD_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    rows = []
    for method, group in combined.groupby("Method", dropna=True):
        rows.append(
            {
                "Method": method,
                "Mean Dice": group["Dice"].mean(),
                "Std Dice": group["Dice"].std(),
                "Mean IoU": group["IoU"].mean(),
                "Std IoU": group["IoU"].std(),
                "Mean HD95": group["HD95"].mean(),
                "Std HD95": group["HD95"].std(),
                "Mean Params": group["Params"].mean(),
                "Std Params": group["Params"].std(),
                "Mean FLOPs": group["FLOPs"].mean(),
                "Std FLOPs": group["FLOPs"].std(),
                "Mean FPS": group["FPS"].mean(),
                "Std FPS": group["FPS"].std(),
                "Mean Inf (s)": group["Inf (s)"].mean(),
                "Std Inf (s)": group["Inf (s)"].std(),
                "Mean Search Time (s)": group.get("Search Time (s)", pd.Series(dtype=float)).mean(),
                "Std Search Time (s)": group.get("Search Time (s)", pd.Series(dtype=float)).std(),
            }
        )
    return pd.DataFrame(rows).reindex(columns=MEAN_STD_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready tables from existing outputs.")
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument("--save-root", type=str, default="statistics/outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    outputs_root = Path(args.outputs_root)
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    metrics = _read_metrics_rows(outputs_root) if outputs_root.exists() else pd.DataFrame()
    timing = _read_timing_rows(outputs_root) if outputs_root.exists() else pd.DataFrame()
    if metrics.empty and timing.empty:
        logging.warning("No outputs found. Empty statistics folder is still prepared at %s", save_root)
        _mean_std_table([]).to_csv(save_root / "table_mean_std_across_datasets.csv", index=False)
        return 0

    datasets = sorted(set(metrics.get("dataset", pd.Series(dtype=str)).dropna().astype(str)) | set(timing.get("dataset", pd.Series(dtype=str)).dropna().astype(str)))
    all_performance_tables: List[pd.DataFrame] = []
    for dataset in datasets:
        dataset_dir = save_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Processing tables for dataset: %s -> %s", dataset, dataset_dir)
        tables = _tables_for_dataset(dataset, metrics, timing)
        for filename, table in tables.items():
            output_path = dataset_dir / filename
            logging.info("Processing table: %s rows=%d -> %s", filename, len(table), output_path)
            table.to_csv(output_path, index=False)
            if filename != "table5_computational_cost.csv":
                all_performance_tables.append(table)
            logging.info("Saved table: %s", output_path)

    mean_std = _mean_std_table(all_performance_tables)
    mean_std_path = save_root / "table_mean_std_across_datasets.csv"
    logging.info("Processing table: table_mean_std_across_datasets.csv rows=%d -> %s", len(mean_std), mean_std_path)
    mean_std.to_csv(mean_std_path, index=False)
    logging.info("Saved table: %s", mean_std_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
