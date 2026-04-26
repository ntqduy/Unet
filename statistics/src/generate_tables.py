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
PERFORMANCE_COLUMNS = ["Dice", "IoU", "HD95", "Params", "FLOPs", "FPS", "Inf (s)", "Search Time (s)"]
TABLE2_COLUMNS = ["Group", "Method", *PERFORMANCE_COLUMNS, "Source Phase", "Raw Method", "Static Ratio"]
TABLE3_COLUMNS = ["Method", *PERFORMANCE_COLUMNS]
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

PGD_TEACHER_DIR = "unet_resnet152_teacher"
PGD_LOSS_TAG = "loss_seg_kd_sparsity"


def _path_parts(path: Path, outputs_root: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(outputs_root).parts
    except ValueError:
        return path.parts


def _is_pgd_focus_path(path: Path, outputs_root: Path) -> bool:
    parts = _path_parts(path, outputs_root)
    return len(parts) >= 5 and parts[0] == "pgd_unet" and parts[2] == PGD_TEACHER_DIR and parts[3] == PGD_LOSS_TAG


def _is_pgd_loss_metric_path(path: Path, outputs_root: Path) -> bool:
    parts = _path_parts(path, outputs_root)
    return len(parts) >= 5 and parts[0] == "pgd_unet" and parts[2] == PGD_TEACHER_DIR and str(parts[3]).startswith("loss_")


def _is_pgd_teacher_phase_metric_path(path: Path, outputs_root: Path) -> bool:
    parts = _path_parts(path, outputs_root)
    return len(parts) >= 5 and parts[0] == "pgd_unet" and parts[2] == PGD_TEACHER_DIR and parts[3] == "1_teacher"


def _loss_scope_from_path(path: Path, outputs_root: Path) -> str:
    parts = _path_parts(path, outputs_root)
    if len(parts) >= 4 and parts[0] == "pgd_unet" and parts[2] == PGD_TEACHER_DIR:
        return parts[3]
    return ""


def _is_basic_metric_path(path: Path, outputs_root: Path) -> bool:
    parts = _path_parts(path, outputs_root)
    return not parts or parts[0] != "pgd_unet"


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


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _fmt_ratio(value: Any) -> str:
    number = _safe_float(value)
    if np.isnan(number):
        return "auto"
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _row_prune_method(row: Dict[str, Any]) -> str:
    return _clean_text(row.get("prune_method") or row.get("config_prune_method") or row.get("method")).lower()


def _row_static_ratio(row: Dict[str, Any]) -> float:
    return _safe_float(
        row.get(
            "static_prune_ratio",
            row.get("config_static_prune_ratio", row.get("pruning_ratio", row.get("prune_ratio"))),
        )
    )


def _method_display(raw_method: str, ratio: Any = np.nan, *, kd: bool = False, summary: str = "") -> str:
    raw_method = str(raw_method or "").lower()
    if summary:
        label = summary
    elif raw_method == "static":
        label = f"Static r={_fmt_ratio(ratio)}"
    elif raw_method == "middle_static":
        label = f"Middle static r={_fmt_ratio(ratio)}"
    elif raw_method == "kneedle":
        label = "Kneedle"
    elif raw_method == "otsu":
        label = "Otsu"
    elif raw_method == "gmm":
        label = "GMM"
    elif raw_method == "middle_kneedle":
        label = "Middle-Kneedle"
    elif raw_method == "middle_otsu":
        label = "Middle-Otsu"
    elif raw_method == "middle_gmm":
        label = "Middle-GMM"
    else:
        label = raw_method.replace("_", " ").title() if raw_method else "Unknown"
    return f"{label} + KD" if kd and "KD" not in label else label


def _metric_from_row(row: Dict[str, Any], name_key: str, name: str) -> Dict[str, Any]:
    metric = _metric_row(row, name_key, name)
    return metric


def _best_by_dice(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not rows:
        return None
    def score(item: Dict[str, Any]) -> tuple[float, float]:
        dice = _safe_float(item.get("dice"))
        hd95 = _safe_float(item.get("hd95"))
        return (-np.inf if np.isnan(dice) else dice, -np.inf if np.isnan(hd95) else -hd95)

    return max(rows, key=score)


def _search_time_for(raw_method: str, ratio: float, timing: pd.DataFrame) -> float:
    if timing.empty:
        return float("nan")
    frame = timing.copy()
    if "method" in frame.columns:
        frame = frame[frame["method"].fillna("").astype(str).str.lower().eq(str(raw_method).lower())]
    if frame.empty:
        return float("nan")
    if str(raw_method).lower() in {"static", "middle_static"} and "static_prune_ratio" in frame.columns and not np.isnan(ratio):
        ratios = pd.to_numeric(frame["static_prune_ratio"], errors="coerce")
        ratio_frame = frame[(ratios - ratio).abs() < 1e-9]
        if not ratio_frame.empty:
            frame = ratio_frame
    values = pd.to_numeric(frame.get("search_time_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else float("nan")


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
            allow_loss_table_metric = csv_path.name == "student_metrics.csv" and _is_pgd_loss_metric_path(csv_path, outputs_root)
            allow_teacher_metric = csv_path.name == "teacher_metrics.csv" and _is_pgd_teacher_phase_metric_path(csv_path, outputs_root)
            if csv_path.name != "basic_metrics.csv" and not _is_pgd_focus_path(csv_path, outputs_root) and not allow_loss_table_metric and not allow_teacher_metric:
                logging.info("Skip non-target PGD metrics CSV: %s", csv_path)
                continue
            if csv_path.name == "basic_metrics.csv" and not _is_basic_metric_path(csv_path, outputs_root):
                logging.info("Skip PGD basic_metrics outside baseline table: %s", csv_path)
                continue
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
                row["source_file"] = csv_path.name
                row["is_focus_loss"] = bool(_is_pgd_focus_path(csv_path, outputs_root) or allow_teacher_metric)
                row["loss_scope"] = _loss_scope_from_path(csv_path, outputs_root)
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
        if not _is_pgd_focus_path(csv_path, outputs_root):
            logging.info("Skip timing outside target PGD loss tag: %s", csv_path)
            continue
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
        if not _is_pgd_focus_path(json_path, outputs_root):
            logging.info("Skip search-time JSON outside target PGD loss tag: %s", json_path)
            continue
        logging.info("Reading pruning search-time JSON: %s", json_path)
        payload = _read_json(json_path)
        if not payload:
            continue
        rows.append(
            {
                "dataset": _infer_dataset(json_path, payload, outputs_root),
                "method": payload.get("prune_method") or payload.get("prune_strategy") or "pruning",
                "static_prune_ratio": payload.get("static_prune_ratio", np.nan),
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
    display_name = str(row.get("loss_method") or row.get("config_loss_method") or row.get("distill_loss_method") or row.get("config_distill_loss_method") or "").strip()
    if display_name and display_name.lower() != "nan":
        label = display_name.upper() if display_name.lower() in {"ce", "dice", "kl", "mse"} else display_name.replace("_", " + ").title()
        kd = _safe_int(row.get("use_kd_output", row.get("config_use_kd_output", 0)))
        return f"Proposed + {label} KD" if kd else label
    loss_tag = str(row.get("loss_tag") or row.get("config_loss_tag") or row.get("loss_scope") or "").lower()
    if loss_tag and loss_tag != "nan":
        label = loss_tag.replace("loss_", "").replace("_", " + ")
        return label.title()
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


def _table2_entry(
    row: Dict[str, Any],
    group: str,
    method: str,
    timing: pd.DataFrame,
    *,
    raw_method: str | None = None,
    source_phase: str | None = None,
    ratio: float | None = None,
) -> Dict[str, Any]:
    resolved_raw_method = raw_method or _row_prune_method(row)
    resolved_ratio = _row_static_ratio(row) if ratio is None else ratio
    entry = _metric_row(row, "Method", method)
    entry["Group"] = group
    entry["Source Phase"] = source_phase or _clean_text(row.get("phase"))
    entry["Raw Method"] = resolved_raw_method
    entry["Static Ratio"] = _fmt_ratio(resolved_ratio)
    if np.isnan(entry.get("Search Time (s)", np.nan)):
        entry["Search Time (s)"] = _search_time_for(resolved_raw_method, resolved_ratio, timing)
    if group == "Reference":
        entry["Search Time (s)"] = 0.0
        entry["Static Ratio"] = "not used"
    return entry


def _build_pruning_table2_rows(dataset_metrics: pd.DataFrame, dataset_timing: pd.DataFrame) -> List[Dict[str, Any]]:
    if dataset_metrics.empty:
        return []

    records = [row.to_dict() for _, row in dataset_metrics.iterrows()]
    teacher_rows = [row for row in records if str(row.get("phase") or "").lower() == "teacher"]
    pruning_rows = [row for row in records if str(row.get("phase") or "").lower() == "pruning" and _row_prune_method(row)]
    student_rows = [row for row in records if str(row.get("phase") or "").lower() == "student" and _row_prune_method(row)]

    rows: List[Dict[str, Any]] = []
    teacher = _best_by_dice(teacher_rows)
    if teacher:
        rows.append(_table2_entry(teacher, "Reference", "Teacher (UNet-ResNet152)", dataset_timing, raw_method="teacher", source_phase="teacher"))

    static_rows = [row for row in pruning_rows if _row_prune_method(row) == "static"]
    static_rows = sorted(static_rows, key=lambda row: (_row_static_ratio(row), -_safe_float(row.get("dice"))))
    seen_static = set()
    for row in static_rows:
        ratio = _row_static_ratio(row)
        ratio_key = _fmt_ratio(ratio)
        if ratio_key in seen_static:
            continue
        seen_static.add(ratio_key)
        rows.append(
            _table2_entry(
                row,
                "Static pruning",
                _method_display("static", ratio),
                dataset_timing,
                raw_method="static",
                source_phase="pruning",
                ratio=ratio,
            )
        )

    for raw_method in ("kneedle", "otsu", "gmm"):
        best = _best_by_dice([row for row in pruning_rows if _row_prune_method(row) == raw_method])
        if best:
            rows.append(_table2_entry(best, "Adaptive threshold", _method_display(raw_method), dataset_timing, raw_method=raw_method, source_phase="pruning"))

    channel_adaptive = [row for row in pruning_rows if _row_prune_method(row) in {"kneedle", "otsu", "gmm"}]
    block_adaptive = [row for row in pruning_rows if _row_prune_method(row) in {"middle_kneedle", "middle_otsu", "middle_gmm"}]
    best_channel = _best_by_dice(channel_adaptive)
    best_block = _best_by_dice(block_adaptive)
    if best_channel:
        rows.append(
            _table2_entry(
                best_channel,
                "Pruning granularity",
                "Channel-wise (best adaptive)",
                dataset_timing,
                raw_method=_row_prune_method(best_channel),
                source_phase="pruning",
            )
        )
    if best_block:
        rows.append(
            _table2_entry(
                best_block,
                "Pruning granularity",
                "Block-level (middle, best adaptive)",
                dataset_timing,
                raw_method=_row_prune_method(best_block),
                source_phase="pruning",
            )
        )
        rows.append(
            _table2_entry(
                best_block,
                "Proposed system",
                "Proposed (Adaptive + Block)",
                dataset_timing,
                raw_method=_row_prune_method(best_block),
                source_phase="pruning",
            )
        )

    best_student = _best_by_dice([row for row in student_rows if _row_prune_method(row) in {"middle_kneedle", "middle_otsu", "middle_gmm"}])
    if best_student is None:
        best_student = _best_by_dice(student_rows)
    if best_student:
        rows.append(
            _table2_entry(
                best_student,
                "Proposed system",
                "Proposed + KD",
                dataset_timing,
                raw_method=_row_prune_method(best_student),
                source_phase="student",
            )
        )
    return rows


def _tables_for_dataset(dataset: str, metrics: pd.DataFrame, timing: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    dataset_metrics = _test_rows(metrics[metrics["dataset"].astype(str).eq(dataset)]) if not metrics.empty else pd.DataFrame()
    focus_metrics = dataset_metrics[
        dataset_metrics.get("is_focus_loss", pd.Series([False] * len(dataset_metrics))).astype(bool)
    ].copy() if not dataset_metrics.empty else pd.DataFrame()
    table1_rows = []
    table3_rows = []
    table4_rows = []

    preferred_sources = {"teacher_metrics.csv", "pruning_metrics.csv", "student_metrics.csv", "basic_metrics.csv"}
    if "source_file" in dataset_metrics.columns and dataset_metrics["source_file"].isin(preferred_sources).any():
        dataset_metrics = dataset_metrics[dataset_metrics["source_file"].isin(preferred_sources)].copy()
    if "source_file" in focus_metrics.columns and focus_metrics["source_file"].isin(preferred_sources).any():
        focus_metrics = focus_metrics[focus_metrics["source_file"].isin(preferred_sources)].copy()

    for _, row_series in dataset_metrics.iterrows():
        row = row_series.to_dict()
        phase = str(row.get("phase") or "").lower()
        model_name = str(row.get("model_name") or row.get("model") or "unknown")
        prune_method = _row_prune_method(row)
        if phase == "basic" or (not prune_method and "pgd_unet" not in str(row.get("source_path", ""))):
            table1_rows.append({key: value for key, value in _metric_row(row, "Model", model_name).items() if key in TABLE1_COLUMNS})
        if prune_method and phase == "student":
            table3_rows.append(_metric_row(row, "Method", _loss_method(row)))

    for _, row_series in focus_metrics.iterrows():
        row = row_series.to_dict()
        phase = str(row.get("phase") or "").lower()
        prune_method = _row_prune_method(row)
        if prune_method and phase == "student":
            table4_rows.append(_metric_row(row, "Component", _component(row)))

    dataset_timing = timing[timing["dataset"].astype(str).eq(dataset)] if not timing.empty else pd.DataFrame()
    table2_rows = _build_pruning_table2_rows(focus_metrics, dataset_timing)
    table5_rows = []
    for _, row_series in dataset_timing.iterrows():
        row = row_series.to_dict()
        raw_method = _clean_text(row.get("method") or row.get("prune_method") or row.get("phase") or "unknown").lower()
        method = _method_display(raw_method, row.get("static_prune_ratio", np.nan))
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

    table2 = pd.DataFrame(table2_rows)
    if not table2.empty:
        table2 = table2.drop_duplicates(subset=["Group", "Method"], keep="first")

    return {
        "table1_baseline.csv": _dedupe_best(table1_rows, "Model").reindex(columns=TABLE1_COLUMNS),
        "table2_pruning.csv": table2.reindex(columns=TABLE2_COLUMNS),
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


def _format_mean_std(mean_value: float, std_value: float) -> str:
    if np.isnan(mean_value):
        return "NaN"
    if np.isnan(std_value):
        std_value = 0.0
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def _table1_baseline_mean_std(table1_tables: List[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [frame.copy() for frame in table1_tables if not frame.empty]
    if not frames:
        formatted_columns = ["Model", "Datasets", *TABLE1_COLUMNS[1:]]
        numeric_columns = ["Model", "Datasets"]
        for metric in TABLE1_COLUMNS[1:]:
            numeric_columns.extend([f"Mean {metric}", f"Std {metric}"])
        return pd.DataFrame(columns=formatted_columns), pd.DataFrame(columns=numeric_columns)

    combined = pd.concat(frames, ignore_index=True)
    rows_formatted: List[Dict[str, Any]] = []
    rows_numeric: List[Dict[str, Any]] = []
    for model, group in combined.groupby("Model", dropna=True):
        dataset_count = int(group["Dataset"].nunique()) if "Dataset" in group.columns else int(len(group))
        formatted_row: Dict[str, Any] = {"Model": model, "Datasets": dataset_count}
        numeric_row: Dict[str, Any] = {"Model": model, "Datasets": dataset_count}
        for metric in TABLE1_COLUMNS[1:]:
            values = pd.to_numeric(group.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            mean_value = float(values.mean()) if not values.empty else float("nan")
            std_value = float(values.std()) if len(values) > 1 else 0.0 if len(values) == 1 else float("nan")
            formatted_row[metric] = _format_mean_std(mean_value, std_value)
            numeric_row[f"Mean {metric}"] = mean_value
            numeric_row[f"Std {metric}"] = std_value
        rows_formatted.append(formatted_row)
        rows_numeric.append(numeric_row)

    formatted = pd.DataFrame(rows_formatted).sort_values("Model").reset_index(drop=True)
    numeric = pd.DataFrame(rows_numeric).sort_values("Model").reset_index(drop=True)
    return formatted, numeric


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

    raw_datasets = set(metrics.get("dataset", pd.Series(dtype=str)).dropna().astype(str)) | set(timing.get("dataset", pd.Series(dtype=str)).dropna().astype(str))
    datasets = sorted(dataset for dataset in raw_datasets if dataset != "unknown" and not Path(dataset).suffix)
    all_performance_tables: List[pd.DataFrame] = []
    table1_tables: List[pd.DataFrame] = []
    for dataset in datasets:
        dataset_dir = save_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Processing tables for dataset: %s -> %s", dataset, dataset_dir)
        tables = _tables_for_dataset(dataset, metrics, timing)
        for filename, table in tables.items():
            output_path = dataset_dir / filename
            logging.info("Processing table: %s rows=%d -> %s", filename, len(table), output_path)
            table.to_csv(output_path, index=False)
            if filename == "table1_baseline.csv" and not table.empty:
                table_for_mean = table.copy()
                table_for_mean.insert(0, "Dataset", dataset)
                table1_tables.append(table_for_mean)
            if filename != "table5_computational_cost.csv":
                all_performance_tables.append(table)
            logging.info("Saved table: %s", output_path)

    table1_mean_std, table1_numeric = _table1_baseline_mean_std(table1_tables)
    table1_mean_std_path = save_root / "table_mean_baseline.csv"
    table1_numeric_path = save_root / "table_mean_baseline_numeric.csv"
    logging.info("Processing table: table_mean_baseline.csv mean+std rows=%d -> %s", len(table1_mean_std), table1_mean_std_path)
    table1_mean_std.to_csv(table1_mean_std_path, index=False)
    table1_numeric.to_csv(table1_numeric_path, index=False)
    logging.info("Saved table: %s", table1_mean_std_path)
    logging.info("Saved table: %s", table1_numeric_path)

    mean_std = _mean_std_table(all_performance_tables)
    mean_std_path = save_root / "table_mean_std_across_datasets.csv"
    logging.info("Processing table: table_mean_std_across_datasets.csv rows=%d -> %s", len(mean_std), mean_std_path)
    mean_std.to_csv(mean_std_path, index=False)
    logging.info("Saved table: %s", mean_std_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
