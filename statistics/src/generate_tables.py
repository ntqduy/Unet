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


PGD_TEACHER_DIR = "unet_resnet152_teacher"
PGD_LOSS_TAG = "loss_seg_kd"
PGD_LOSS_TAGS = {PGD_LOSS_TAG}
PGD_COMPARISON_LOSS_TAGS = {PGD_LOSS_TAG, "loss_seg_only"}
CHANNEL_METHODS = {"static", "kneedle", "otsu", "gmm"}
MIDDLE_METHODS = {"middle_static", "middle_kneedle", "middle_otsu", "middle_gmm"}
FULL_METHODS = {"full_static", "full_kneedle", "full_otsu", "full_gmm"}

TABLE1_COLUMNS = ["Model", "Dice", "IoU", "HD95", "Params", "FLOPs", "FPS", "Inf (s)"]
PERFORMANCE_COLUMNS = ["Dice", "IoU", "HD95", "Params", "FLOPs", "FPS", "Inf (s)", "Search Time (s)"]
TABLE3_COLUMNS = ["Method", *PERFORMANCE_COLUMNS]
TABLE4_COLUMNS = ["Component", "Dice", "IoU", "HD95", "Params", "FLOPs", "FPS", "Inf (s)", "Search Time (s)"]
TABLE5_COLUMNS = ["Method", "Pruning Time (s)", "Search Time (s)", "Training Time (s)", "Inference Time (s)", "Total Time (s)"]
TABLE5_METHOD_DIRS = [
    ("Static pruning r = 0.5", Path(PGD_LOSS_TAG) / "output_s1_static_0.5_no" / "3_student"),
    ("Kneedle", Path(PGD_LOSS_TAG) / "output_s2_kneedle_auto_no" / "3_student"),
    ("Otsu", Path(PGD_LOSS_TAG) / "output_s3_otsu_auto_no" / "3_student"),
    ("GMM", Path(PGD_LOSS_TAG) / "output_s4_gmm_auto_no" / "3_student"),
    ("Middle Static Pruning (r = 0.5)", Path(PGD_LOSS_TAG) / "output_s5_middle_static_0.5_no" / "3_student"),
    ("Middle Kneedle", Path(PGD_LOSS_TAG) / "output_s6_middle_kneedle_auto_no" / "3_student"),
    ("Middle Otsu", Path(PGD_LOSS_TAG) / "output_s7_middle_otsu_auto_no" / "3_student"),
    ("Middle GMM", Path(PGD_LOSS_TAG) / "output_s8_middle_gmm_auto_no" / "3_student"),
    ("Full Static Block (r = 0.5)", Path(PGD_LOSS_TAG) / "output_s9_full_static_0.5_no" / "3_student"),
    ("Full Kneedle Block", Path(PGD_LOSS_TAG) / "output_s10_full_kneedle_auto_no" / "3_student"),
    ("Full Otsu Block", Path(PGD_LOSS_TAG) / "output_s11_full_otsu_auto_no" / "3_student"),
    ("Full GMM Block", Path(PGD_LOSS_TAG) / "output_s12_full_gmm_auto_no" / "3_student"),
]
TABLE6_COLUMNS = [
    "Method",
    "Dice $\\uparrow$",
    "IoU $\\uparrow$",
    "HD95 $\\downarrow$",
    "Params (M)",
    "FPS $\\uparrow$",
    "Inf (s) $\\downarrow$",
]
TABLE2_COLUMNS = TABLE6_COLUMNS
TABLE2_METHOD_DIRS = [
    ("Teacher (UNet-ResNet152)", Path("1_teacher")),
    ("Static pruning r = 0.5", Path(PGD_LOSS_TAG) / "output_s1_static_0.5_no" / "2_pruning"),
    ("Kneedle", Path(PGD_LOSS_TAG) / "output_s2_kneedle_auto_no" / "2_pruning"),
    ("Otsu", Path(PGD_LOSS_TAG) / "output_s3_otsu_auto_no" / "2_pruning"),
    ("GMM", Path(PGD_LOSS_TAG) / "output_s4_gmm_auto_no" / "2_pruning"),
    ("Middle Static Pruning (r = 0.5)", Path(PGD_LOSS_TAG) / "output_s5_middle_static_0.5_no" / "2_pruning"),
    ("Middle Kneedle", Path(PGD_LOSS_TAG) / "output_s6_middle_kneedle_auto_no" / "2_pruning"),
    ("Middle Otsu", Path(PGD_LOSS_TAG) / "output_s7_middle_otsu_auto_no" / "2_pruning"),
    ("Middle GMM", Path(PGD_LOSS_TAG) / "output_s8_middle_gmm_auto_no" / "2_pruning"),
    ("Full Static Block (r = 0.5)", Path(PGD_LOSS_TAG) / "output_s9_full_static_0.5_no" / "2_pruning"),
    ("Full Kneedle Block", Path(PGD_LOSS_TAG) / "output_s10_full_kneedle_auto_no" / "2_pruning"),
    ("Full Otsu Block", Path(PGD_LOSS_TAG) / "output_s11_full_otsu_auto_no" / "2_pruning"),
    ("Full GMM Block", Path(PGD_LOSS_TAG) / "output_s12_full_gmm_auto_no" / "2_pruning"),
]
TABLE6_METHOD_DIRS = [
    ("Teacher (UNet-ResNet152)", Path("1_teacher")),
    ("Static pruning r = 0.5", Path(PGD_LOSS_TAG) / "output_s1_static_0.5_no" / "3_student"),
    ("Kneedle", Path(PGD_LOSS_TAG) / "output_s2_kneedle_auto_no" / "3_student"),
    ("Otsu", Path(PGD_LOSS_TAG) / "output_s3_otsu_auto_no" / "3_student"),
    ("GMM", Path(PGD_LOSS_TAG) / "output_s4_gmm_auto_no" / "3_student"),
    ("Middle Static Pruning (r = 0.5)", Path(PGD_LOSS_TAG) / "output_s5_middle_static_0.5_no" / "3_student"),
    ("Middle Kneedle", Path(PGD_LOSS_TAG) / "output_s6_middle_kneedle_auto_no" / "3_student"),
    ("Middle Otsu", Path(PGD_LOSS_TAG) / "output_s7_middle_otsu_auto_no" / "3_student"),
    ("Middle GMM", Path(PGD_LOSS_TAG) / "output_s8_middle_gmm_auto_no" / "3_student"),
    ("Full Static Block (r = 0.5)", Path(PGD_LOSS_TAG) / "output_s9_full_static_0.5_no" / "3_student"),
    ("Full Kneedle Block", Path(PGD_LOSS_TAG) / "output_s10_full_kneedle_auto_no" / "3_student"),
    ("Full Otsu Block", Path(PGD_LOSS_TAG) / "output_s11_full_otsu_auto_no" / "3_student"),
    ("Full GMM Block", Path(PGD_LOSS_TAG) / "output_s12_full_gmm_auto_no" / "3_student"),
]
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
MEAN_TABLE_METRICS = ["Dice", "IoU", "HD95", "Params", "FLOPs", "FPS", "Inf (s)", "Search Time (s)"]
TABLE7_COLUMNS = [
    "Method",
    "Runs",
    "Datasets",
    *MEAN_TABLE_METRICS,
]
TABLE7_NUMERIC_COLUMNS = [
    "Method",
    "Runs",
    "Datasets",
    *[f"Mean {metric}" for metric in MEAN_TABLE_METRICS],
    *[f"Std {metric}" for metric in MEAN_TABLE_METRICS],
]
TABLE8_COLUMNS = [
    "Component",
    "Runs",
    "Datasets",
    *MEAN_TABLE_METRICS,
]
TABLE8_NUMERIC_COLUMNS = [
    "Component",
    "Runs",
    "Datasets",
    *[f"Mean {metric}" for metric in MEAN_TABLE_METRICS],
    *[f"Std {metric}" for metric in MEAN_TABLE_METRICS],
]
TABLE9_METRICS = ["Dice", "IoU", "HD95", "Params (M)", "FLOPs", "FPS", "Inf (s)", "Search Time (s)"]
TABLE9_COLUMNS = ["Method", "Runs", "Datasets", *TABLE9_METRICS]
TABLE9_NUMERIC_COLUMNS = [
    "Method",
    "Runs",
    "Datasets",
    *[f"Mean {metric}" for metric in TABLE9_METRICS],
    *[f"Std {metric}" for metric in TABLE9_METRICS],
]
TABLE10_METRICS = TABLE2_COLUMNS[1:]
TABLE10_COLUMNS = ["Method", "Runs", "Datasets", *TABLE10_METRICS]
TABLE10_NUMERIC_COLUMNS = [
    "Method",
    "Runs",
    "Datasets",
    *[f"Mean {metric}" for metric in TABLE10_METRICS],
    *[f"Std {metric}" for metric in TABLE10_METRICS],
]
TABLE11_METRICS = TABLE5_COLUMNS[1:]
TABLE11_COLUMNS = ["Method", "Runs", "Datasets", *TABLE11_METRICS]
TABLE11_NUMERIC_COLUMNS = [
    "Method",
    "Runs",
    "Datasets",
    *[f"Mean {metric}" for metric in TABLE11_METRICS],
    *[f"Std {metric}" for metric in TABLE11_METRICS],
]

def _path_parts(path: Path, outputs_root: Path) -> tuple[str, ...]:
    try:
        return path.relative_to(outputs_root).parts
    except ValueError:
        return path.parts


def _is_pgd_focus_path(path: Path, outputs_root: Path) -> bool:
    parts = _path_parts(path, outputs_root)
    return len(parts) >= 5 and parts[0] == "pgd_unet" and parts[2] == PGD_TEACHER_DIR and parts[3] in PGD_LOSS_TAGS


def _is_pgd_loss_metric_path(path: Path, outputs_root: Path) -> bool:
    parts = _path_parts(path, outputs_root)
    return len(parts) >= 5 and parts[0] == "pgd_unet" and parts[2] == PGD_TEACHER_DIR and parts[3] in PGD_COMPARISON_LOSS_TAGS


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
        if value is None:
            return float("nan")
        try:
            if pd.isna(value):
                return float("nan")
        except (TypeError, ValueError):
            pass
        if value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _row_number(row: Dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        if value == "":
            continue
        number = _safe_float(value)
        if not np.isnan(number):
            return number
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


def _row_loss_scope(row: Dict[str, Any]) -> str:
    return (
        _clean_text(row.get("loss_scope"))
        or _clean_text(row.get("loss_tag"))
        or _clean_text(row.get("config_loss_tag"))
    ).lower()


def _method_display(raw_method: str, ratio: Any = np.nan, *, kd: bool = False, summary: str = "") -> str:
    raw_method = str(raw_method or "").lower()
    if summary:
        label = summary
    elif raw_method == "static":
        label = f"Static Blueprint r={_fmt_ratio(ratio)}"
    elif raw_method == "kneedle":
        label = "Kneedle Blueprint"
    elif raw_method == "otsu":
        label = "Otsu Blueprint"
    elif raw_method == "gmm":
        label = "GMM Blueprint"
    elif raw_method == "middle_static":
        label = f"Middle-Static Conv2 r={_fmt_ratio(ratio)}"
    elif raw_method == "middle_kneedle":
        label = "Middle-Kneedle Conv2"
    elif raw_method == "middle_otsu":
        label = "Middle-Otsu Conv2"
    elif raw_method == "middle_gmm":
        label = "Middle-GMM Conv2"
    elif raw_method == "full_static":
        label = f"Full-Static Block r={_fmt_ratio(ratio)}"
    elif raw_method == "full_kneedle":
        label = "Full-Kneedle Block"
    elif raw_method == "full_otsu":
        label = "Full-Otsu Block"
    elif raw_method == "full_gmm":
        label = "Full-GMM Block"
    else:
        label = raw_method.replace("_", " ").title() if raw_method else "Unknown"
    return f"{label} + KD" if kd and "KD" not in label else label


def _method_from_output_dir(output_dir: Path) -> tuple[str, float]:
    parts = output_dir.name.split("_")
    if len(parts) >= 3 and parts[0] == "output" and parts[1].startswith("s") and parts[1][1:].isdigit():
        parts = [parts[0], *parts[2:]]
    if len(parts) >= 3 and parts[1] == "middle":
        method = f"middle_{parts[2]}"
        ratio = _safe_float(parts[3]) if len(parts) >= 5 and parts[3] != "auto" else np.nan
        return method, ratio
    if len(parts) >= 3 and parts[1] == "full":
        method = f"full_{parts[2]}"
        ratio = _safe_float(parts[3]) if len(parts) >= 5 and parts[3] != "auto" else np.nan
        return method, ratio
    if len(parts) >= 2 and parts[0] == "output":
        method = parts[1]
        ratio = _safe_float(parts[2]) if len(parts) >= 4 and parts[2] != "auto" else np.nan
        return method, ratio
    return output_dir.name, np.nan


def _method_sort_key(raw_method: str, ratio: float) -> tuple[int, float, str]:
    priorities = {
        "static": 0,
        "kneedle": 1,
        "otsu": 2,
        "gmm": 3,
        "middle_static": 4,
        "middle_kneedle": 5,
        "middle_otsu": 6,
        "middle_gmm": 7,
        "full_static": 8,
        "full_kneedle": 9,
        "full_otsu": 10,
        "full_gmm": 11,
    }
    method = str(raw_method or "").lower()
    ratio_key = _safe_float(ratio)
    if np.isnan(ratio_key):
        ratio_key = float("inf")
    return priorities.get(method, 99), ratio_key, method


def _discover_method_dirs(
    outputs_root: Path,
    dataset: str,
    phase_dir: str,
    fallback_dirs: List[tuple[str, Path]],
    *,
    include_teacher: bool = False,
) -> List[tuple[str, Path]]:
    base_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR
    loss_root = base_root / PGD_LOSS_TAG
    discovered = []
    if loss_root.is_dir():
        for output_dir in loss_root.iterdir():
            if not output_dir.is_dir() or not output_dir.name.startswith("output_"):
                continue
            phase_path = output_dir / phase_dir
            if not phase_path.exists():
                continue
            raw_method, ratio = _method_from_output_dir(output_dir)
            discovered.append(
                (
                    _method_sort_key(raw_method, ratio),
                    _method_display(raw_method, ratio),
                    Path(PGD_LOSS_TAG) / output_dir.name / phase_dir,
                )
            )
    if not discovered:
        return fallback_dirs
    method_dirs = sorted(discovered, key=lambda item: item[0])
    rows = [(label, relative_dir) for _, label, relative_dir in method_dirs]
    if include_teacher:
        return [("Teacher (UNet-ResNet152)", Path("1_teacher")), *rows]
    return rows


def _method_group_component(raw_method: str) -> str:
    method = str(raw_method or "").lower()
    if method in CHANNEL_METHODS:
        return "Blueprint"
    if method in MIDDLE_METHODS:
        return "Middle Conv2"
    if method in FULL_METHODS:
        return "Full Block"
    return "Other"


def _metric_from_row(row: Dict[str, Any], name_key: str, name: str) -> Dict[str, Any]:
    metric = _metric_row(row, name_key, name)
    return metric


def _best_by_dice(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not rows:
        return None
    def score(item: Dict[str, Any]) -> tuple[float, float]:
        dice = _row_number(item, "dice", "Dice", "val_dice", "val_macro_dice", "test_dice")
        hd95 = _row_number(item, "hd95", "HD95", "val_hd95", "test_hd95")
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
            allow_loss_table_metric = csv_path.name in {"student_metrics.csv", "metrics_summary.csv"} and _is_pgd_loss_metric_path(csv_path, outputs_root)
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
        "Dice": _row_number(row, "dice", "Dice", "val_dice", "val_macro_dice", "test_dice"),
        "IoU": _row_number(row, "iou", "IoU", "val_iou", "test_iou"),
        "HD95": _row_number(row, "hd95", "HD95", "val_hd95", "test_hd95"),
        "Params": _row_number(row, "params", "Params", "parameters", "num_params"),
        "FLOPs": _row_number(row, "flops", "FLOPs"),
        "FPS": _row_number(row, "fps", "FPS"),
        "Inf (s)": _row_number(row, "inference_time_seconds", "Inf (s)", "Inference Time (s)"),
        "Search Time (s)": _row_number(row, "search_time_seconds", "Search Time (s)"),
    }


def _params_to_millions(value: Any) -> float:
    number = _safe_float(value)
    if np.isnan(number):
        return number
    return number / 1e6 if abs(number) > 1e5 else number


def _best_test_row_from_csv(csv_path: Path) -> Dict[str, Any] | None:
    if not csv_path.is_file():
        logging.warning("Missing Table 6 metrics CSV: %s", csv_path)
        return None
    try:
        frame = pd.read_csv(csv_path)
    except Exception as error:
        logging.warning("Cannot read Table 6 metrics CSV: %s | %s", csv_path, error)
        return None
    if frame.empty:
        logging.warning("Table 6 metrics CSV is empty: %s", csv_path)
        return None
    if "split" in frame.columns:
        test_frame = frame[frame["split"].fillna("").astype(str).str.lower().eq("test")]
        if not test_frame.empty:
            frame = test_frame
    rows = [row.to_dict() for _, row in frame.iterrows()]
    return _best_by_dice(rows)


def _table6_row(method: str, row: Dict[str, Any] | None) -> Dict[str, Any]:
    if row is None:
        return {
            "Method": method,
            "Dice $\\uparrow$": np.nan,
            "IoU $\\uparrow$": np.nan,
            "HD95 $\\downarrow$": np.nan,
            "Params (M)": np.nan,
            "FPS $\\uparrow$": np.nan,
            "Inf (s) $\\downarrow$": np.nan,
        }
    return {
        "Method": method,
        "Dice $\\uparrow$": _row_number(row, "dice", "Dice", "val_dice", "val_macro_dice", "test_dice"),
        "IoU $\\uparrow$": _row_number(row, "iou", "IoU", "val_iou", "test_iou"),
        "HD95 $\\downarrow$": _row_number(row, "hd95", "HD95", "val_hd95", "test_hd95"),
        "Params (M)": _params_to_millions(_row_number(row, "params", "Params", "parameters", "num_params", "Params (M)")),
        "FPS $\\uparrow$": _row_number(row, "fps", "FPS"),
        "Inf (s) $\\downarrow$": _row_number(row, "inference_time_seconds", "Inf (s)", "Inference Time (s)"),
    }


def _table6_method_comparison(outputs_root: Path, dataset: str) -> pd.DataFrame:
    base_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR
    rows = []
    method_dirs = _discover_method_dirs(outputs_root, dataset, "3_student", TABLE6_METHOD_DIRS, include_teacher=True)
    for method, relative_dir in method_dirs:
        metrics_path = base_root / relative_dir / "metrics_summary.csv"
        rows.append(_table6_row(method, _best_test_row_from_csv(metrics_path)))
    return pd.DataFrame(rows).reindex(columns=TABLE6_COLUMNS)


def _timing_row(method: str, row: Dict[str, Any] | None) -> Dict[str, Any]:
    if row is None:
        return {
            "Method": method,
            "Pruning Time (s)": np.nan,
            "Search Time (s)": np.nan,
            "Training Time (s)": np.nan,
            "Inference Time (s)": np.nan,
            "Total Time (s)": np.nan,
        }
    return {
        "Method": method,
        "Pruning Time (s)": _safe_float(row.get("pruning_time_seconds")),
        "Search Time (s)": _safe_float(row.get("search_time_seconds")),
        "Training Time (s)": _safe_float(row.get("training_time_seconds")),
        "Inference Time (s)": _safe_float(row.get("inference_time_seconds")),
        "Total Time (s)": _safe_float(row.get("total_time_seconds")),
    }


def _first_timing_row_from_csv(csv_path: Path) -> Dict[str, Any] | None:
    if not csv_path.is_file():
        logging.warning("Missing Table 5 timing CSV: %s", csv_path)
        return None
    try:
        frame = pd.read_csv(csv_path)
    except Exception as error:
        logging.warning("Cannot read Table 5 timing CSV: %s | %s", csv_path, error)
        return None
    if frame.empty:
        logging.warning("Table 5 timing CSV is empty: %s", csv_path)
        return None
    if "phase" in frame.columns:
        student_frame = frame[frame["phase"].fillna("").astype(str).str.lower().eq("student")]
        if not student_frame.empty:
            frame = student_frame
    return frame.iloc[0].to_dict()


def _table5_timing_method_comparison(outputs_root: Path, dataset: str) -> pd.DataFrame:
    base_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR
    rows = []
    method_dirs = _discover_method_dirs(outputs_root, dataset, "3_student", TABLE5_METHOD_DIRS)
    for method, relative_dir in method_dirs:
        timing_path = base_root / relative_dir / "timing_summary.csv"
        rows.append(_timing_row(method, _first_timing_row_from_csv(timing_path)))
    return pd.DataFrame(rows).reindex(columns=TABLE5_COLUMNS)


def _table2_pruning_method_comparison(outputs_root: Path, dataset: str) -> pd.DataFrame:
    base_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR
    rows = []
    method_dirs = _discover_method_dirs(outputs_root, dataset, "2_pruning", TABLE2_METHOD_DIRS, include_teacher=True)
    for method, relative_dir in method_dirs:
        metrics_path = base_root / relative_dir / "metrics_summary.csv"
        rows.append(_table6_row(method, _best_test_row_from_csv(metrics_path)))
    return pd.DataFrame(rows).reindex(columns=TABLE2_COLUMNS)


def _table4_group_ablation_rows(dataset_metrics: pd.DataFrame) -> List[Dict[str, Any]]:
    if dataset_metrics.empty:
        return []
    records = [row.to_dict() for _, row in dataset_metrics.iterrows()]
    student_rows = [
        row
        for row in records
        if str(row.get("phase") or "").lower() == "student"
        and _row_loss_scope(row) in {"", PGD_LOSS_TAG}
        and _row_prune_method(row)
    ]
    if not student_rows:
        student_rows = [
            row
            for row in records
            if str(row.get("phase") or "").lower() == "student" and _row_prune_method(row)
        ]
    rows = []
    for component, methods in (
        ("Blueprint", CHANNEL_METHODS),
        ("Middle Conv2", MIDDLE_METHODS),
        ("Full Block", FULL_METHODS),
    ):
        best = _best_by_dice([row for row in student_rows if _row_prune_method(row) in methods])
        if best:
            rows.append(_metric_row(best, "Component", component))
    return rows


def _loss_method(row: Dict[str, Any]) -> str:
    display_name = str(row.get("loss_method") or row.get("config_loss_method") or row.get("distill_loss_method") or row.get("config_distill_loss_method") or "").strip()
    if display_name and display_name.lower() != "nan":
        label = display_name.upper() if display_name.lower() in {"ce", "dice", "kl", "mse"} else display_name.replace("_", " + ").title()
        kd = _safe_int(row.get("use_kd_output", row.get("config_use_kd_output", 0)))
        return f"Proposed + {label} KD" if kd else label
    loss_tag = _row_loss_scope(row)
    if loss_tag:
        if loss_tag == "loss_seg_only":
            return "Seg only"
        if loss_tag == "loss_seg_kd":
            return "Seg + KD"
        label = loss_tag.replace("loss_", "").replace("_", " + ")
        return label.title().replace("Kd", "KD")
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
    adaptive = method in {"kneedle", "otsu", "gmm", "middle_kneedle", "middle_otsu", "middle_gmm", "full_kneedle", "full_otsu", "full_gmm"}
    block = method.startswith("middle_") or method.startswith("full_")
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


def _preferred_metrics_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "source_file" not in frame.columns:
        return frame.copy()
    preferred_sources = {"teacher_metrics.csv", "pruning_metrics.csv", "student_metrics.csv", "basic_metrics.csv"}
    if frame["source_file"].isin(preferred_sources).any():
        return frame[frame["source_file"].isin(preferred_sources)].copy()
    return frame.copy()


def _preferred_loss_comparison_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    selected = frame.copy()
    selected["_dataset_key"] = selected.get("dataset", pd.Series([""] * len(selected), index=selected.index)).fillna("").astype(str)
    selected["_phase_key"] = selected.get("phase", pd.Series([""] * len(selected), index=selected.index)).fillna("").astype(str).str.lower()
    selected["_loss_scope_key"] = selected.apply(lambda row: _row_loss_scope(row.to_dict()), axis=1)
    selected["_prune_method_key"] = selected.apply(lambda row: _row_prune_method(row.to_dict()), axis=1)
    selected["_static_ratio_key"] = selected.apply(lambda row: _fmt_ratio(_row_static_ratio(row.to_dict())), axis=1)
    source_priority = {
        "metrics_summary.csv": 0,
        "student_metrics.csv": 1,
        "pipeline_metrics.csv": 2,
        "pruning_metrics.csv": 3,
    }
    selected["_source_priority"] = (
        selected.get("source_file", pd.Series([""] * len(selected), index=selected.index))
        .map(source_priority)
        .fillna(99)
    )
    selected = selected.sort_values(["_source_priority", "_dataset_key", "_loss_scope_key", "_prune_method_key", "_static_ratio_key"])
    selected = selected.drop_duplicates(
        subset=["_dataset_key", "_phase_key", "_loss_scope_key", "_prune_method_key", "_static_ratio_key"],
        keep="first",
    )
    return selected.drop(
        columns=[
            "_dataset_key",
            "_phase_key",
            "_loss_scope_key",
            "_prune_method_key",
            "_static_ratio_key",
            "_source_priority",
        ],
        errors="ignore",
    ).reset_index(drop=True)


def _mean_metric_table(
    rows: Iterable[Dict[str, Any]],
    group_key: str,
    columns: List[str],
    metrics: List[str] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    metrics = metrics or MEAN_TABLE_METRICS
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame.get(metric, pd.Series(dtype=float)), errors="coerce")
    if "Dataset" not in frame.columns:
        frame["Dataset"] = "unknown"
    dataset_level = (
        frame.groupby([group_key, "Dataset"], dropna=True)[metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    rows_out: List[Dict[str, Any]] = []
    for group_name, group_frame in dataset_level.groupby(group_key, dropna=True):
        source_rows = frame[frame[group_key].astype(str).eq(str(group_name))]
        row: Dict[str, Any] = {
            group_key: group_name,
            "Runs": int(len(source_rows)),
            "Datasets": int(group_frame["Dataset"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(group_frame[metric], errors="coerce").dropna()
            row[f"Mean {metric}"] = float(values.mean()) if not values.empty else float("nan")
            row[f"Std {metric}"] = float(values.std()) if len(values) > 1 else 0.0 if len(values) == 1 else float("nan")
        rows_out.append(row)
    return pd.DataFrame(rows_out).reindex(columns=columns)


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

    for raw_method in ("kneedle", "otsu", "gmm", "full_kneedle", "full_otsu", "full_gmm"):
        best = _best_by_dice([row for row in pruning_rows if _row_prune_method(row) == raw_method])
        if best:
            rows.append(_table2_entry(best, "Adaptive threshold", _method_display(raw_method), dataset_timing, raw_method=raw_method, source_phase="pruning"))

    channel_adaptive = [row for row in pruning_rows if _row_prune_method(row) in {"kneedle", "otsu", "gmm"}]
    block_adaptive = [row for row in pruning_rows if _row_prune_method(row) in {"middle_kneedle", "middle_otsu", "middle_gmm", "full_kneedle", "full_otsu", "full_gmm"}]
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

    best_student = _best_by_dice([row for row in student_rows if _row_prune_method(row) in {"middle_kneedle", "middle_otsu", "middle_gmm", "full_kneedle", "full_otsu", "full_gmm"}])
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
    loss_comparison_metrics = _preferred_loss_comparison_frame(dataset_metrics)
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

    for _, row_series in loss_comparison_metrics.iterrows():
        row = row_series.to_dict()
        phase = str(row.get("phase") or "").lower()
        prune_method = _row_prune_method(row)
        if prune_method and phase == "student":
            table3_rows.append(_metric_row(row, "Method", _loss_method(row)))

    table4_rows = _table4_group_ablation_rows(focus_metrics)
    if not table4_rows:
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


def _format_mean_std(mean_value: float, std_value: float) -> str:
    if np.isnan(mean_value):
        return "NaN"
    if np.isnan(std_value):
        std_value = 0.0
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def _format_mean_metric_table(
    numeric_table: pd.DataFrame,
    group_key: str,
    metrics: List[str],
    columns: List[str],
) -> pd.DataFrame:
    if numeric_table.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for _, row_series in numeric_table.iterrows():
        row = row_series.to_dict()
        runs = _safe_float(row.get("Runs"))
        datasets = _safe_float(row.get("Datasets"))
        formatted: Dict[str, Any] = {
            group_key: row.get(group_key),
            "Runs": int(runs) if not np.isnan(runs) else 0,
            "Datasets": int(datasets) if not np.isnan(datasets) else 0,
        }
        for metric in metrics:
            formatted[metric] = _format_mean_std(
                _safe_float(row.get(f"Mean {metric}")),
                _safe_float(row.get(f"Std {metric}")),
            )
        rows.append(formatted)
    return pd.DataFrame(rows).reindex(columns=columns)


def _sort_by_known_order(table: pd.DataFrame, key: str, order: Dict[str, int]) -> pd.DataFrame:
    if table.empty or key not in table.columns:
        return table
    return (
        table.assign(_order=table[key].map(order).fillna(9999))
        .sort_values(["_order", key])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def _table7_loss_mean(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty:
        return pd.DataFrame(columns=TABLE7_COLUMNS), pd.DataFrame(columns=TABLE7_NUMERIC_COLUMNS)
    frame = _preferred_loss_comparison_frame(_test_rows(metrics))
    rows: List[Dict[str, Any]] = []
    for _, row_series in frame.iterrows():
        row = row_series.to_dict()
        if str(row.get("phase") or "").lower() != "student" or not _row_prune_method(row):
            continue
        entry = _metric_row(row, "Method", _loss_method(row))
        entry["Dataset"] = _clean_text(row.get("dataset")) or "unknown"
        rows.append(entry)
    numeric = _mean_metric_table(rows, "Method", TABLE7_NUMERIC_COLUMNS, MEAN_TABLE_METRICS)
    if not numeric.empty:
        numeric = numeric.sort_values("Method").reset_index(drop=True)
    formatted = _format_mean_metric_table(numeric, "Method", MEAN_TABLE_METRICS, TABLE7_COLUMNS)
    return formatted, numeric


def _table8_ablation_group_mean(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metrics.empty:
        return pd.DataFrame(columns=TABLE8_COLUMNS), pd.DataFrame(columns=TABLE8_NUMERIC_COLUMNS)
    frame = _test_rows(_preferred_metrics_frame(metrics))
    rows: List[Dict[str, Any]] = []
    for _, row_series in frame.iterrows():
        row = row_series.to_dict()
        if str(row.get("phase") or "").lower() != "student":
            continue
        raw_method = _row_prune_method(row)
        if not raw_method or _row_loss_scope(row) not in {"", PGD_LOSS_TAG}:
            continue
        component = _method_group_component(raw_method)
        if component == "Other":
            continue
        entry = _metric_row(row, "Component", component)
        entry["Dataset"] = _clean_text(row.get("dataset")) or "unknown"
        rows.append(entry)
    order = {"Blueprint": 0, "Middle Conv2": 1, "Full Block": 2}
    numeric = _mean_metric_table(rows, "Component", TABLE8_NUMERIC_COLUMNS, MEAN_TABLE_METRICS)
    numeric = _sort_by_known_order(numeric, "Component", order)
    formatted = _format_mean_metric_table(numeric, "Component", MEAN_TABLE_METRICS, TABLE8_COLUMNS)
    return formatted, numeric


def _table9_metric_row(method: str, row: Dict[str, Any] | None) -> Dict[str, Any]:
    if row is None:
        return {metric: np.nan for metric in TABLE9_METRICS} | {"Method": method}
    return {
        "Method": method,
        "Dice": _row_number(row, "dice", "Dice", "val_dice", "val_macro_dice", "test_dice"),
        "IoU": _row_number(row, "iou", "IoU", "val_iou", "test_iou"),
        "HD95": _row_number(row, "hd95", "HD95", "val_hd95", "test_hd95"),
        "Params (M)": _params_to_millions(_row_number(row, "params", "Params", "parameters", "num_params", "Params (M)")),
        "FLOPs": _row_number(row, "flops", "FLOPs"),
        "FPS": _row_number(row, "fps", "FPS"),
        "Inf (s)": _row_number(row, "inference_time_seconds", "Inf (s)", "Inference Time (s)"),
        "Search Time (s)": _row_number(row, "search_time_seconds", "Search Time (s)"),
    }


def _table9_method_mean(outputs_root: Path, datasets: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    method_order: Dict[str, int] = {}
    for dataset in datasets:
        base_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR
        method_dirs = _discover_method_dirs(outputs_root, dataset, "3_student", TABLE6_METHOD_DIRS, include_teacher=True)
        for method, relative_dir in method_dirs:
            method_order.setdefault(method, len(method_order))
            metrics_path = base_root / relative_dir / "metrics_summary.csv"
            metric_row = _best_test_row_from_csv(metrics_path)
            if metric_row is None:
                continue
            entry = _table9_metric_row(method, metric_row)
            entry["Dataset"] = dataset
            rows.append(entry)
    numeric = _mean_metric_table(rows, "Method", TABLE9_NUMERIC_COLUMNS, TABLE9_METRICS)
    numeric = _sort_by_known_order(numeric, "Method", method_order)
    formatted = _format_mean_metric_table(numeric, "Method", TABLE9_METRICS, TABLE9_COLUMNS)
    return formatted, numeric


def _mean_from_dataset_tables(
    tables: Iterable[pd.DataFrame],
    metrics: List[str],
    columns: List[str],
    numeric_columns: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    method_order: Dict[str, int] = {}
    for frame in tables:
        if frame.empty or "Method" not in frame.columns:
            continue
        for _, row_series in frame.iterrows():
            source = row_series.to_dict()
            method = _clean_text(source.get("Method"))
            if not method:
                continue
            entry: Dict[str, Any] = {
                "Method": method,
                "Dataset": _clean_text(source.get("Dataset")) or "unknown",
            }
            has_value = False
            for metric in metrics:
                value = _safe_float(source.get(metric))
                entry[metric] = value
                has_value = has_value or not np.isnan(value)
            if not has_value:
                continue
            method_order.setdefault(method, len(method_order))
            rows.append(entry)
    numeric = _mean_metric_table(rows, "Method", numeric_columns, metrics)
    numeric = _sort_by_known_order(numeric, "Method", method_order)
    formatted = _format_mean_metric_table(numeric, "Method", metrics, columns)
    return formatted, numeric


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
        pd.DataFrame(columns=TABLE7_COLUMNS).to_csv(save_root / "table7_loss_mean.csv", index=False)
        pd.DataFrame(columns=TABLE7_NUMERIC_COLUMNS).to_csv(save_root / "table7_loss_mean_numeric.csv", index=False)
        pd.DataFrame(columns=TABLE8_COLUMNS).to_csv(save_root / "table8_ablation_mean.csv", index=False)
        pd.DataFrame(columns=TABLE8_NUMERIC_COLUMNS).to_csv(save_root / "table8_ablation_mean_numeric.csv", index=False)
        pd.DataFrame(columns=TABLE9_COLUMNS).to_csv(save_root / "table9_method_mean.csv", index=False)
        pd.DataFrame(columns=TABLE9_NUMERIC_COLUMNS).to_csv(save_root / "table9_method_mean_numeric.csv", index=False)
        pd.DataFrame(columns=TABLE10_COLUMNS).to_csv(save_root / "table10_pruning_mean.csv", index=False)
        pd.DataFrame(columns=TABLE10_NUMERIC_COLUMNS).to_csv(save_root / "table10_pruning_mean_numeric.csv", index=False)
        pd.DataFrame(columns=TABLE11_COLUMNS).to_csv(save_root / "table11_computational_cost_mean.csv", index=False)
        pd.DataFrame(columns=TABLE11_NUMERIC_COLUMNS).to_csv(save_root / "table11_computational_cost_mean_numeric.csv", index=False)
        return 0

    raw_datasets = set(metrics.get("dataset", pd.Series(dtype=str)).dropna().astype(str)) | set(timing.get("dataset", pd.Series(dtype=str)).dropna().astype(str))
    datasets = sorted(dataset for dataset in raw_datasets if dataset != "unknown" and not Path(dataset).suffix)
    all_performance_tables: List[pd.DataFrame] = []
    table1_tables: List[pd.DataFrame] = []
    table2_tables: List[pd.DataFrame] = []
    table5_tables: List[pd.DataFrame] = []
    for dataset in datasets:
        dataset_dir = save_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Processing tables for dataset: %s -> %s", dataset, dataset_dir)
        tables = _tables_for_dataset(dataset, metrics, timing)
        tables["table2_pruning.csv"] = _table2_pruning_method_comparison(outputs_root, dataset)
        tables["table5_computational_cost.csv"] = _table5_timing_method_comparison(outputs_root, dataset)
        tables["table6_method_comparison.csv"] = _table6_method_comparison(outputs_root, dataset)
        for filename, table in tables.items():
            output_path = dataset_dir / filename
            logging.info("Processing table: %s rows=%d -> %s", filename, len(table), output_path)
            table.to_csv(output_path, index=False)
            if filename == "table1_baseline.csv" and not table.empty:
                table_for_mean = table.copy()
                table_for_mean.insert(0, "Dataset", dataset)
                table1_tables.append(table_for_mean)
            if filename == "table2_pruning.csv" and not table.empty:
                table_for_mean = table.copy()
                table_for_mean.insert(0, "Dataset", dataset)
                table2_tables.append(table_for_mean)
            if filename == "table5_computational_cost.csv" and not table.empty:
                table_for_mean = table.copy()
                table_for_mean.insert(0, "Dataset", dataset)
                table5_tables.append(table_for_mean)
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

    table7, table7_numeric = _table7_loss_mean(metrics)
    table7_path = save_root / "table7_loss_mean.csv"
    table7_numeric_path = save_root / "table7_loss_mean_numeric.csv"
    logging.info("Processing table: table7_loss_mean.csv rows=%d -> %s", len(table7), table7_path)
    table7.to_csv(table7_path, index=False)
    table7_numeric.to_csv(table7_numeric_path, index=False)
    logging.info("Saved table: %s", table7_path)
    logging.info("Saved table: %s", table7_numeric_path)

    table8, table8_numeric = _table8_ablation_group_mean(metrics)
    table8_path = save_root / "table8_ablation_mean.csv"
    table8_numeric_path = save_root / "table8_ablation_mean_numeric.csv"
    logging.info("Processing table: table8_ablation_mean.csv rows=%d -> %s", len(table8), table8_path)
    table8.to_csv(table8_path, index=False)
    table8_numeric.to_csv(table8_numeric_path, index=False)
    logging.info("Saved table: %s", table8_path)
    logging.info("Saved table: %s", table8_numeric_path)

    table9, table9_numeric = _table9_method_mean(outputs_root, datasets)
    table9_path = save_root / "table9_method_mean.csv"
    table9_numeric_path = save_root / "table9_method_mean_numeric.csv"
    logging.info("Processing table: table9_method_mean.csv rows=%d -> %s", len(table9), table9_path)
    table9.to_csv(table9_path, index=False)
    table9_numeric.to_csv(table9_numeric_path, index=False)
    logging.info("Saved table: %s", table9_path)
    logging.info("Saved table: %s", table9_numeric_path)

    table10, table10_numeric = _mean_from_dataset_tables(table2_tables, TABLE10_METRICS, TABLE10_COLUMNS, TABLE10_NUMERIC_COLUMNS)
    table10_path = save_root / "table10_pruning_mean.csv"
    table10_numeric_path = save_root / "table10_pruning_mean_numeric.csv"
    logging.info("Processing table: table10_pruning_mean.csv rows=%d -> %s", len(table10), table10_path)
    table10.to_csv(table10_path, index=False)
    table10_numeric.to_csv(table10_numeric_path, index=False)
    logging.info("Saved table: %s", table10_path)
    logging.info("Saved table: %s", table10_numeric_path)

    table11, table11_numeric = _mean_from_dataset_tables(table5_tables, TABLE11_METRICS, TABLE11_COLUMNS, TABLE11_NUMERIC_COLUMNS)
    table11_path = save_root / "table11_computational_cost_mean.csv"
    table11_numeric_path = save_root / "table11_computational_cost_mean_numeric.csv"
    logging.info("Processing table: table11_computational_cost_mean.csv rows=%d -> %s", len(table11), table11_path)
    table11.to_csv(table11_path, index=False)
    table11_numeric.to_csv(table11_numeric_path, index=False)
    logging.info("Saved table: %s", table11_path)
    logging.info("Saved table: %s", table11_numeric_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
