from __future__ import annotations

import argparse
import hashlib
import logging
import re
import textwrap
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
try:
    import matplotlib.image as mpimg
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as error:  # pragma: no cover - dependency guard
    raise SystemExit("Missing dependency: pandas/matplotlib. Install project requirements with `pip install -r requirements.txt`.") from error

PGD_TEACHER_DIR = "unet_resnet152_teacher"
PGD_LOSS_TAG = "loss_seg_kd"
PGD_LOSS_TAGS = (PGD_LOSS_TAG,)
PGD_COMPARISON_LOSS_TAGS = (PGD_LOSS_TAG, "loss_seg_only")
CANONICAL_STATIC_RATIO = 0.5
CHANNEL_METHODS = {"static", "kneedle", "otsu", "gmm"}
MIDDLE_METHODS = {"middle_static", "middle_kneedle", "middle_otsu", "middle_gmm"}
FULL_METHODS = {"full_static", "full_kneedle", "full_otsu", "full_gmm"}
LEGACY_METHOD_COLUMN_KEY = bytes.fromhex("7068756f6e672070686170").decode("ascii")
LEGACY_TEACHER_LABEL_KEY = bytes.fromhex("6769616f207669656e").decode("ascii")
FIGURE15_DATASET = "cvc_clinicdb"
FIGURE15_FALLBACK_METHOD_DIRS = [
    ("Teacher", Path("1_teacher")),
    ("Static r=0.5", Path(PGD_LOSS_TAG) / "output_s1_static_0.5_no" / "3_student"),
    ("Kneedle", Path(PGD_LOSS_TAG) / "output_s2_kneedle_auto_no" / "3_student"),
    ("Otsu", Path(PGD_LOSS_TAG) / "output_s3_otsu_auto_no" / "3_student"),
    ("GMM", Path(PGD_LOSS_TAG) / "output_s4_gmm_auto_no" / "3_student"),
    ("Middle Static", Path(PGD_LOSS_TAG) / "output_s5_middle_static_0.5_no" / "3_student"),
    ("Middle Kneedle", Path(PGD_LOSS_TAG) / "output_s6_middle_kneedle_auto_no" / "3_student"),
    ("Middle Otsu", Path(PGD_LOSS_TAG) / "output_s7_middle_otsu_auto_no" / "3_student"),
    ("Middle GMM", Path(PGD_LOSS_TAG) / "output_s8_middle_gmm_auto_no" / "3_student"),
    ("Full Static", Path(PGD_LOSS_TAG) / "output_s9_full_static_0.5_no" / "3_student"),
    ("Full Kneedle", Path(PGD_LOSS_TAG) / "output_s10_full_kneedle_auto_no" / "3_student"),
    ("Full Otsu", Path(PGD_LOSS_TAG) / "output_s11_full_otsu_auto_no" / "3_student"),
    ("Full GMM", Path(PGD_LOSS_TAG) / "output_s12_full_gmm_auto_no" / "3_student"),
]


def _safe_float(value, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _fmt_ratio(value) -> str:
    number = _safe_float(value)
    if np.isnan(number):
        return "auto"
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _display_method(raw_method: str, ratio=np.nan) -> str:
    raw_method = str(raw_method or "").lower()
    mapping = {
        "kneedle": "Kneedle",
        "otsu": "Otsu",
        "gmm": "GMM",
        "middle_kneedle": "Middle-Kneedle",
        "middle_otsu": "Middle-Otsu",
        "middle_gmm": "Middle-GMM",
        "full_kneedle": "Full-Kneedle Block",
        "full_otsu": "Full-Otsu Block",
        "full_gmm": "Full-GMM Block",
    }
    if raw_method == "static":
        return f"Static r={_fmt_ratio(ratio)}"
    if raw_method == "middle_static":
        return f"Middle Static Pruning (r = {_fmt_ratio(ratio)})"
    if raw_method == "full_static":
        return f"Full-Static Block (r = {_fmt_ratio(ratio)})"
    return mapping.get(raw_method, raw_method.replace("_", " ").title())


def _method_slug(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(label).lower()).strip("_")
    return slug or "method"


def _wrap_labels(labels: Sequence[str], width: int = 14) -> List[str]:
    return ["\n".join(textwrap.wrap(str(label), width=width, break_long_words=False)) for label in labels]


def _smooth(values: pd.Series, window: int = 3) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rolling(window=window, min_periods=1, center=True).mean()


def _save_pdf(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved figure: %s", path)


def _save_pdf_multi(fig, paths: Sequence[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        logging.info("Saved figure: %s", path)
    plt.close(fig)


def _placeholder(path: Path, message: str) -> None:
    logging.warning("%s", message)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save_pdf(fig, path)


def _run_figure(name: str, output_hint: Path, func, *args, **kwargs) -> None:
    logging.info("Processing %s -> %s", name, output_hint)
    func(*args, **kwargs)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        frame = _normalize_table_labels(pd.read_csv(path))
        if not frame.empty:
            frame = frame.copy()
            frame["csv_source_path"] = str(path)
            frame["csv_source_file"] = path.name
            if "source_path" not in frame.columns:
                frame["source_path"] = str(path)
            if "source_file" not in frame.columns:
                frame["source_file"] = path.name
        return frame
    except Exception as error:
        logging.warning("Cannot read CSV %s | %s", path, error)
        return pd.DataFrame()


def _best_test_row(path: Path) -> pd.Series | None:
    frame = _read_csv(path)
    if frame.empty:
        logging.warning("Missing or empty Figure 15 metrics CSV: %s", path)
        return None
    if "split" in frame.columns:
        test_frame = frame[frame["split"].fillna("").astype(str).str.lower().eq("test")]
        if not test_frame.empty:
            frame = test_frame
    dice_col = _first_existing_column(frame, ("dice", "Dice", "test_dice", "val_dice", "val_macro_dice"))
    if dice_col is None:
        logging.warning("Figure 15 metrics CSV has no dice column: %s", path)
        return None
    scores = pd.to_numeric(frame[dice_col], errors="coerce").fillna(-np.inf)
    if scores.empty or float(scores.max()) == -np.inf:
        return None
    return frame.loc[scores.idxmax()]


def _params_to_millions(value) -> float:
    number = _safe_float(value)
    if np.isnan(number):
        return number
    return number / 1e6 if abs(number) > 1e5 else number


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    lower_lookup = {str(column).lower(): column for column in frame.columns}
    for column in candidates:
        match = lower_lookup.get(str(column).lower())
        if match is not None:
            return match
    return None


def _row_value(row, candidates: Sequence[str], default=np.nan):
    for key in candidates:
        try:
            value = row.get(key)
        except AttributeError:
            value = None
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
    return default


def _params_from_row(row) -> float:
    value = _row_value(row, ("params", "Params", "parameters", "num_params", "Params (M)"))
    if np.isnan(value):
        return value
    if "Params (M)" in getattr(row, "index", []):
        params_m = _safe_float(row.get("Params (M)"))
        if not np.isnan(params_m):
            return params_m
    return _params_to_millions(value)


def _params_from_metric_params(row) -> float:
    value = _row_value(row, ("params",))
    if np.isnan(value):
        return _params_from_row(row)
    return _params_to_millions(value)


def _dice_from_row(row) -> float:
    return _row_value(row, ("dice", "Dice", "val_dice", "val_macro_dice", "test_dice"))


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


def _figure15_method_dirs(outputs_root: Path, dataset: str) -> List[tuple[str, Path]]:
    base_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR
    loss_root = base_root / PGD_LOSS_TAG
    if not loss_root.is_dir():
        return FIGURE15_FALLBACK_METHOD_DIRS

    discovered = []
    for output_dir in loss_root.iterdir():
        if not output_dir.is_dir() or not output_dir.name.startswith("output_"):
            continue
        raw_method, ratio = _method_from_output_dir(output_dir)
        discovered.append(
            (
                _method_sort_key(raw_method, ratio),
                _display_method(raw_method, ratio),
                Path(PGD_LOSS_TAG) / output_dir.name / "3_student",
            )
        )
    if not discovered:
        return FIGURE15_FALLBACK_METHOD_DIRS

    discovered = sorted(discovered, key=lambda item: item[0])
    method_dirs = [("Teacher", Path("1_teacher"))]
    method_dirs.extend((label, relative_dir) for _, label, relative_dir in discovered)
    return method_dirs


def _method_group_label(raw_method: str) -> str:
    method = str(raw_method or "").lower()
    if method in CHANNEL_METHODS:
        return "Blueprint"
    if method in MIDDLE_METHODS:
        return "Middle Conv2"
    if method in FULL_METHODS:
        return "Full Block"
    if method == "teacher":
        return "Teacher"
    return "Other"


def _is_static_method(raw_method: str) -> bool:
    return str(raw_method or "").lower() in {"static", "middle_static", "full_static"}


def _ratio_matches(value: float, target: float) -> bool:
    number = _safe_float(value)
    return not np.isnan(number) and abs(number - float(target)) < 1e-9


def _global_metric_rows(
    outputs_root: Path,
    save_root: Path,
    *,
    include_teacher: bool = False,
    canonical_static_ratio: float | None = None,
    static_only: bool = False,
) -> pd.DataFrame:
    rows = []
    for dataset in _datasets(outputs_root, save_root):
        base_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR
        if include_teacher and not static_only:
            teacher_row = _best_test_row(base_root / "1_teacher" / "metrics_summary.csv")
            if teacher_row is not None:
                rows.append(
                    {
                        "Dataset": dataset,
                        "Method": "Teacher",
                        "Group": "Teacher",
                        "Raw Method": "teacher",
                        "Static Ratio": np.nan,
                        "Params (M)": _params_from_row(teacher_row),
                        "Dice": _dice_from_row(teacher_row),
                        "Sort": -1.0,
                    }
                )

        loss_root = base_root / PGD_LOSS_TAG
        if not loss_root.is_dir():
            continue
        for output_dir in loss_root.iterdir():
            if not output_dir.is_dir() or not output_dir.name.startswith("output_"):
                continue
            raw_method, ratio = _method_from_output_dir(output_dir)
            is_static = _is_static_method(raw_method)
            if static_only and not is_static:
                continue
            if canonical_static_ratio is not None and is_static and not _ratio_matches(ratio, canonical_static_ratio):
                continue
            metrics_path = output_dir / "3_student" / "metrics_summary.csv"
            row = _best_test_row(metrics_path)
            if row is None:
                continue
            rows.append(
                {
                    "Dataset": dataset,
                    "Method": _display_method(raw_method, ratio),
                    "Group": _method_group_label(raw_method),
                    "Raw Method": raw_method,
                    "Static Ratio": ratio,
                    "Params (M)": _params_from_row(row),
                    "Dice": _dice_from_row(row),
                    "Sort": float(_method_sort_key(raw_method, ratio)[0]) + (0.001 * (ratio if not np.isnan(_safe_float(ratio)) else 999.0)),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["Params (M)"] = pd.to_numeric(frame["Params (M)"], errors="coerce")
    frame["Dice"] = pd.to_numeric(frame["Dice"], errors="coerce")
    return frame.dropna(subset=["Params (M)", "Dice"]).reset_index(drop=True)


def _mean_metric_summary(frame: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    dataset_level = (
        frame.groupby(["Dataset", *group_cols], dropna=False)
        .agg(
            **{
                "Mean Dice": ("Dice", "mean"),
                "Mean Params (M)": ("Params (M)", "mean"),
                "Sort": ("Sort", "min"),
            }
        )
        .reset_index()
    )
    summary = (
        dataset_level.groupby(list(group_cols), dropna=False)
        .agg(
            **{
                "Mean Dice": ("Mean Dice", "mean"),
                "Std Dice": ("Mean Dice", "std"),
                "Mean Params (M)": ("Mean Params (M)", "mean"),
                "Std Params (M)": ("Mean Params (M)", "std"),
                "Datasets": ("Dataset", "nunique"),
                "Sort": ("Sort", "min"),
            }
        )
        .reset_index()
    )
    return summary.fillna({"Std Dice": 0.0, "Std Params (M)": 0.0}).sort_values("Sort").reset_index(drop=True)


def _teacher_ratio_zero_points(outputs_root: Path, save_root: Path, group_order: Dict[str, int]) -> pd.DataFrame:
    rows = []
    for dataset in _datasets(outputs_root, save_root):
        teacher_row = _best_test_row(outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR / "1_teacher" / "metrics_summary.csv")
        if teacher_row is None:
            continue
        rows.append(
            {
                "Dataset": dataset,
                "Dice": _dice_from_row(teacher_row),
                "Params (M)": _params_from_row(teacher_row),
            }
        )
    teacher_frame = pd.DataFrame(rows)
    if teacher_frame.empty:
        return pd.DataFrame()
    teacher_frame["Dice"] = pd.to_numeric(teacher_frame["Dice"], errors="coerce")
    teacher_frame["Params (M)"] = pd.to_numeric(teacher_frame["Params (M)"], errors="coerce")
    teacher_frame = teacher_frame.dropna(subset=["Dice"])
    if teacher_frame.empty:
        return pd.DataFrame()

    dice_values = teacher_frame["Dice"].dropna()
    param_values = teacher_frame["Params (M)"].dropna()
    teacher_summary = {
        "Static Ratio": 0.0,
        "Mean Dice": float(dice_values.mean()),
        "Std Dice": float(dice_values.std()) if len(dice_values) > 1 else 0.0,
        "Mean Params (M)": float(param_values.mean()) if not param_values.empty else float("nan"),
        "Std Params (M)": float(param_values.std()) if len(param_values) > 1 else 0.0 if len(param_values) == 1 else float("nan"),
        "Datasets": int(teacher_frame["Dataset"].nunique()),
    }
    return pd.DataFrame(
        [
            {
                "Group": group,
                "Sort": float(order) - 0.1,
                **teacher_summary,
            }
            for group, order in group_order.items()
        ]
    )


def _plot_params_dice_summary(frame: pd.DataFrame, path: Path, *, label_column: str = "Method") -> None:
    if frame.empty:
        _placeholder(path, "No valid mean Params/Dice rows found.")
        return
    x = np.arange(len(frame))
    labels = frame[label_column].astype(str).tolist()
    fig_width = max(8.5, min(18.0, 0.72 * len(labels) + 4.0))
    fig, ax_params = plt.subplots(figsize=(fig_width, 4.9))
    ax_dice = ax_params.twinx()
    params_bar = ax_params.bar(x, frame["Mean Params (M)"], width=0.58, color="#4c78a8", alpha=0.72, label="Mean Params (M)")
    dice_line = ax_dice.plot(x, frame["Mean Dice"], marker="s", linewidth=1.9, color="#f58518", label="Mean Dice")
    ax_params.set_xlabel("Method")
    ax_params.set_ylabel("Mean Params (M)")
    ax_dice.set_ylabel("Mean Dice")
    ax_params.set_xticks(x)
    ax_params.set_xticklabels(_wrap_labels(labels, width=13), rotation=25, ha="right", fontsize=8)
    ax_params.grid(alpha=0.25, axis="y")
    lines = [params_bar, *dice_line]
    ax_params.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8, frameon=True, framealpha=0.9)
    _save_pdf(fig, path)


def _decode_mojibake(value: str) -> str:
    text = str(value)
    for _ in range(2):
        decoded = None
        for encoding in ("latin-1", "cp1252"):
            try:
                decoded = text.encode(encoding).decode("utf-8")
                break
            except UnicodeError:
                continue
        if decoded is None or decoded == text:
            break
        text = decoded
    return text


def _ascii_key(value) -> str:
    text = _decode_mojibake(str(value or ""))
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii").lower().strip()


def _is_method_column(column) -> bool:
    return _ascii_key(column) in {"method", LEGACY_METHOD_COLUMN_KEY}


def _normalize_method_label(value):
    try:
        if pd.isna(value):
            return value
    except (TypeError, ValueError):
        pass
    text = _decode_mojibake(str(value)).strip()
    key = _ascii_key(text)
    if LEGACY_TEACHER_LABEL_KEY in key:
        return "Teacher (UNet-ResNet152)" if "unet-resnet152" in key else "Teacher"
    return text


def _normalize_table_labels(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    rename_map = {column: "Method" for column in frame.columns if column != "Method" and _is_method_column(column)}
    if rename_map:
        frame = frame.rename(columns=rename_map)
    if "Method" in frame.columns:
        frame["Method"] = frame["Method"].map(_normalize_method_label)
    return frame


def _method_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if _is_method_column(column):
            return column
    return str(frame.columns[0]) if len(frame.columns) else None


def _dataset_from_path(path: Path, outputs_root: Path) -> str:
    try:
        parts = path.relative_to(outputs_root).parts
    except ValueError:
        parts = path.parts
    if len(parts) >= 3 and parts[0] == "pgd_unet":
        return parts[1]
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def _pgd_focus_root(outputs_root: Path, dataset: str) -> Path:
    return outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR / PGD_LOSS_TAG


def _pgd_focus_roots(outputs_root: Path, dataset: str) -> List[Path]:
    base_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR
    focus_root = base_root / PGD_LOSS_TAG
    return [focus_root] if focus_root.exists() else []


def _pgd_teacher_phase_root(outputs_root: Path, dataset: str) -> Path:
    return outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR / "1_teacher"


def _is_pgd_focus_path(path: Path, outputs_root: Path, dataset: str) -> bool:
    try:
        relative = path.relative_to(_pgd_focus_root(outputs_root, dataset))
    except ValueError:
        return False
    return bool(relative.parts)


def _is_non_default_loss_path(path: Path, outputs_root: Path) -> bool:
    try:
        parts = path.relative_to(outputs_root).parts
    except ValueError:
        parts = path.parts
    return (
        len(parts) >= 4
        and parts[0] == "pgd_unet"
        and parts[2] == PGD_TEACHER_DIR
        and str(parts[3]).startswith("loss_")
        and parts[3] != PGD_LOSS_TAG
    )


def _datasets(outputs_root: Path, save_root: Path) -> List[str]:
    names = set()
    if save_root.exists():
        for path in save_root.iterdir():
            if path.is_dir() and path.name != "paper_figures" and not Path(path.name).suffix:
                names.add(path.name)
    if outputs_root.exists():
        pgd_root = outputs_root / "pgd_unet"
        if pgd_root.exists():
            for dataset_dir in pgd_root.iterdir():
                if _pgd_focus_roots(outputs_root, dataset_dir.name):
                    names.add(dataset_dir.name)
        for csv_path in outputs_root.rglob("*.csv"):
            if _is_non_default_loss_path(csv_path, outputs_root):
                continue
            dataset = _dataset_from_path(csv_path, outputs_root)
            if dataset != "unknown" and not Path(dataset).suffix:
                names.add(dataset)
    return sorted(name for name in names if name and name != "unknown")


def _find_dataset_files(outputs_root: Path, dataset: str, filename: str) -> List[Path]:
    if not outputs_root.exists():
        return []
    for focus_root in _pgd_focus_roots(outputs_root, dataset):
        focused = [path for path in focus_root.rglob(filename)]
        if focused:
            logging.info("Found %d target PGD files for %s/%s under %s", len(focused), dataset, filename, focus_root)
            return focused
    fallback = [
        path
        for path in outputs_root.rglob(filename)
        if dataset in path.parts and not Path(dataset).suffix and not _is_non_default_loss_path(path, outputs_root)
    ]
    if fallback:
        logging.warning("Using fallback files for %s/%s because target PGD path has no matches.", dataset, filename)
    return fallback


def _find_figure23_files(outputs_root: Path, dataset: str, filename: str) -> List[Path]:
    preferred_root = _pgd_focus_root(outputs_root, dataset)
    if preferred_root.exists():
        preferred = [path for path in preferred_root.rglob(filename)]
        if preferred:
            logging.info("Found %d Figure 23 files for %s/%s under %s", len(preferred), dataset, filename, preferred_root)
            return preferred
    logging.warning("No Figure 23 files for %s/%s under exact KD dataset root: %s", dataset, filename, preferred_root)
    return []


def _find_teacher_channel_files(outputs_root: Path, dataset: str, filenames: Sequence[str]) -> List[Path]:
    teacher_root = _pgd_teacher_phase_root(outputs_root, dataset)
    if not teacher_root.exists():
        logging.warning("Teacher phase root is missing for %s: %s", dataset, teacher_root)
        return []
    channel_root = teacher_root / "artifacts" / "channel_analysis"
    search_root = channel_root if channel_root.exists() else teacher_root
    paths: List[Path] = []
    for filename in filenames:
        matches = list(search_root.rglob(filename))
        if matches:
            logging.info("Found %d teacher channel files for %s/%s under %s", len(matches), dataset, filename, search_root)
            paths.extend(matches)
    if not paths:
        logging.warning("No teacher channel files found for %s under %s", dataset, search_root)
    return paths


def _concat_csvs(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [_read_csv(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def figure1(save_root: Path) -> None:
    path = save_root / "paper_figures" / "figure1_overall_framework.pdf"
    fig, ax = plt.subplots(figsize=(12, 2.4))
    ax.axis("off")
    labels = ["Teacher", "Importance", "Pruning", "Student", "Distillation", "Evaluation"]
    x_positions = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label) in enumerate(zip(x_positions, labels)):
        box = patches.FancyBboxPatch((x - 0.065, 0.42), 0.13, 0.22, boxstyle="round,pad=0.02", linewidth=1.2, edgecolor="black", facecolor="#f4f4f4")
        ax.add_patch(box)
        ax.text(x, 0.53, label, ha="center", va="center", fontsize=10)
        if index < len(labels) - 1:
            ax.annotate("", xy=(x_positions[index + 1] - 0.075, 0.53), xytext=(x + 0.075, 0.53), arrowprops={"arrowstyle": "->", "lw": 1.2})
    _save_pdf(fig, path)


def figure4(save_root: Path) -> None:
    path = save_root / "paper_figures" / "figure4_block_pruning_strategy.pdf"
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")
    enc_x, dec_x = 0.18, 0.72
    ys = [0.78, 0.62, 0.46, 0.30]
    for i, y in enumerate(ys):
        keep = i in {0, len(ys) - 1}
        color = "#d9ead3" if keep else "#f4cccc"
        ax.add_patch(patches.Rectangle((enc_x, y), 0.16, 0.09, facecolor=color, edgecolor="black"))
        ax.add_patch(patches.Rectangle((dec_x, y), 0.16, 0.09, facecolor=color, edgecolor="black"))
        ax.text(enc_x + 0.08, y + 0.045, "Keep" if keep else "Prune", ha="center", va="center", fontsize=9)
        ax.text(dec_x + 0.08, y + 0.045, "Keep" if keep else "Prune", ha="center", va="center", fontsize=9)
        ax.plot([enc_x + 0.16, dec_x], [y + 0.045, y + 0.045], color="gray", linestyle="--", linewidth=1)
    ax.add_patch(patches.Rectangle((0.44, 0.13), 0.16, 0.09, facecolor="#f4cccc", edgecolor="black"))
    ax.text(0.52, 0.175, "Prune", ha="center", va="center", fontsize=9)
    ax.text(enc_x + 0.08, 0.9, "Encoder", ha="center", va="center", fontsize=10)
    ax.text(dec_x + 0.08, 0.9, "Decoder", ha="center", va="center", fontsize=10)
    ax.text(0.52, 0.27, "Bottleneck", ha="center", va="center", fontsize=10)
    _save_pdf(fig, path)


def figure2_importance(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    path = dataset_dir / "figure2_importance_distribution.pdf"
    detail_files = _find_dataset_files(outputs_root, dataset, "channel_level_detail.csv")
    frame = _concat_csvs(detail_files)
    if frame.empty:
        frame = _concat_csvs(_find_dataset_files(outputs_root, dataset, "channel_importance.csv"))
    if frame.empty or "importance" not in frame.columns:
        _placeholder(path, f"No channel importance rows found for {dataset}.")
        return
    layers = list(dict.fromkeys(frame["layer_name"].astype(str)))[:2]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for layer in layers:
        values = pd.to_numeric(frame[frame["layer_name"].astype(str).eq(layer)]["importance"], errors="coerce").dropna()
        if not values.empty:
            ax.hist(values, bins=30, alpha=0.55, label=f"Layer {layers.index(layer) + 1}")
    ax.set_xlabel("Channel importance")
    ax.set_ylabel("Number of channels")
    ax.legend()
    _save_pdf(fig, path)


def figure3_thresholds(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    path = dataset_dir / "figure3_thresholding_methods.pdf"
    details = _concat_csvs(_find_dataset_files(outputs_root, dataset, "channel_level_detail.csv"))
    summaries = _concat_csvs(_find_dataset_files(outputs_root, dataset, "pruning_summary.csv"))
    if details.empty or "importance" not in details.columns:
        _placeholder(path, f"No pruning importance rows found for {dataset}.")
        return
    layer = ""
    if not summaries.empty and "layer_name" in summaries.columns:
        summary_layers = summaries["layer_name"].dropna().astype(str)
        detail_layers = set(details.get("layer_name", pd.Series(dtype=str)).dropna().astype(str))
        common_layers = [name for name in summary_layers if name in detail_layers]
        if common_layers:
            layer = pd.Series(common_layers).mode().iloc[0]
    if not layer:
        layer = str(details["layer_name"].dropna().astype(str).iloc[0])
    values = pd.to_numeric(details[details["layer_name"].astype(str).eq(layer)]["importance"], errors="coerce").dropna()
    if values.empty:
        _placeholder(path, f"No channel importance values found for representative layer in {dataset}.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(values, bins=28, alpha=0.72, color="#9ecae1", edgecolor="white", linewidth=0.4)
    x_min = float(values.quantile(0.005))
    x_max = float(values.quantile(0.995))
    if x_max <= x_min:
        x_min, x_max = float(values.min()), float(values.max())
    margin = max((x_max - x_min) * 0.08, 1e-6)
    x_low, x_high = x_min - margin, x_max + margin
    if not summaries.empty and "pruning_threshold" in summaries.columns:
        layer_summaries = summaries.copy()
        if "layer_name" in layer_summaries.columns:
            layer_summaries = layer_summaries[layer_summaries["layer_name"].astype(str).eq(layer)]
        layer_summaries = _prefer_static_display_ratio(layer_summaries)
        method_series = layer_summaries.get("prune_method", pd.Series(["Method"] * len(layer_summaries))).astype(str).str.lower()
        threshold_styles = {
            "static": {"color": "#d62728", "linestyle": "--"},
            "kneedle": {"color": "#2ca02c", "linestyle": "-."},
            "otsu": {"color": "#ff7f0e", "linestyle": ":"},
            "gmm": {"color": "#9467bd", "linestyle": (0, (5, 1))},
        }
        for method in ("static", "kneedle", "otsu", "gmm"):
            group = layer_summaries[method_series.eq(method)]
            threshold = pd.to_numeric(group["pruning_threshold"], errors="coerce").dropna()
            if not threshold.empty:
                value = float(threshold.median())
                if x_low <= value <= x_high:
                    ax.axvline(value, linewidth=1.7, label=_threshold_method_label(method, group), **threshold_styles[method])
                else:
                    logging.info("Skip out-of-range threshold line for %s/%s: %.6f", dataset, method, value)
    ax.set_xlim(x_low, x_high)
    ax.set_xlabel("Channel importance")
    ax.set_ylabel("Number of channels")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
    _save_pdf(fig, path)
    _save_group_threshold_figures(details, summaries, dataset, dataset_dir)


def _method_family(method: str) -> str:
    method = str(method or "").lower()
    for prefix in ("middle_", "full_"):
        if method.startswith(prefix):
            return method[len(prefix) :]
    return method


def _static_ratio_from_output_folder(row) -> float:
    for column in ("csv_source_path", "source_path", "csv_source_file", "source_file"):
        try:
            value = row.get(column)
        except AttributeError:
            value = None
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        for part in Path(str(value)).parts:
            if not part.startswith("output_"):
                continue
            method, ratio = _method_from_output_dir(Path(part))
            if _method_family(method) == "static" and not np.isnan(_safe_float(ratio)):
                return float(ratio)
    return float("nan")


def _configured_static_ratio_from_row(row) -> float:
    folder_ratio = _static_ratio_from_output_folder(row)
    if not np.isnan(folder_ratio):
        return folder_ratio
    for column in ("requested_static_prune_ratio", "static_prune_ratio"):
        value = _row_value(row, (column,))
        if not np.isnan(value):
            return float(value)
    return float("nan")


def _representative_prune_ratio(frame: pd.DataFrame) -> float:
    values = []
    for _, row_series in frame.iterrows():
        ratio = _configured_static_ratio_from_row(row_series)
        if not np.isnan(ratio):
            values.append(ratio)
    return float(np.median(values)) if values else float("nan")


def _prefer_static_display_ratio(frame: pd.DataFrame, target_ratio: float = CANONICAL_STATIC_RATIO) -> pd.DataFrame:
    if frame.empty or "prune_method" not in frame.columns:
        return frame.copy()
    method_series = frame["prune_method"].fillna("").astype(str).str.lower()
    static_mask = method_series.map(_method_family).eq("static")
    if not static_mask.any():
        return frame.copy()
    static_frame = frame[static_mask].copy()
    ratios = static_frame.apply(_configured_static_ratio_from_row, axis=1)
    target_frame = static_frame[(pd.to_numeric(ratios, errors="coerce") - float(target_ratio)).abs() < 1e-9]
    if target_frame.empty:
        return frame.copy()
    return pd.concat([frame[~static_mask], target_frame], axis=0).sort_index()


def _threshold_method_label(method: str, frame: pd.DataFrame) -> str:
    method = str(method or "").lower()
    if _method_family(method) != "static":
        return _display_method(method)
    ratio = _representative_prune_ratio(frame)
    label = _display_method(method, ratio)
    if np.isnan(_safe_float(ratio)):
        return label.replace(" r=auto", "").replace(" (r = auto)", "")
    return label


def _save_group_threshold_figures(details: pd.DataFrame, summaries: pd.DataFrame, dataset: str, dataset_dir: Path) -> None:
    group_specs = [
        (
            "Blueprint Stage",
            CHANNEL_METHODS,
            "stage_output",
            dataset_dir / "figure3_blueprint_importance_thresholds.pdf",
        ),
        (
            "Middle Conv2 to Conv3",
            MIDDLE_METHODS,
            "conv2_output_to_conv3_input",
            dataset_dir / "figure3_middle_conv2_importance_thresholds.pdf",
        ),
        (
            "Full Block Conv3 Output",
            FULL_METHODS,
            "conv3_output",
            dataset_dir / "figure3_full_block_output_importance_thresholds.pdf",
        ),
        (
            "Full Block Conv1 Output",
            FULL_METHODS,
            "conv1_output",
            dataset_dir / "figure3_full_block_conv1_importance_thresholds.pdf",
        ),
        (
            "Full Block Conv2 Output",
            FULL_METHODS,
            "conv2_output",
            dataset_dir / "figure3_full_block_conv2_importance_thresholds.pdf",
        ),
    ]
    method_series_details = details.get("prune_method", pd.Series([""] * len(details), index=details.index)).astype(str).str.lower()
    method_series_summaries = summaries.get("prune_method", pd.Series([""] * len(summaries), index=summaries.index)).astype(str).str.lower()
    threshold_styles = {
        "static": {"color": "#d62728", "linestyle": "--"},
        "kneedle": {"color": "#2ca02c", "linestyle": "-."},
        "otsu": {"color": "#ff7f0e", "linestyle": ":"},
        "gmm": {"color": "#9467bd", "linestyle": (0, (5, 1))},
    }
    for title, methods, channel_role, path in group_specs:
        group_details = details[method_series_details.isin(methods)].copy()
        if "channel_role" in group_details.columns:
            group_details = group_details[group_details["channel_role"].astype(str).eq(channel_role)]
        values = pd.to_numeric(group_details.get("importance", pd.Series(dtype=float)), errors="coerce").dropna()
        if values.empty:
            _placeholder(path, f"No importance rows found for {dataset} / {title}.")
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.hist(values, bins=32, alpha=0.72, color="#9ecae1", edgecolor="white", linewidth=0.4)
        x_min = float(values.quantile(0.005))
        x_max = float(values.quantile(0.995))
        if x_max <= x_min:
            x_min, x_max = float(values.min()), float(values.max())
        margin = max((x_max - x_min) * 0.08, 1e-6)
        x_low, x_high = x_min - margin, x_max + margin
        group_summaries = summaries[method_series_summaries.isin(methods)].copy()
        if "plot_role" in group_summaries.columns:
            group_summaries = group_summaries[group_summaries["plot_role"].astype(str).eq(channel_role)]
        group_summaries = _prefer_static_display_ratio(group_summaries)
        if "pruning_threshold" in group_summaries.columns:
            group_methods = group_summaries.get("prune_method", pd.Series([""] * len(group_summaries), index=group_summaries.index)).astype(str).str.lower()
            for method in sorted(set(group_methods)):
                family = _method_family(method)
                thresholds = pd.to_numeric(group_summaries[group_methods.eq(method)]["pruning_threshold"], errors="coerce").dropna()
                if thresholds.empty:
                    continue
                value = float(thresholds.median())
                if x_low <= value <= x_high:
                    method_rows = group_summaries[group_methods.eq(method)]
                    ax.axvline(value, linewidth=1.7, label=_threshold_method_label(method, method_rows), **threshold_styles.get(family, {}))
        ax.set_xlim(x_low, x_high)
        ax.set_xlabel("Channel importance")
        ax.set_ylabel("Number of channels")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
        _save_pdf(fig, path)


def _unique_source_paths(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    source_column = "csv_source_path" if "csv_source_path" in frame.columns else "source_path" if "source_path" in frame.columns else ""
    if not source_column:
        return ""
    sources = sorted({_clean_text(value) for value in frame[source_column].dropna().tolist() if _clean_text(value)})
    return " | ".join(sources)


def _threshold_demo_panels(details: pd.DataFrame, summaries: pd.DataFrame) -> tuple[str, str, List[Dict[str, object]]] | None:
    if details.empty or summaries.empty or "importance" not in details.columns or "pruning_threshold" not in summaries.columns:
        return None
    methods = [("kneedle", "Kneedle"), ("gmm", "GMM"), ("otsu", "Otsu")]
    method_series_details = details.get("prune_method", pd.Series([""] * len(details), index=details.index)).astype(str).str.lower()
    method_series_summaries = summaries.get("prune_method", pd.Series([""] * len(summaries), index=summaries.index)).astype(str).str.lower()
    role_candidates = ["stage_output", "conv2_output_to_conv3_input", "conv3_output", "conv1_output", "conv2_output", ""]
    best_result: tuple[int, str, str, List[Dict[str, object]]] | None = None

    for role in role_candidates:
        role_details = details.copy()
        role_summaries = summaries.copy()
        if role and "channel_role" in role_details.columns:
            role_details = role_details[role_details["channel_role"].astype(str).eq(role)]
        if role and "plot_role" in role_summaries.columns:
            role_summaries = role_summaries[role_summaries["plot_role"].astype(str).eq(role)]
        if role_details.empty or role_summaries.empty or "layer_name" not in role_details.columns or "layer_name" not in role_summaries.columns:
            continue

        common_layers: set[str] | None = None
        for method, _ in methods:
            detail_layers = set(role_details[method_series_details.reindex(role_details.index).eq(method)]["layer_name"].dropna().astype(str))
            summary_layers = set(role_summaries[method_series_summaries.reindex(role_summaries.index).eq(method)]["layer_name"].dropna().astype(str))
            layers = detail_layers & summary_layers
            common_layers = layers if common_layers is None else common_layers & layers
        if not common_layers:
            continue

        for layer in sorted(common_layers):
            panels: List[Dict[str, object]] = []
            valid = True
            for method, label in methods:
                detail_mask = method_series_details.reindex(role_details.index).eq(method) & role_details["layer_name"].astype(str).eq(layer)
                summary_mask = method_series_summaries.reindex(role_summaries.index).eq(method) & role_summaries["layer_name"].astype(str).eq(layer)
                detail_rows = role_details[detail_mask]
                summary_rows = role_summaries[summary_mask]
                values = pd.to_numeric(detail_rows["importance"], errors="coerce").dropna().to_numpy()
                thresholds = pd.to_numeric(summary_rows["pruning_threshold"], errors="coerce").dropna()
                if values.size == 0 or thresholds.empty:
                    valid = False
                    break
                panels.append(
                    {
                        "method": method,
                        "label": label,
                        "layer_name": layer,
                        "plot_role": role or "all",
                        "values": values,
                        "threshold": float(thresholds.median()),
                        "detail_source_paths": _unique_source_paths(detail_rows),
                        "summary_source_paths": _unique_source_paths(summary_rows),
                        "threshold_count": int(thresholds.size),
                    }
                )
            if valid:
                score = min(int(np.asarray(panel["values"]).size) for panel in panels)
                if best_result is None or score > best_result[0]:
                    best_result = (score, layer, role or "all", panels)
    if best_result is None:
        return None
    _, layer, role, panels = best_result
    return layer, role, panels


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def _normal_pdf(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    variance = max(float(variance), 1e-12)
    scale = np.sqrt(2.0 * np.pi * variance)
    return np.exp(-0.5 * ((x - float(mean)) ** 2) / variance) / scale


def _gmm_density_curves(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or np.allclose(values, values[0]):
        return None
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None

    try:
        model = GaussianMixture(n_components=2, random_state=42)
        model.fit(values.reshape(-1, 1))
        grid = np.linspace(float(values.min()), float(values.max()), 300)
        means = model.means_.reshape(-1)
        variances = model.covariances_.reshape(-1)
        weights = model.weights_.reshape(-1)
    except Exception as error:
        logging.warning("Could not fit GMM curve for Figure 23: %s", error)
        return None
    order = np.argsort(means)
    components = [weights[index] * _normal_pdf(grid, means[index], variances[index]) for index in order]
    mixture = np.sum(components, axis=0)
    return grid, components, mixture


def _gmm_threshold_from_values(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or np.allclose(values, values[0]):
        return float(values[0]) if values.size else np.nan
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return np.nan

    try:
        model = GaussianMixture(n_components=2, random_state=42)
        model.fit(values.reshape(-1, 1))
    except Exception as error:
        logging.warning("Could not recompute GMM threshold for Figure 23: %s", error)
        return np.nan

    means = model.means_.reshape(-1)
    variances = model.covariances_.reshape(-1)
    weights = model.weights_.reshape(-1)
    order = np.argsort(means)
    mu1, mu2 = float(means[order[0]]), float(means[order[1]])
    var1, var2 = max(float(variances[order[0]]), 1e-12), max(float(variances[order[1]]), 1e-12)
    w1, w2 = max(float(weights[order[0]]), 1e-12), max(float(weights[order[1]]), 1e-12)
    sigma1 = np.sqrt(var1)
    sigma2 = np.sqrt(var2)
    a = 1.0 / (2.0 * var2) - 1.0 / (2.0 * var1)
    b = mu1 / var1 - mu2 / var2
    c = (mu2**2) / (2.0 * var2) - (mu1**2) / (2.0 * var1) + np.log((w2 * sigma1) / (w1 * sigma2))
    if abs(a) < 1e-12:
        return float(0.5 * (mu1 + mu2)) if abs(b) < 1e-12 else float(-c / b)
    roots = np.roots([a, b, c])
    roots = np.real(roots[np.isreal(roots)])
    valid = roots[(roots >= mu1) & (roots <= mu2)]
    if len(valid) > 0:
        return float(valid[0])
    return float((mu1 * sigma2 + mu2 * sigma1) / (sigma1 + sigma2))


def _otsu_between_class_curve(values: np.ndarray, num_bins: int = 256):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or np.allclose(values, values[0]):
        return None
    hist, bin_edges = np.histogram(values, bins=num_bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return None
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centers)
    mu_t = mu[-1]
    numerator = (mu_t * omega - mu) ** 2
    denominator = omega * (1.0 - omega)
    denominator[denominator == 0] = 1e-12
    sigma_b2 = numerator / denominator
    return centers, _normalize01(sigma_b2)


def _otsu_threshold_from_values(values: np.ndarray, num_bins: int = 256) -> float:
    curve = _otsu_between_class_curve(values, num_bins=num_bins)
    if curve is None:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(values[0]) if values.size else np.nan
    centers, variance_curve = curve
    return float(centers[int(np.argmax(variance_curve))])


def _hist_bins(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 1
    return max(8, min(40, int(np.sqrt(values.size) * 2)))


def _values_signature(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return ""
    rounded = np.round(values, decimals=12)
    return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]


def _figure23_dataset_panels(outputs_root: Path, dataset: str) -> tuple[str, str, List[Dict[str, object]]] | None:
    details = _concat_csvs(_find_figure23_files(outputs_root, dataset, "channel_level_detail.csv"))
    summaries = _concat_csvs(_find_figure23_files(outputs_root, dataset, "pruning_summary.csv"))
    return _threshold_demo_panels(details, summaries)


def _build_figure23_threshold_figure(
    selected_dataset: str,
    selected_layer: str,
    selected_role: str,
    selected_panels: List[Dict[str, object]],
) -> tuple[plt.Figure, List[Dict[str, object]]]:
    metadata_rows: List[Dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.25), sharey=False)
    colors = {"Kneedle": "#2ca02c", "GMM": "#9467bd", "Otsu": "#ff7f0e"}
    for ax, panel in zip(np.ravel(axes), selected_panels):
        values = np.asarray(panel["values"], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        threshold = float(panel["threshold"])
        method = str(panel["method"])
        label = str(panel["label"])
        method_color = colors.get(label, "#d62728")
        closest_index = int(np.argmin(np.abs(values - threshold)))
        closest_value = float(values[closest_index])
        criterion_value = np.nan
        recomputed_threshold = np.nan
        selection_view = ""

        if method == "kneedle":
            sorted_values = np.sort(values)
            x_norm = np.linspace(0.0, 1.0, sorted_values.size)
            y_norm = _normalize01(sorted_values)
            diff = y_norm - x_norm
            knee_index = int(np.argmax(diff)) if diff.size else 0
            criterion_value = float(diff[knee_index]) if diff.size else np.nan
            recomputed_threshold = float(sorted_values[knee_index]) if sorted_values.size else np.nan
            value_span = float(values.max() - values.min())
            if value_span < 1e-12:
                threshold_norm = 0.0
            else:
                threshold_norm = float(np.clip((threshold - values.min()) / value_span, 0.0, 1.0))
            ax.plot(x_norm, y_norm, color="#4c78a8", linewidth=1.8, label="Sorted importance")
            ax.plot([0.0, 1.0], [0.0, 1.0], color="#7f7f7f", linewidth=1.1, linestyle=":", label="Reference")
            ax.plot(
                [x_norm[knee_index], x_norm[knee_index]],
                [x_norm[knee_index], y_norm[knee_index]],
                color=method_color,
                linewidth=2.0,
                label="Max gap",
            )
            ax.scatter(
                [x_norm[knee_index]],
                [threshold_norm],
                s=52,
                color=method_color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label="Threshold",
            )
            ax.set_xlabel("Kneedle: normalized sorted index")
            ax.set_ylabel("Normalized importance")
            selection_view = "Maximize y - x on normalized sorted importance"
            _legend_below_compact(ax, fontsize=6, max_rows=2)
        elif method == "gmm":
            bins = _hist_bins(values)
            ax.hist(values, bins=bins, density=True, color="#4c78a8", alpha=0.34, label="Importance density")
            curves = _gmm_density_curves(values)
            if curves is not None:
                grid, components, mixture = curves
                for component_index, component in enumerate(components, start=1):
                    ax.plot(grid, component, linewidth=1.4, linestyle="--", label=f"Gaussian {component_index}")
                ax.plot(grid, mixture, color=method_color, linewidth=1.9, label="GMM mixture")
            ax.axvline(threshold, color=method_color, linewidth=1.7, linestyle="--", label="Threshold")
            ax.set_xlabel("GMM: channel importance")
            ax.set_ylabel("Density")
            recomputed_threshold = _gmm_threshold_from_values(values)
            selection_view = "Intersection of two weighted Gaussian components"
            _legend_below_compact(ax, fontsize=6, max_rows=2)
        elif method == "otsu":
            bins = _hist_bins(values)
            ax.hist(values, bins=bins, density=True, color="#4c78a8", alpha=0.32, label="Importance density")
            ax.axvline(threshold, color=method_color, linewidth=1.7, linestyle="--", label="Threshold")
            ax.set_xlabel("Otsu: channel importance")
            ax.set_ylabel("Density")
            recomputed_threshold = _otsu_threshold_from_values(values)
            selection_view = "Maximize between-class variance"
            curve = _otsu_between_class_curve(values)
            if curve is not None:
                centers, variance_curve = curve
                ax_var = ax.twinx()
                ax_var.plot(centers, variance_curve, color=method_color, linewidth=1.9, label="Between-class variance")
                ax_var.set_ylabel("Normalized variance")
                handles, labels = ax.get_legend_handles_labels()
                handles_var, labels_var = ax_var.get_legend_handles_labels()
                all_handles = handles + handles_var
                all_labels = labels + labels_var
                ncol = min(len(all_labels), max(1, int(np.ceil(len(all_labels) / 2))))
                ax.legend(
                    all_handles,
                    all_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.22),
                    fontsize=6,
                    frameon=True,
                    framealpha=0.9,
                    ncol=ncol,
                    columnspacing=0.95,
                    handlelength=1.7,
                    borderaxespad=0.2,
                )
            else:
                _legend_below_compact(ax, fontsize=6, max_rows=2)
        else:
            sorted_values = np.sort(values)
            x = np.arange(1, sorted_values.size + 1)
            ax.scatter(x, sorted_values, s=13, color="#4c78a8", alpha=0.72, linewidths=0)
            ax.axhline(threshold, color=method_color, linewidth=1.6, linestyle="--", label="Threshold")
            ax.set_xlabel("Sorted channel index")
            ax.set_ylabel("Channel importance")
            selection_view = "Threshold on sorted importance"
            _legend_below_compact(ax, fontsize=6, max_rows=2)

        threshold_abs_error = (
            float(abs(threshold - recomputed_threshold))
            if np.isfinite(threshold) and np.isfinite(recomputed_threshold)
            else np.nan
        )
        if np.isfinite(threshold_abs_error) and threshold_abs_error > 1e-5:
            logging.warning(
                "Figure 23 threshold check differs for %s/%s/%s: CSV=%.8f, recomputed=%.8f",
                selected_dataset,
                method,
                selected_layer,
                threshold,
                recomputed_threshold,
            )
        ax.grid(alpha=0.25)
        metadata_rows.append(
            {
                "dataset": selected_dataset,
                "layer_name": selected_layer,
                "plot_role": selected_role,
                "method": method,
                "selection_view": selection_view,
                "importance_source_column": "importance",
                "threshold_source_column": "pruning_threshold",
                "detail_source_paths": str(panel.get("detail_source_paths", "")),
                "summary_source_paths": str(panel.get("summary_source_paths", "")),
                "threshold": threshold,
                "recomputed_threshold_from_importance": recomputed_threshold,
                "threshold_abs_error": threshold_abs_error,
                "threshold_rows_used": int(panel.get("threshold_count", 0)),
                "num_channels": int(values.size),
                "importance_min": float(np.min(values)),
                "importance_max": float(np.max(values)),
                "importance_mean": float(np.mean(values)),
                "importance_std": float(np.std(values, ddof=0)),
                "importance_signature": _values_signature(values),
                "closest_importance": closest_value,
                "criterion_value": criterion_value,
            }
        )
    fig.tight_layout(w_pad=1.6, rect=(0.0, 0.12, 1.0, 1.0))
    return fig, metadata_rows


def figure23_threshold_selection(outputs_root: Path, save_root: Path) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    path = save_root / "figure23_threshold_selection_methods.pdf"
    all_metadata_rows: List[Dict[str, object]] = []
    rendered_any = False
    pdf: PdfPages | None = None

    try:
        for dataset in _datasets(outputs_root, save_root):
            result = _figure23_dataset_panels(outputs_root, dataset)
            if result is None:
                logging.warning("Figure 23 has no complete Kneedle/GMM/Otsu threshold data for %s.", dataset)
                continue
            selected_layer, selected_role, selected_panels = result
            fig, metadata_rows = _build_figure23_threshold_figure(dataset, selected_layer, selected_role, selected_panels)

            dataset_dir = save_root / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            dataset_pdf = dataset_dir / "figure23_threshold_selection_methods.pdf"
            dataset_csv = dataset_dir / "figure23_threshold_selection_methods.csv"
            for row in metadata_rows:
                row["figure_path"] = str(dataset_pdf)
                row["aggregate_figure_path"] = str(path)

            fig.savefig(dataset_pdf, bbox_inches="tight")
            logging.info("Saved figure: %s", dataset_pdf)
            pd.DataFrame(metadata_rows).to_csv(dataset_csv, index=False)
            logging.info("Saved Figure 23 metadata: %s", dataset_csv)

            if pdf is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                pdf = PdfPages(path)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            all_metadata_rows.extend(metadata_rows)
            rendered_any = True
    finally:
        if pdf is not None:
            pdf.close()
            logging.info("Saved aggregate figure: %s", path)

    if not rendered_any:
        _placeholder(path, "No channel-importance threshold data found for Kneedle, GMM, and Otsu.")
        return

    pd.DataFrame(all_metadata_rows).to_csv(save_root / "figure23_threshold_selection_methods.csv", index=False)


def figure5_layerwise(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    path = dataset_dir / "figure5_layerwise_pruning_ratio.pdf"
    frame = _concat_csvs(_find_dataset_files(outputs_root, dataset, "pruning_summary.csv"))
    if frame.empty or "actual_prune_ratio" not in frame.columns:
        _placeholder(path, f"No pruning summary rows found for {dataset}.")
        return

    output_paths = {
        "blueprint": dataset_dir / "figure5_blueprint_stage_pruning_ratio.pdf",
        "middle_conv2": dataset_dir / "figure5_middle_conv2_layerwise_pruning_ratio.pdf",
        "full_block": dataset_dir / "figure5_full_block_layerwise_pruning_ratio.pdf",
        "full_block_conv1": dataset_dir / "figure5_full_block_conv1_pruning_ratio.pdf",
        "full_block_conv2": dataset_dir / "figure5_full_block_conv2_pruning_ratio.pdf",
        "full_block_conv3": dataset_dir / "figure5_full_block_conv3_pruning_ratio.pdf",
    }
    groups = [
        ("Blueprint Stage", CHANNEL_METHODS, "stage_output", "Stage index: stem, down1, down2, down3, down4", output_paths["blueprint"]),
        ("Middle Conv2 Block", MIDDLE_METHODS, "conv2_output_to_conv3_input", "Teacher ResNet bottleneck block index", output_paths["middle_conv2"]),
        ("Full Block Output", FULL_METHODS, "conv3_output", "Teacher ResNet bottleneck block index", output_paths["full_block"]),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9.4, 9.2), sharey=True)
    plotted_any = False
    for ax, (title, methods, plot_role, xlabel, group_path) in zip(axes, groups):
        group_frame = _filter_plot_role(_filter_pruning_group(frame, methods), plot_role)
        if group_frame.empty:
            _placeholder(group_path, f"No pruning summary rows found for {dataset} / {title}.")
            ax.axis("off")
            ax.text(0.5, 0.5, f"No data for {title}", ha="center", va="center", transform=ax.transAxes)
            continue
        _plot_pruning_ratio_group(ax, group_frame, title=title, xlabel=xlabel)
        _save_pruning_ratio_group_pdf(group_frame, group_path, title=title, xlabel=xlabel)
        plotted_any = True
    full_frame = _filter_pruning_group(frame, FULL_METHODS)
    _save_full_role_ratio_pdf(
        full_frame,
        output_paths["full_block_conv1"],
        ratio_column="internal_prune_ratio",
        title="Full Block Conv1 Output",
        xlabel="Teacher ResNet bottleneck block index",
    )
    _save_full_role_ratio_pdf(
        full_frame,
        output_paths["full_block_conv2"],
        ratio_column="conv2_prune_ratio",
        title="Full Block Conv2 Output",
        xlabel="Teacher ResNet bottleneck block index",
    )
    _save_full_role_ratio_pdf(
        full_frame,
        output_paths["full_block_conv3"],
        ratio_column="actual_prune_ratio",
        title="Full Block Conv3 Output",
        xlabel="Teacher ResNet bottleneck block index",
    )
    if not plotted_any:
        plt.close(fig)
        _placeholder(path, f"No pruning rows found for {dataset}.")
        return
    fig.tight_layout(h_pad=3.2)
    _save_pdf(fig, path)


def _pruning_ratio_series(frame: pd.DataFrame) -> pd.Series:
    method_series = frame.get("prune_method", pd.Series([""] * len(frame), index=frame.index)).astype(str).str.lower()
    ratios = pd.Series([np.nan] * len(frame), index=frame.index, dtype=float)
    for index, row_series in frame.iterrows():
        if _method_family(method_series.loc[index]) != "static":
            continue
        ratios.loc[index] = _configured_static_ratio_from_row(row_series)
    return ratios


def _filter_pruning_group(frame: pd.DataFrame, methods: set[str]) -> pd.DataFrame:
    method_series = frame.get("prune_method", pd.Series(["Method"] * len(frame), index=frame.index)).astype(str).str.lower()
    return frame[method_series.isin(methods)].copy()


def _filter_plot_role(frame: pd.DataFrame, plot_role: str) -> pd.DataFrame:
    if frame.empty or "plot_role" not in frame.columns:
        return frame.copy()
    return frame[frame["plot_role"].fillna("").astype(str).eq(plot_role)].copy()


def _legend_below_compact(ax, *, fontsize: int = 7, max_rows: int = 2) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ncol = min(len(labels), max(1, int(np.ceil(len(labels) / max(1, max_rows)))))
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=fontsize,
        frameon=True,
        framealpha=0.9,
        ncol=ncol,
        columnspacing=0.95,
        handlelength=1.7,
        borderaxespad=0.2,
    )


def _plot_pruning_ratio_group(ax, frame: pd.DataFrame, *, title: str, xlabel: str) -> None:
    method_series = frame.get("prune_method", pd.Series(["Method"] * len(frame), index=frame.index)).astype(str).str.lower()
    ratio_series = _pruning_ratio_series(frame)
    grouped_keys = pd.DataFrame({"method": method_series, "ratio": ratio_series}, index=frame.index)
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for index, ((method, ratio), group_index) in enumerate(grouped_keys.groupby(["method", "ratio"], dropna=False).groups.items()):
        group = frame.loc[list(group_index)]
        if "plot_index" in group.columns:
            group = group.assign(_plot_index=pd.to_numeric(group["plot_index"], errors="coerce")).sort_values(["_plot_index", "layer_name"], na_position="last")
        ratios = pd.to_numeric(group["actual_prune_ratio"], errors="coerce").dropna().to_numpy()
        if str(method).endswith("static") and not np.isnan(_safe_float(ratio)) and ratios.size:
            ratios = np.full_like(ratios, fill_value=float(ratio), dtype=float)
        if ratios.size == 0:
            continue
        label = _display_method(method, ratio)
        ax.plot(
            np.arange(1, len(ratios) + 1),
            ratios,
            linewidth=1.6,
            label=label,
            color=colors[index % len(colors)] if colors else None,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pruning ratio")
    ax.set_xticks([])
    ax.grid(alpha=0.25)
    _legend_below_compact(ax)


def _save_pruning_ratio_group_pdf(frame: pd.DataFrame, path: Path, *, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    _plot_pruning_ratio_group(ax, frame, title=title, xlabel=xlabel)
    fig.tight_layout()
    _save_pdf(fig, path)


def _save_full_role_ratio_pdf(frame: pd.DataFrame, path: Path, *, ratio_column: str, title: str, xlabel: str) -> None:
    if frame.empty or ratio_column not in frame.columns:
        _placeholder(path, f"No {ratio_column} rows found for {title}.")
        return
    plot_frame = frame.copy()
    plot_frame["actual_prune_ratio"] = pd.to_numeric(plot_frame[ratio_column], errors="coerce")
    plot_frame = plot_frame.dropna(subset=["actual_prune_ratio"])
    if plot_frame.empty:
        _placeholder(path, f"No valid {ratio_column} values found for {title}.")
        return
    _save_pruning_ratio_group_pdf(plot_frame, path, title=title, xlabel=xlabel)


def _figure6_metric_rows(outputs_root: Path, dataset: str) -> pd.DataFrame:
    base_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR
    rows = []
    for label, relative_dir in _figure15_method_dirs(outputs_root, dataset):
        metrics_path = base_root / relative_dir / "metrics_summary.csv"
        row = _best_test_row(metrics_path)
        if row is None:
            continue
        if relative_dir == Path("1_teacher"):
            raw_method = "teacher"
            group = "Teacher"
        else:
            output_dir = base_root / relative_dir.parent
            raw_method, _ = _method_from_output_dir(output_dir)
            group = _method_group_label(raw_method)
        rows.append(
            {
                "Method": label,
                "Group": group,
                "Params (M)": _params_from_metric_params(row),
                "Dice": _dice_from_row(row),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["Params (M)"] = pd.to_numeric(frame["Params (M)"], errors="coerce")
    frame["Dice"] = pd.to_numeric(frame["Dice"], errors="coerce")
    return frame.dropna(subset=["Params (M)", "Dice"]).reset_index(drop=True)


def figure6_tradeoff(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    path = dataset_dir / "figure6_accuracy_efficiency_tradeoff.pdf"
    frame = _figure6_metric_rows(outputs_root, dataset)
    if frame.empty:
        frame = _read_csv(dataset_dir / "table6_method_comparison.csv")
    if frame.empty:
        _placeholder(path, "No after fine-tuning metrics available for trade-off figure.")
        return
    method_col = _method_column(frame)
    if method_col is None:
        _placeholder(path, "No Method column available for trade-off figure.")
        return
    dice_col = "Dice" if "Dice" in frame.columns else "Dice $\\uparrow$"
    if dice_col not in frame.columns:
        _placeholder(path, "No Dice column available for trade-off figure.")
        return
    x_col = "Params (M)" if "Params (M)" in frame.columns else "Params"
    if x_col not in frame.columns:
        _placeholder(path, "No Params column available for trade-off figure.")
        return
    x_values = pd.to_numeric(frame[x_col], errors="coerce")
    scale = 1e6 if x_col != "Params (M)" and x_values.max(skipna=True) > 1e5 else 1.0
    xlabel = "Params (M)" if scale == 1e6 else x_col
    frame = frame.assign(_x=x_values / scale, _y=pd.to_numeric(frame[dice_col], errors="coerce"))
    frame = frame.dropna(subset=["_x", "_y"])
    if frame.empty:
        _placeholder(path, "No valid Dice/efficiency values available for trade-off figure.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    groups = frame.get("Group", pd.Series(["Method"] * len(frame))).astype(str)
    for group, group_frame in frame.groupby(groups):
        ax.scatter(group_frame["_x"], group_frame["_y"], s=52, alpha=0.88, label=group)

    label_candidates = frame[
        frame[method_col].astype(str).str.contains("Teacher|GMM|Middle", regex=True, case=False, na=False)
    ]
    for offset_index, (_, row) in enumerate(label_candidates.iterrows()):
        offset = (6, 6 + 5 * (offset_index % 2))
        ax.annotate(
            str(row.get(method_col, ""))[:28],
            (row["_x"], row["_y"]),
            fontsize=7,
            xytext=offset,
            textcoords="offset points",
            arrowprops={"arrowstyle": "-", "lw": 0.6, "alpha": 0.55},
        )

    teacher = frame[frame[method_col].astype(str).str.contains("Teacher", regex=True, case=False, na=False)]
    proposed = frame[frame[method_col].astype(str).str.contains("Middle GMM|GMM", regex=True, case=False, na=False)]
    if not teacher.empty and not proposed.empty:
        src = teacher.iloc[0]
        dst = proposed.sort_values("_y", ascending=False, na_position="last").iloc[0]
        ax.annotate(
            "",
            xy=(dst["_x"], dst["_y"]),
            xytext=(src["_x"], src["_y"]),
            arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "black", "alpha": 0.7},
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Dice after fine-tuning")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9)
    frame.drop(columns=[column for column in ("_x", "_y") if column in frame.columns]).to_csv(dataset_dir / "figure6_params_dice_tradeoff.csv", index=False)
    _save_pdf(fig, path)


def _student_dirs_for_loss(outputs_root: Path, dataset: str, loss_tag: str) -> List[Path]:
    loss_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR / loss_tag
    if not loss_root.is_dir():
        return []
    discovered = []
    for output_dir in loss_root.iterdir():
        if not output_dir.is_dir() or not output_dir.name.startswith("output_"):
            continue
        student_dir = output_dir / "3_student"
        if not student_dir.is_dir():
            continue
        raw_method, ratio = _method_from_output_dir(output_dir)
        discovered.append((_method_sort_key(raw_method, ratio), student_dir))
    return [student_dir for _, student_dir in sorted(discovered, key=lambda item: item[0])]


def _curve_paths_for_student_dir(student_dir: Path) -> List[Path]:
    return [
        student_dir / "metrics" / "student_epoch_diagnostics.csv",
        student_dir / "checkpoints" / "train_log.csv",
        student_dir / "metrics_summary.csv",
        student_dir / "metrics" / "student_metrics.csv",
    ]


def _training_curve_frame(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    if "phase" in frame.columns:
        student = frame[frame["phase"].fillna("").astype(str).str.lower().eq("student")]
        if not student.empty:
            frame = student
    if "split" in frame.columns:
        val_frame = frame[frame["split"].fillna("").astype(str).str.lower().eq("val")]
        if not val_frame.empty:
            frame = val_frame
    if "checkpoint_name" in frame.columns:
        last_rows = frame[frame["checkpoint_name"].fillna("").astype(str).eq("last.pth")]
        if not last_rows.empty:
            frame = last_rows
    dice_col = _first_existing_column(frame, ("val_macro_dice", "val_dice", "best_val_dice", "dice", "Dice"))
    train_loss_col = _first_existing_column(frame, ("train_total_loss", "total_loss", "train_seg_loss"))
    if dice_col is None and train_loss_col is None:
        return pd.DataFrame()
    frame = frame.copy()
    frame["_epoch"] = pd.to_numeric(frame.get("epoch", pd.Series(range(1, len(frame) + 1))), errors="coerce")
    if dice_col is not None:
        frame["_dice"] = pd.to_numeric(frame[dice_col], errors="coerce")
    if train_loss_col is not None:
        frame["_train_loss"] = pd.to_numeric(frame[train_loss_col], errors="coerce")
    required = ["_epoch"]
    if dice_col is not None:
        required.append("_dice")
    if train_loss_col is not None:
        required.append("_train_loss")
    frame = frame.dropna(subset=required).sort_values("_epoch")
    return frame.drop_duplicates(subset=["_epoch"], keep="last")


def _select_training_curve(outputs_root: Path, dataset: str, loss_tag: str) -> tuple[Path | None, pd.DataFrame]:
    base_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR / loss_tag
    preferred_dir = base_root / "output_s2_kneedle_auto_no" / "3_student"
    candidate_paths = []
    if preferred_dir.is_dir():
        candidate_paths.extend(_curve_paths_for_student_dir(preferred_dir))
    for student_dir in _student_dirs_for_loss(outputs_root, dataset, loss_tag):
        candidate_paths.extend(_curve_paths_for_student_dir(student_dir))

    seen = set()
    best_path: Path | None = None
    best_frame = pd.DataFrame()
    best_score = (-1, -np.inf)
    for candidate_path in candidate_paths:
        if candidate_path in seen or not candidate_path.is_file():
            continue
        seen.add(candidate_path)
        frame = _training_curve_frame(candidate_path)
        if frame.empty:
            continue
        final_dice = float(frame["_dice"].dropna().iloc[-1]) if "_dice" in frame and frame["_dice"].dropna().size else -np.inf
        score = (len(frame), final_dice)
        if best_path is None or score > best_score:
            best_path = candidate_path
            best_frame = frame
            best_score = score
    return best_path, best_frame


def figure7_training(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    curve_path, frame = _select_training_curve(outputs_root, dataset, PGD_LOSS_TAG)
    if frame.empty:
        _placeholder(dataset_dir / "figure7_training_curve.pdf", f"No training log found for {dataset}.")
        return
    logging.info("Figure 7 training curve for %s uses %s", dataset, curve_path)
    epoch = frame["_epoch"]
    saved_loss = False
    saved_dice = False
    if "_train_loss" in frame.columns:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(epoch, _smooth(frame["_train_loss"]), label="Training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.grid(alpha=0.25)
        ax.legend()
        _save_pdf(fig, dataset_dir / "figure7a_training_loss_curve.pdf")
        saved_loss = True
    if "_dice" in frame.columns:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(epoch, _smooth(frame["_dice"]), label="Validation Dice")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Dice")
        ax.grid(alpha=0.25)
        ax.legend()
        _save_pdf(fig, dataset_dir / "figure7b_validation_dice_curve.pdf")
        saved_dice = True
    if saved_loss and saved_dice:
        fig, axes = plt.subplots(2, 1, figsize=(7, 6.2), sharex=True)
        axes[0].plot(epoch, _smooth(frame["_train_loss"]), label="Training loss", color="#4c78a8")
        axes[0].set_ylabel("Training loss")
        axes[0].grid(alpha=0.25)
        axes[1].plot(epoch, _smooth(frame["_dice"]), label="Validation Dice", color="#f58518")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation Dice")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        _save_pdf(fig, dataset_dir / "figure7_training_curve.pdf")


def _read_image(path_value) -> np.ndarray | None:
    try:
        path = Path(str(path_value))
        if not path.is_file():
            return None
        return mpimg.imread(path)
    except Exception:
        return None


def _as_2d_mask(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    array = np.asarray(image)
    array = np.squeeze(array)
    if array.ndim == 3:
        # Masks may be saved as RGB/RGBA PNGs. Contour needs a single 2D plane.
        array = array[..., :3].mean(axis=2)
    if array.ndim != 2:
        return None
    return array


def _sample_metrics(outputs_root: Path, dataset: str) -> pd.DataFrame:
    return _concat_csvs(_find_dataset_files(outputs_root, dataset, "sample_metrics.csv"))


def _sample_id_column(frame: pd.DataFrame) -> str | None:
    return _first_existing_column(frame, ("sample_id", "case"))


def _valid_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "prediction_path" not in frame.columns:
        return pd.DataFrame()
    prediction = frame["prediction_path"].fillna("").astype(str)
    return frame[~prediction.str.lower().isin({"", "nan", "none"})].copy()


def _teacher_sample_frame(outputs_root: Path, dataset: str) -> pd.DataFrame:
    return _read_csv(outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR / "1_teacher" / "evaluations" / "test" / "sample_metrics.csv")


def _student_sample_frame(outputs_root: Path, dataset: str, loss_tag: str, raw_method: str, target_ratio: float | None = None) -> pd.DataFrame:
    loss_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR / loss_tag
    if not loss_root.is_dir():
        return pd.DataFrame()
    candidates = []
    for output_dir in loss_root.iterdir():
        if not output_dir.is_dir() or not output_dir.name.startswith("output_"):
            continue
        method, ratio = _method_from_output_dir(output_dir)
        if method != raw_method:
            continue
        if target_ratio is not None and not _ratio_matches(ratio, target_ratio):
            continue
        candidates.append((_method_sort_key(method, ratio), output_dir / "3_student" / "evaluations" / "test" / "sample_metrics.csv"))
    for _, csv_path in sorted(candidates, key=lambda item: item[0]):
        frame = _read_csv(csv_path)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _row_for_sample(frame: pd.DataFrame, sample_id: str) -> Dict[str, object] | None:
    if frame.empty:
        return None
    id_col = _sample_id_column(frame)
    if id_col is None:
        return None
    subset = frame[frame[id_col].astype(str).eq(str(sample_id))]
    if subset.empty:
        return None
    valid = _valid_prediction_frame(subset)
    return (valid if not valid.empty else subset).iloc[0].to_dict()


def _row_prediction_is_drawable(row: Dict[str, object] | None) -> bool:
    return row is not None and _read_image(row.get("prediction_path")) is not None


def _strict_gallery_sample(anchor_frame: pd.DataFrame, required_prediction_frames: Iterable[pd.DataFrame]) -> tuple[str, Dict[str, object]] | None:
    anchor_frame = _valid_prediction_frame(anchor_frame)
    id_col = _sample_id_column(anchor_frame)
    if anchor_frame.empty or id_col is None:
        return None
    sample_ids = sorted(str(value) for value in anchor_frame[id_col].dropna().astype(str).unique())
    for sample_id in sample_ids:
        anchor_row = _row_for_sample(anchor_frame, sample_id)
        if anchor_row is None:
            continue
        if _read_image(anchor_row.get("image_path")) is None or _read_image(anchor_row.get("mask_path")) is None:
            continue
        if not _row_prediction_is_drawable(anchor_row):
            continue
        valid = True
        for frame in required_prediction_frames:
            row = _row_for_sample(frame, sample_id)
            if not _row_prediction_is_drawable(row):
                valid = False
                break
        if valid:
            return sample_id, anchor_row
    return None


def _gallery_group_specs() -> List[Dict[str, object]]:
    return [
        {
            "figure_id": "figure20",
            "name": "blueprint_prediction_gallery",
            "path": "figure20_blueprint_prediction_gallery.pdf",
            "csv": "figure20_blueprint_prediction_gallery.csv",
            "methods": [("static", 0.5, "Static 0.5"), ("kneedle", None, "Kneedle"), ("otsu", None, "Otsu"), ("gmm", None, "GMM")],
        },
        {
            "figure_id": "figure21",
            "name": "middle_conv2_prediction_gallery",
            "path": "figure21_middle_conv2_prediction_gallery.pdf",
            "csv": "figure21_middle_conv2_prediction_gallery.csv",
            "methods": [
                ("middle_static", 0.5, "Static 0.5"),
                ("middle_kneedle", None, "Kneedle"),
                ("middle_otsu", None, "Otsu"),
                ("middle_gmm", None, "GMM"),
            ],
        },
        {
            "figure_id": "figure22",
            "name": "full_block_prediction_gallery",
            "path": "figure22_full_block_prediction_gallery.pdf",
            "csv": "figure22_full_block_prediction_gallery.csv",
            "methods": [
                ("full_static", 0.5, "Static 0.5"),
                ("full_kneedle", None, "Kneedle"),
                ("full_otsu", None, "Otsu"),
                ("full_gmm", None, "GMM"),
            ],
        },
    ]


def _loss_gallery_specs() -> List[tuple[str, str]]:
    return [("loss_seg_only", "loss_seg_only"), ("loss_seg_kd", PGD_LOSS_TAG)]


def _dataset_display_name(dataset: str) -> str:
    return str(dataset).replace("_", " ").upper()


def _prediction_gallery(outputs_root: Path, save_root: Path, spec: Dict[str, object], loss_label: str, loss_tag: str) -> None:
    gallery_dir = save_root / loss_label
    gallery_dir.mkdir(parents=True, exist_ok=True)
    output_path = gallery_dir / str(spec["path"])
    methods = list(spec["methods"])
    columns = ["Image", "GT", "Teacher", *[method_label for _, _, method_label in methods]]

    rows = []
    metadata_rows = []
    for dataset in _datasets(outputs_root, save_root):
        teacher_frame = _teacher_sample_frame(outputs_root, dataset)
        method_frames: Dict[str, pd.DataFrame] = {}
        for raw_method, target_ratio, _ in methods:
            method_frames[raw_method] = _student_sample_frame(outputs_root, dataset, loss_tag, raw_method, target_ratio)

        anchor_method = str(methods[0][0])
        anchor_frame = method_frames.get(anchor_method, pd.DataFrame())
        strict_sample = _strict_gallery_sample(anchor_frame, [teacher_frame, *method_frames.values()])
        if strict_sample is None:
            logging.warning(
                "Prediction gallery skips %s/%s/%s because no exact static-ratio sample has image, GT, teacher prediction, and all method predictions.",
                dataset,
                loss_tag,
                spec["name"],
            )
            continue
        sample_id, image_source = strict_sample

        panels = [
            _read_image(image_source.get("image_path")),
            _read_image(image_source.get("mask_path")),
        ]
        metadata_rows.extend(
            [
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "column": "Image",
                    "loss_tag": loss_tag,
                    "method": "image",
                    "anchor_method": anchor_method,
                    "source_csv_path": image_source.get("csv_source_path", image_source.get("source_path", "")),
                    "image_path": image_source.get("image_path"),
                    "mask_path": image_source.get("mask_path"),
                    "prediction_path": "NaN",
                    "strict_exact_sample": True,
                },
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "column": "GT",
                    "loss_tag": loss_tag,
                    "method": "ground_truth",
                    "anchor_method": anchor_method,
                    "source_csv_path": image_source.get("csv_source_path", image_source.get("source_path", "")),
                    "image_path": image_source.get("image_path"),
                    "mask_path": image_source.get("mask_path"),
                    "prediction_path": "NaN",
                    "strict_exact_sample": True,
                },
            ]
        )
        teacher_row = _row_for_sample(teacher_frame, sample_id)
        panels.append(_read_image(teacher_row.get("prediction_path")) if teacher_row is not None else None)
        metadata_rows.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "column": "Teacher",
                "loss_tag": "teacher",
                "method": "teacher",
                "anchor_method": anchor_method,
                "source_csv_path": teacher_row.get("csv_source_path", teacher_row.get("source_path", "")) if teacher_row is not None else "NaN",
                "image_path": image_source.get("image_path"),
                "mask_path": image_source.get("mask_path"),
                "row_image_path": teacher_row.get("image_path") if teacher_row is not None else "NaN",
                "row_mask_path": teacher_row.get("mask_path") if teacher_row is not None else "NaN",
                "prediction_path": teacher_row.get("prediction_path") if teacher_row is not None else "NaN",
                "strict_exact_sample": True,
            }
        )

        for raw_method, _, method_label in methods:
            row = _row_for_sample(method_frames[raw_method], sample_id)
            panels.append(_read_image(row.get("prediction_path")) if row is not None else None)
            metadata_rows.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "column": method_label,
                    "loss_tag": loss_tag,
                    "method": raw_method,
                    "anchor_method": anchor_method,
                    "source_csv_path": row.get("csv_source_path", row.get("source_path", "")) if row is not None else "NaN",
                    "image_path": image_source.get("image_path"),
                    "mask_path": image_source.get("mask_path"),
                    "row_image_path": row.get("image_path") if row is not None else "NaN",
                    "row_mask_path": row.get("mask_path") if row is not None else "NaN",
                    "prediction_path": row.get("prediction_path") if row is not None else "NaN",
                    "strict_exact_sample": True,
                }
            )
        rows.append({"dataset": dataset, "sample_id": sample_id, "panels": panels})

    pd.DataFrame(metadata_rows).to_csv(gallery_dir / str(spec["csv"]), index=False)
    if not rows:
        _placeholder(output_path, "No drawable gallery samples found.")
        return

    nrows = len(rows)
    ncols = len(columns)
    fig_width = max(11.0, min(24.0, 1.34 * ncols + 1.4))
    fig_height = max(2.8, 1.45 * nrows + 0.85)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    for col_index, label in enumerate(columns):
        axes[0, col_index].text(0.5, 1.06, label, transform=axes[0, col_index].transAxes, ha="center", va="bottom", fontsize=7)
    for row_index, row in enumerate(rows):
        axes[row_index, 0].text(
            -0.10,
            0.5,
            _dataset_display_name(str(row["dataset"])),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
        for col_index, image in enumerate(row["panels"]):
            ax = axes[row_index, col_index]
            ax.axis("off")
            if image is None:
                ax.text(0.5, 0.5, "NA", ha="center", va="center", fontsize=7, color="#777777", transform=ax.transAxes)
                continue
            ax.imshow(image, cmap="gray")
    fig.subplots_adjust(left=0.08, right=0.995, top=0.91, bottom=0.02, wspace=0.03, hspace=0.08)
    _save_pdf(fig, output_path)


def figure_prediction_galleries(outputs_root: Path, save_root: Path) -> None:
    for loss_label, loss_tag in _loss_gallery_specs():
        for spec in _gallery_group_specs():
            _prediction_gallery(outputs_root, save_root, spec, loss_label, loss_tag)


def figure8_visual(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    frame = _sample_metrics(outputs_root, dataset)
    if frame.empty or "prediction_path" not in frame.columns:
        logging.warning("Skip figure8 for %s because sample_metrics/predictions are missing.", dataset)
        return
    frame = frame[frame["prediction_path"].astype(str).ne("NaN")]
    if frame.empty:
        logging.warning("Skip figure8 for %s because no prediction paths are available.", dataset)
        return
    row = frame.iloc[(pd.to_numeric(frame["dice"], errors="coerce") - pd.to_numeric(frame["dice"], errors="coerce").median()).abs().argsort().iloc[0]]
    sample_id = row["sample_id"]
    sample_rows = frame[frame["sample_id"].astype(str).eq(str(sample_id))].head(4)
    panels = [("Image", _read_image(row.get("image_path"))), ("GT", _read_image(row.get("mask_path")))]
    for _, sample_row in sample_rows.iterrows():
        panels.append((str(sample_row.get("method", "Prediction"))[:18], _read_image(sample_row.get("prediction_path"))))
    panels = [(label, image) for label, image in panels if image is not None]
    if len(panels) < 3:
        logging.warning("Skip figure8 for %s because images cannot be loaded.", dataset)
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
    for ax, (label, image) in zip(np.ravel(axes), panels):
        ax.imshow(image, cmap="gray")
        ax.axis("off")
    _save_pdf(fig, dataset_dir / "figure8_visual_comparison.pdf")


def figure9_failures(outputs_root: Path, dataset: str, dataset_dir: Path, topk: int = 5) -> None:
    frame = _sample_metrics(outputs_root, dataset)
    if frame.empty or "dice" not in frame.columns:
        logging.warning("Skip figure9 for %s because sample_metrics.csv is missing.", dataset)
        return
    sorted_frame = frame.sort_values("dice", ascending=True, na_position="last").head(topk)
    sorted_frame.to_csv(dataset_dir / "failure_cases_topk.csv", index=False)
    drawable = sorted_frame[sorted_frame["prediction_path"].astype(str).ne("NaN")]
    if drawable.empty:
        logging.warning("No drawable failure cases for %s; CSV was still saved.", dataset)
        return
    rows = []
    for _, row in drawable.head(3).iterrows():
        rows.append([_read_image(row.get("image_path")), _read_image(row.get("mask_path")), _read_image(row.get("prediction_path"))])
    rows = [[image for image in row if image is not None] for row in rows]
    rows = [row for row in rows if len(row) >= 3]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    axes = np.atleast_2d(axes)
    for row_axes, images in zip(axes, rows):
        for ax, image in zip(row_axes, images[:3]):
            ax.imshow(image, cmap="gray")
            ax.axis("off")
    _save_pdf(fig, dataset_dir / "figure9_failure_cases.pdf")


def figure10_boundary(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    frame = _sample_metrics(outputs_root, dataset)
    if frame.empty or "dice" not in frame.columns:
        logging.warning("Skip figure10 for %s because sample metrics are missing.", dataset)
        return
    row = frame.sort_values("dice", ascending=True, na_position="last").iloc[0]
    image = _read_image(row.get("image_path"))
    gt = _as_2d_mask(_read_image(row.get("mask_path")))
    pred = _as_2d_mask(_read_image(row.get("prediction_path")))
    if image is None or gt is None or pred is None:
        logging.warning("Skip figure10 for %s because boundary masks cannot be loaded.", dataset)
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(image, cmap="gray")
    ax.contour(gt, levels=[0.5], colors="lime", linewidths=1)
    ax.contour(pred, levels=[0.5], colors="red", linewidths=1)
    ax.axis("off")
    _save_pdf(fig, dataset_dir / "figure10_boundary_comparison.pdf")


def _method_from_output_dir(output_dir: Path) -> tuple[str, float]:
    parts = output_dir.name.split("_")
    if len(parts) >= 3 and re.fullmatch(r"s\d+", parts[1]):
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


def _student_metric_score(student_dir: Path) -> float:
    metrics_path = student_dir / "metrics" / "student_metrics.csv"
    frame = _read_csv(metrics_path)
    if frame.empty or "dice" not in frame.columns:
        return -np.inf
    if "split" in frame.columns:
        test_frame = frame[frame["split"].fillna("").astype(str).str.lower().eq("test")]
        if not test_frame.empty:
            frame = test_frame
    scores = pd.to_numeric(frame["dice"], errors="coerce").dropna()
    return float(scores.max()) if not scores.empty else -np.inf


def _student_channel_context_from_output_dirs(outputs_root: Path, dataset: str) -> tuple[str, Path | None, Path | None]:
    focus_roots = _pgd_focus_roots(outputs_root, dataset)
    if not focus_roots:
        logging.warning("PGD focus root is missing for %s", dataset)
        return "best student", None, None

    candidates: List[tuple[float, int, str, Path, Path]] = []
    method_priority = {
        "middle_kneedle": 0,
        "middle_otsu": 1,
        "middle_gmm": 2,
        "kneedle": 3,
        "otsu": 4,
        "gmm": 5,
        "middle_static": 6,
        "static": 7,
        "full_kneedle": 8,
        "full_otsu": 9,
        "full_gmm": 10,
        "full_static": 11,
    }
    for focus_root in focus_roots:
        for output_dir in sorted(path for path in focus_root.iterdir() if path.is_dir() and path.name.startswith("output_")):
            student_dir = output_dir / "3_student"
            summary_path = _student_final_channel_summary_path(student_dir)
            if summary_path is None:
                continue
            raw_method, ratio = _method_from_output_dir(output_dir)
            label = _display_method(raw_method, ratio)
            score = _student_metric_score(student_dir)
            priority = method_priority.get(raw_method, 99)
            candidates.append((score, -priority, label, student_dir, summary_path))

    if not candidates:
        logging.warning("No student channel-analysis candidates found for %s", dataset)
        return "best student", None, None

    score, _, label, student_dir, summary_path = max(candidates, key=lambda item: (item[0], item[1]))
    logging.info("Selected student channel source for %s | method=%s | dice=%.6f | dir=%s", dataset, label, score, student_dir)
    return label, student_dir, summary_path


def _pruning_analysis_paths_from_student_dir(student_dir: Path | None) -> tuple[Path | None, Path | None]:
    if student_dir is None:
        return None, None
    output_dir = student_dir.parent
    analysis_dir = output_dir / "2_pruning" / "artifacts" / "pruning_analysis"
    comparison_path = analysis_dir / "teacher_vs_student_channels.csv"
    summary_path = analysis_dir / "pruning_summary.csv"
    return (
        comparison_path if comparison_path.is_file() else None,
        summary_path if summary_path.is_file() else None,
    )


def _best_student_channel_context(outputs_root: Path, dataset: str) -> tuple[str, Path | None]:
    best_score = -np.inf
    best_label = "best student"
    best_path: Path | None = None
    for metrics_path in _find_dataset_files(outputs_root, dataset, "student_metrics.csv"):
        frame = _read_csv(metrics_path)
        if frame.empty or "dice" not in frame.columns:
            continue
        if "split" in frame.columns:
            test_frame = frame[frame["split"].fillna("").astype(str).str.lower().eq("test")]
            if not test_frame.empty:
                frame = test_frame
        for _, row in frame.iterrows():
            dice = _safe_float(row.get("dice"), default=-np.inf)
            if dice > best_score:
                method = _clean_text(row.get("prune_method") or row.get("method") or row.get("threshold_method"))
                ratio = row.get("static_prune_ratio", row.get("pruning_ratio", np.nan))
                best_score = dice
                best_label = _display_method(method, ratio)
                best_path = metrics_path
    return best_label, best_path


def _phase_dir_from_metric_path(metrics_path: Path | None) -> Path | None:
    if metrics_path is None:
        return None
    for parent in metrics_path.parents:
        if parent.name in {"1_teacher", "2_pruning", "3_student"}:
            return parent
    return metrics_path.parent.parent if metrics_path.parent.name == "metrics" else metrics_path.parent


def _read_first_channel_table(paths: Sequence[Path]) -> pd.DataFrame:
    for path in paths:
        frame = _read_csv(path)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _read_first_channel_table_with_path(paths: Sequence[Path]) -> tuple[pd.DataFrame, Path | None]:
    for path in paths:
        frame = _read_csv(path)
        if not frame.empty:
            return frame, path
    return pd.DataFrame(), None


def _teacher_channel_summary_path(outputs_root: Path, dataset: str) -> Path | None:
    channel_root = _pgd_teacher_phase_root(outputs_root, dataset) / "artifacts" / "channel_analysis"
    candidates = [
        channel_root / "channel_summary.csv",
        channel_root / "teacher_channel_summary.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    logging.warning("Teacher channel summary not found for %s under %s", dataset, channel_root)
    return None


def _student_final_channel_summary_path(student_phase_dir: Path | None) -> Path | None:
    if student_phase_dir is None:
        return None
    channel_root = student_phase_dir / "artifacts" / "channel_analysis"
    candidates = [
        channel_root / "student_final_channel_summary.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    logging.warning("Student channel summary not found under %s", channel_root)
    return None


def figure11_12(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    best_label, student_phase_dir, student_summary_path = _student_channel_context_from_output_dirs(outputs_root, dataset)
    comparison_path, pruning_summary_path = _pruning_analysis_paths_from_student_dir(student_phase_dir)
    comparison_frame = _read_csv(comparison_path) if comparison_path is not None else pd.DataFrame()
    pruning_summary = _read_csv(pruning_summary_path) if pruning_summary_path is not None else pd.DataFrame()
    logging.info("Figure11/12 student phase dir for %s: %s", dataset, student_phase_dir or "missing")
    logging.info("Figure11/12 teacher-vs-student source for %s: %s", dataset, comparison_path or "missing")
    logging.info("Figure11/12 pruning summary source for %s: %s", dataset, pruning_summary_path or "missing")
    logging.info("Figure11/12 raw student channel source for %s: %s", dataset, student_summary_path or "not used")

    if comparison_frame.empty:
        _placeholder(dataset_dir / "figure11_output_channels_per_layer_pruned_student.pdf", f"No teacher-vs-student pruning rows found for {dataset}.")
        _placeholder(dataset_dir / "figure12_mean_channel_importance_per_layer_pruned_student.pdf", f"No pruning summary rows found for {dataset}.")
        return

    method_slug = _method_slug(best_label)
    comparison_frame = comparison_frame.copy()
    comparison_frame["_teacher"] = pd.to_numeric(comparison_frame.get("teacher_out_channels"), errors="coerce")
    comparison_frame["_student"] = pd.to_numeric(comparison_frame.get("student_out_channels"), errors="coerce")
    comparison_frame = comparison_frame.dropna(subset=["_teacher", "_student"])
    if comparison_frame.empty:
        _placeholder(dataset_dir / "figure11_output_channels_per_layer_pruned_student.pdf", f"No valid teacher/student channel columns found for {dataset}.")
        return

    x = np.arange(1, len(comparison_frame) + 1)
    width = 0.42
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar(x - width / 2, comparison_frame["_teacher"], width=width, label="Teacher", color="#8da0cb")
    ax.bar(x + width / 2, comparison_frame["_student"], width=width, label=f"Student ({best_label})", color="#66c2a5")
    ax.set_xlabel("Pruned layer index (1, 2, 3, ...)")
    ax.set_ylabel("Output channels")
    ax.set_xticks([])
    ax.grid(alpha=0.22, axis="y")
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
    base_path = dataset_dir / "figure11_output_channels_per_layer_pruned_student.pdf"
    alias_path = dataset_dir / f"figure11_output_channels_per_layer_pruned_student_best_student_{method_slug}.pdf"
    _save_pdf_multi(fig, [base_path, alias_path])

    if pruning_summary.empty or "importance_mean" not in pruning_summary.columns:
        _placeholder(dataset_dir / "figure12_mean_channel_importance_per_layer_pruned_student.pdf", f"No pruning importance summary found for {dataset}.")
        return
    importance = pd.to_numeric(pruning_summary["importance_mean"], errors="coerce").dropna()
    if importance.empty:
        _placeholder(dataset_dir / "figure12_mean_channel_importance_per_layer_pruned_student.pdf", f"No valid importance values found for {dataset}.")
        return
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar(np.arange(1, len(importance) + 1), importance, color="#6baed6")
    ax.set_xlabel("Pruned layer index (1, 2, 3, ...)")
    ax.set_ylabel("Mean channel importance")
    ax.set_xticks([])
    ax.grid(alpha=0.22, axis="y")
    base_path = dataset_dir / "figure12_mean_channel_importance_per_layer_pruned_student.pdf"
    alias_path = dataset_dir / f"figure12_mean_channel_importance_per_layer_pruned_student_best_student_{method_slug}.pdf"
    _save_pdf_multi(fig, [base_path, alias_path])


def figure13_search(dataset_dir: Path) -> None:
    table = _read_csv(dataset_dir / "table5_computational_cost.csv")
    if table.empty:
        _placeholder(dataset_dir / "figure13_search_time_comparison.pdf", "No timing table available for search-time figure.")
        return
    method_col = _method_column(table)
    if method_col is None or "Search Time (s)" not in table.columns:
        _placeholder(dataset_dir / "figure13_search_time_comparison.pdf", "No Method/Search Time columns available for search-time figure.")
        return
    values = pd.to_numeric(table["Search Time (s)"], errors="coerce").fillna(0.0)
    table = table.assign(_search_time=values)
    table = table[table["_search_time"].gt(0.0)].copy()
    if table.empty:
        _placeholder(dataset_dir / "figure13_search_time_comparison.pdf", "No non-zero search-time rows available for search-time figure.")
        return
    labels = table[method_col].astype(str).tolist()
    fig_width = max(7.5, min(13.5, 0.65 * len(labels) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    values = table["_search_time"]
    ax.bar(np.arange(len(labels)), values, color="#4c78a8")
    ax.set_xlabel("Method")
    ax.set_ylabel("Search Time (s)")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(_wrap_labels(labels, width=13), rotation=35, ha="right", fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    _save_pdf(fig, dataset_dir / "figure13_search_time_comparison.pdf")


def figure14_cost(dataset_dir: Path) -> None:
    table = _read_csv(dataset_dir / "table5_computational_cost.csv")
    if table.empty:
        _placeholder(dataset_dir / "figure14_computational_cost_breakdown.pdf", "No timing table available for computational-cost figure.")
        return
    components = ["Pruning Time (s)", "Search Time (s)", "Training Time (s)", "Inference Time (s)"]
    for component in components:
        table[component] = pd.to_numeric(table.get(component, pd.Series([0] * len(table))), errors="coerce").fillna(0.0)
    table["Total"] = table[components].sum(axis=1)
    table = table.sort_values("Total", ascending=True).tail(12)
    labels = table["Method"].astype(str).tolist()
    values_matrix = table[components].to_numpy()
    scale = 60.0 if np.nanmax(values_matrix) > 120 else 1.0
    unit = "min" if scale == 60.0 else "s"
    y = np.arange(len(labels))
    left = np.zeros(len(labels))
    fig_height = max(4.6, 0.38 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    for component in components:
        values = table[component].to_numpy() / scale
        ax.barh(y, values, left=left, label=component.replace(" (s)", ""))
        left += values
    ax.set_xlabel(f"Time ({unit})")
    ax.set_ylabel("Method")
    ax.set_yticks(y)
    ax.set_yticklabels(_wrap_labels(labels, width=18), fontsize=8)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
    ax.grid(alpha=0.25, axis="x")
    _save_pdf(fig, dataset_dir / "figure14_computational_cost_breakdown.pdf")


def figure15(outputs_root: Path, save_root: Path, dataset: str = FIGURE15_DATASET) -> None:
    path = save_root / "figure15_params_dice_tradeoff.pdf"
    raw_frame = _global_metric_rows(outputs_root, save_root, include_teacher=True, canonical_static_ratio=0.5)
    frame = _mean_metric_summary(raw_frame, ["Method"])
    if frame.empty:
        _placeholder(path, "No valid mean Params/Dice rows found for Figure 15.")
        return
    frame.to_csv(save_root / "figure15_mean_params_dice.csv", index=False)
    _plot_params_dice_summary(frame, path)


def figure16(outputs_root: Path, save_root: Path, dataset: str = "cvc_clinicdb") -> None:
    path = save_root / "figure16_performance_clinicdb.pdf"
    raw_frame = _global_metric_rows(outputs_root, save_root, include_teacher=True, canonical_static_ratio=0.5)
    frame = _mean_metric_summary(raw_frame, ["Method"])
    logging.info("Figure 16 mean rows loaded: %d", len(frame))
    if frame.empty:
        _placeholder(path, "No valid mean Params/Dice rows found for Figure 16.")
        return
    frame.to_csv(save_root / "figure16_mean_params_dice.csv", index=False)
    _plot_params_dice_summary(frame, path)


def figure18_static_ratio_mean_dice(outputs_root: Path, save_root: Path) -> None:
    path = save_root / "figure18_static_ratio_mean_dice.pdf"
    raw_frame = _global_metric_rows(outputs_root, save_root, static_only=True)
    frame = _mean_metric_summary(raw_frame, ["Group", "Static Ratio"])
    frame = frame.dropna(subset=["Static Ratio"]) if not frame.empty else frame
    group_order = {"Blueprint": 0, "Middle Conv2": 1, "Full Block": 2}
    teacher_points = _teacher_ratio_zero_points(outputs_root, save_root, group_order)
    if not teacher_points.empty:
        if not frame.empty:
            ratio_values = pd.to_numeric(frame["Static Ratio"], errors="coerce")
            teacher_groups = frame["Group"].isin(group_order)
            ratio_zero = ratio_values.sub(0.0).abs().lt(1e-9)
            frame = frame[~(teacher_groups & ratio_zero)].copy()
        frame = pd.concat([teacher_points, frame], ignore_index=True)
        frame = (
            frame.assign(_group_order=frame["Group"].map(group_order).fillna(99))
            .sort_values(["_group_order", "Static Ratio"])
            .drop(columns="_group_order")
            .reset_index(drop=True)
        )
    if frame.empty:
        _placeholder(path, "No valid static-ratio Dice rows found for Figure 18.")
        return
    frame.to_csv(save_root / "figure18_static_ratio_mean_dice.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = {
        "Blueprint": "#4c78a8",
        "Middle Conv2": "#f58518",
        "Full Block": "#54a24b",
    }
    for group, group_frame in frame.groupby("Group", sort=False):
        group_frame = group_frame.sort_values("Static Ratio")
        ax.plot(
            group_frame["Static Ratio"],
            group_frame["Mean Dice"],
            marker="o",
            linewidth=1.9,
            label=str(group),
            color=colors.get(str(group)),
        )
    ax.set_xlabel("Static pruning ratio")
    ax.set_ylabel("Mean Dice")
    ax.set_xticks(sorted(frame["Static Ratio"].dropna().unique()))
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9)
    _save_pdf(fig, path)


def figure19_static_ratio_params_dice(outputs_root: Path, save_root: Path) -> None:
    path = save_root / "figure19_static_ratio_params_dice.pdf"
    raw_frame = _global_metric_rows(outputs_root, save_root, static_only=True)
    frame = _mean_metric_summary(raw_frame, ["Group", "Static Ratio"])
    frame = frame.dropna(subset=["Static Ratio"]) if not frame.empty else frame
    if frame.empty:
        _placeholder(path, "No valid static-ratio Params/Dice rows found for Figure 19.")
        return
    group_order = {"Blueprint": 0, "Middle Conv2": 1, "Full Block": 2}
    frame = (
        frame.assign(_group_order=frame["Group"].map(group_order).fillna(99))
        .sort_values(["_group_order", "Static Ratio"])
        .drop(columns="_group_order")
        .reset_index(drop=True)
    )
    frame["Label"] = frame.apply(lambda row: f"{row['Group']} r={_fmt_ratio(row['Static Ratio'])}", axis=1)
    frame.to_csv(save_root / "figure19_static_ratio_params_dice.csv", index=False)
    _plot_params_dice_summary(frame, path, label_column="Label")


def _classify_train_log_run(path: Path, frame: pd.DataFrame) -> str | None:
    path_text = str(path).replace("\\", "/").lower()
    if "loss_seg_only" in path_text or "no_kd" in path_text:
        return "Seg only"
    if "loss_seg_kd" in path_text:
        return "Seg + KD"
    for column in ("use_kd_output", "config_use_kd_output"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                return "Seg + KD" if int(values.iloc[-1]) == 1 else "Seg only"
    return None


def _train_log_curve_from_path(path: Path) -> tuple[str | None, pd.DataFrame]:
    frame = _read_csv(path)
    if frame.empty:
        return None, frame
    if "phase" in frame.columns:
        student = frame[frame["phase"].fillna("").astype(str).str.lower().eq("student")]
        if not student.empty:
            frame = student
    if "checkpoint_name" in frame.columns:
        last_rows = frame[frame["checkpoint_name"].fillna("").astype(str).eq("last.pth")]
        if not last_rows.empty:
            frame = last_rows
    dice_col = "val_macro_dice" if "val_macro_dice" in frame.columns else "val_dice" if "val_dice" in frame.columns else ""
    if not dice_col:
        return None, pd.DataFrame()
    frame = frame.copy()
    frame["_epoch"] = pd.to_numeric(frame.get("epoch", pd.Series(range(1, len(frame) + 1))), errors="coerce")
    frame["_dice"] = pd.to_numeric(frame[dice_col], errors="coerce")
    frame = frame.dropna(subset=["_epoch", "_dice"]).sort_values("_epoch")
    frame = frame.drop_duplicates(subset=["_epoch"], keep="last")
    return _classify_train_log_run(path, frame), frame


def figure17_kd_vs_seg(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    path = dataset_dir / "figure17_kd_vs_seg_validation_dice.pdf"
    curves: Dict[str, pd.DataFrame] = {}
    warnings = []
    for label, loss_tag in (("Seg only", "loss_seg_only"), ("Seg + KD", PGD_LOSS_TAG)):
        curve_path, frame = _select_training_curve(outputs_root, dataset, loss_tag)
        if frame.empty or "_dice" not in frame.columns:
            logging.warning("Figure 17 missing curve for %s/%s under %s", dataset, label, loss_tag)
            continue
        logging.info("Figure 17 %s curve for %s uses %s", label, dataset, curve_path)
        curves[label] = frame

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for label, color in (("Seg only", "#4c78a8"), ("Seg + KD", "#f58518")):
        frame = curves.get(label)
        if frame is None or frame.empty:
            warnings.append(f"Missing curve: {label}")
            logging.warning("Figure 17 missing curve for %s/%s", dataset, label)
            continue
        ax.plot(frame["_epoch"], frame["_dice"], marker="o", markersize=3, linewidth=1.8, label=label, color=color)
    if not curves:
        plt.close(fig)
        _placeholder(path, f"No usable train_log.csv validation Dice curves found for {dataset}.")
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Dice")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9)
    if warnings:
        ax.text(0.01, 0.02, "; ".join(warnings), transform=ax.transAxes, fontsize=7, color="#8a4b00")
    _save_pdf(fig, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready PDF figures from outputs/statistics tables.")
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument("--save-root", type=str, default="statistics/outputs")
    parser.add_argument("--failure-topk", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    outputs_root = Path(args.outputs_root)
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    logging.info("Figure 1 is disabled by request; remaining figures will be generated normally.")
    _run_figure(
        "figure4_block_pruning_strategy",
        save_root / "paper_figures" / "figure4_block_pruning_strategy.pdf",
        figure4,
        save_root,
    )

    for dataset in _datasets(outputs_root, save_root):
        dataset_dir = save_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Processing dataset figures: %s -> %s", dataset, dataset_dir)
        _run_figure("figure2_importance_distribution", dataset_dir / "figure2_importance_distribution.pdf", figure2_importance, outputs_root, dataset, dataset_dir)
        _run_figure("figure3_thresholding_methods", dataset_dir / "figure3_thresholding_methods.pdf", figure3_thresholds, outputs_root, dataset, dataset_dir)
        _run_figure("figure5_layerwise_pruning_ratio", dataset_dir / "figure5_layerwise_pruning_ratio.pdf", figure5_layerwise, outputs_root, dataset, dataset_dir)
        _run_figure("figure6_accuracy_efficiency_tradeoff", dataset_dir / "figure6_accuracy_efficiency_tradeoff.pdf", figure6_tradeoff, outputs_root, dataset, dataset_dir)
        _run_figure("figure7_training_curve", dataset_dir / "figure7_training_curve.pdf", figure7_training, outputs_root, dataset, dataset_dir)
        _run_figure("figure8_visual_comparison", dataset_dir / "figure8_visual_comparison.pdf", figure8_visual, outputs_root, dataset, dataset_dir)
        _run_figure("figure9_failure_cases", dataset_dir / "figure9_failure_cases.pdf", figure9_failures, outputs_root, dataset, dataset_dir, topk=args.failure_topk)
        _run_figure("figure10_boundary_comparison", dataset_dir / "figure10_boundary_comparison.pdf", figure10_boundary, outputs_root, dataset, dataset_dir)
        _run_figure("figure11_12_channel_analysis", dataset_dir / "figure11/figure12 channel-analysis PDFs", figure11_12, outputs_root, dataset, dataset_dir)
        _run_figure("figure13_search_time_comparison", dataset_dir / "figure13_search_time_comparison.pdf", figure13_search, dataset_dir)
        _run_figure("figure14_computational_cost_breakdown", dataset_dir / "figure14_computational_cost_breakdown.pdf", figure14_cost, dataset_dir)
        _run_figure("figure17_kd_vs_seg_validation_dice", dataset_dir / "figure17_kd_vs_seg_validation_dice.pdf", figure17_kd_vs_seg, outputs_root, dataset, dataset_dir)

    _run_figure("figure15_params_dice_tradeoff", save_root / "figure15_params_dice_tradeoff.pdf", figure15, outputs_root, save_root)
    _run_figure("figure16_performance_clinicdb", save_root / "figure16_performance_clinicdb.pdf", figure16, outputs_root, save_root)
    _run_figure("figure18_static_ratio_mean_dice", save_root / "figure18_static_ratio_mean_dice.pdf", figure18_static_ratio_mean_dice, outputs_root, save_root)
    _run_figure("figure19_static_ratio_params_dice", save_root / "figure19_static_ratio_params_dice.pdf", figure19_static_ratio_params_dice, outputs_root, save_root)
    _run_figure("figure23_threshold_selection_methods", save_root / "figure23_threshold_selection_methods.pdf", figure23_threshold_selection, outputs_root, save_root)
    _run_figure("figure20_22_prediction_galleries", save_root / "figure20_22_prediction_galleries.pdf", figure_prediction_galleries, outputs_root, save_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
