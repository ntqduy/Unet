from __future__ import annotations

import argparse
import logging
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
try:
    import matplotlib.image as mpimg
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as error:  # pragma: no cover - dependency guard
    raise SystemExit("Missing dependency: pandas/matplotlib. Install project requirements with `pip install -r requirements.txt`.") from error

PGD_TEACHER_DIR = "unet_resnet152_teacher"
PGD_LOSS_TAG = "loss_seg_kd_sparsity"
CHANNEL_METHODS = {"static", "kneedle", "otsu", "gmm"}
MIDDLE_METHODS = {"middle_static", "middle_kneedle", "middle_otsu", "middle_gmm"}
FIGURE15_DATASET = "cvc_clinicdb"
FIGURE15_METHOD_DIRS = [
    ("Teacher", Path("1_teacher")),
    ("Static r=0.5", Path(PGD_LOSS_TAG) / "output_static_0.5_no" / "3_student"),
    ("Kneedle", Path(PGD_LOSS_TAG) / "output_kneedle_auto_no" / "3_student"),
    ("Otsu", Path(PGD_LOSS_TAG) / "output_otsu_auto_no" / "3_student"),
    ("GMM", Path(PGD_LOSS_TAG) / "output_gmm_auto_no" / "3_student"),
    ("Middle Static", Path(PGD_LOSS_TAG) / "output_middle_static_0.5_no" / "3_student"),
    ("Middle Kneedle", Path(PGD_LOSS_TAG) / "output_middle_kneedle_auto_no" / "3_student"),
    ("Middle Otsu", Path(PGD_LOSS_TAG) / "output_middle_otsu_auto_no" / "3_student"),
    ("Middle GMM", Path(PGD_LOSS_TAG) / "output_middle_gmm_auto_no" / "3_student"),
]


def _safe_float(value, default: float = np.nan) -> float:
    try:
        if value in ("", None):
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
    }
    if raw_method == "static":
        return f"Static r={_fmt_ratio(ratio)}"
    if raw_method == "middle_static":
        return f"Middle static r={_fmt_ratio(ratio)}"
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
        return pd.read_csv(path)
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
    if "dice" not in frame.columns:
        logging.warning("Figure 15 metrics CSV has no dice column: %s", path)
        return None
    scores = pd.to_numeric(frame["dice"], errors="coerce").fillna(-np.inf)
    if scores.empty or float(scores.max()) == -np.inf:
        return None
    return frame.loc[scores.idxmax()]


def _params_to_millions(value) -> float:
    number = _safe_float(value)
    if np.isnan(number):
        return number
    return number / 1e6 if abs(number) > 1e5 else number


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


def _pgd_teacher_phase_root(outputs_root: Path, dataset: str) -> Path:
    return outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR / "1_teacher"


def _is_pgd_focus_path(path: Path, outputs_root: Path, dataset: str) -> bool:
    try:
        relative = path.relative_to(_pgd_focus_root(outputs_root, dataset))
    except ValueError:
        return False
    return bool(relative.parts)


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
                if (_pgd_focus_root(outputs_root, dataset_dir.name)).exists():
                    names.add(dataset_dir.name)
        for csv_path in outputs_root.rglob("*.csv"):
            dataset = _dataset_from_path(csv_path, outputs_root)
            if dataset != "unknown" and not Path(dataset).suffix:
                names.add(dataset)
    return sorted(name for name in names if name and name != "unknown")


def _find_dataset_files(outputs_root: Path, dataset: str, filename: str) -> List[Path]:
    if not outputs_root.exists():
        return []
    focus_root = _pgd_focus_root(outputs_root, dataset)
    if focus_root.exists():
        focused = [path for path in focus_root.rglob(filename)]
        if focused:
            logging.info("Found %d target PGD files for %s/%s under %s", len(focused), dataset, filename, focus_root)
            return focused
    fallback = [path for path in outputs_root.rglob(filename) if dataset in path.parts and not Path(dataset).suffix]
    if fallback:
        logging.warning("Using fallback files for %s/%s because target PGD path has no matches.", dataset, filename)
    return fallback


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
                    ax.axvline(value, linewidth=1.7, label=_display_method(method), **threshold_styles[method])
                else:
                    logging.info("Skip out-of-range threshold line for %s/%s: %.6f", dataset, method, value)
    ax.set_xlim(x_low, x_high)
    ax.set_xlabel("Channel importance")
    ax.set_ylabel("Number of channels")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
    _save_pdf(fig, path)


def figure5_layerwise(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    path = dataset_dir / "figure5_layerwise_pruning_ratio.pdf"
    frame = _concat_csvs(_find_dataset_files(outputs_root, dataset, "pruning_summary.csv"))
    if frame.empty or "actual_prune_ratio" not in frame.columns:
        _placeholder(path, f"No pruning summary rows found for {dataset}.")
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    method_series = frame.get("prune_method", pd.Series(["Method"] * len(frame))).astype(str).str.lower()
    ratio_series = frame.get("static_prune_ratio", pd.Series([np.nan] * len(frame)))
    grouped_keys = pd.DataFrame({"method": method_series, "ratio": ratio_series})
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for index, ((method, ratio), group_index) in enumerate(grouped_keys.groupby(["method", "ratio"], dropna=False).groups.items()):
        group = frame.loc[list(group_index)]
        ratios = pd.to_numeric(group["actual_prune_ratio"], errors="coerce").dropna().to_numpy()
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
    ax.set_xlabel("Layer index (1, 2, 3, ...)")
    ax.set_ylabel("Pruning ratio")
    ax.set_xticks([])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.9, ncol=1)
    _save_pdf(fig, path)


def figure6_tradeoff(dataset_dir: Path) -> None:
    path = dataset_dir / "figure6_accuracy_efficiency_tradeoff.pdf"
    frame = _read_csv(dataset_dir / "table2_pruning.csv")
    if frame.empty:
        _placeholder(path, "No table metrics available for trade-off figure.")
        return
    method_col = "Method" if "Method" in frame.columns else "Phương pháp" if "Phương pháp" in frame.columns else frame.columns[0]
    dice_col = "Dice" if "Dice" in frame.columns else "Dice $\\uparrow$"
    if dice_col not in frame.columns:
        _placeholder(path, "No Dice column available for trade-off figure.")
        return
    if "FLOPs" in frame.columns and pd.to_numeric(frame["FLOPs"], errors="coerce").notna().any():
        x_col = "FLOPs"
    elif "Params" in frame.columns:
        x_col = "Params"
    else:
        x_col = "Params (M)"
    if x_col not in frame.columns:
        _placeholder(path, "No Params/FLOPs column available for trade-off figure.")
        return
    x_values = pd.to_numeric(frame[x_col], errors="coerce")
    scale = 1e6 if x_col != "Params (M)" and x_values.max(skipna=True) > 1e5 else 1.0
    xlabel = f"{x_col} (M)" if scale == 1e6 else x_col
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
        frame[method_col].astype(str).str.contains("Teacher|Giáo viên|GMM|Middle", regex=True, case=False, na=False)
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

    teacher = frame[frame[method_col].astype(str).str.contains("Teacher|Giáo viên", regex=True, case=False, na=False)]
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
    ax.set_ylabel("Dice")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.9)
    _save_pdf(fig, path)


def figure7_training(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    frame = _concat_csvs(_find_dataset_files(outputs_root, dataset, "student_epoch_diagnostics.csv") + _find_dataset_files(outputs_root, dataset, "train_log.csv"))
    if frame.empty:
        _placeholder(dataset_dir / "figure7_training_curve.pdf", f"No training log found for {dataset}.")
        return
    epoch = pd.to_numeric(frame.get("epoch", pd.Series(range(1, len(frame) + 1))), errors="coerce")
    if "train_total_loss" in frame.columns:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(epoch, _smooth(frame["train_total_loss"]), label="Training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.grid(alpha=0.25)
        ax.legend()
        _save_pdf(fig, dataset_dir / "figure7a_training_loss_curve.pdf")
    if "val_macro_dice" in frame.columns:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(epoch, _smooth(frame["val_macro_dice"]), label="Validation Dice")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Dice")
        ax.grid(alpha=0.25)
        ax.legend()
        _save_pdf(fig, dataset_dir / "figure7b_validation_dice_curve.pdf")


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
    panels = [("Input", _read_image(row.get("image_path"))), ("Ground Truth", _read_image(row.get("mask_path")))]
    for _, sample_row in sample_rows.iterrows():
        panels.append((str(sample_row.get("method", "Prediction"))[:18], _read_image(sample_row.get("prediction_path"))))
    panels = [(label, image) for label, image in panels if image is not None]
    if len(panels) < 3:
        logging.warning("Skip figure8 for %s because images cannot be loaded.", dataset)
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
    for ax, (_, image) in zip(np.ravel(axes), panels):
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
    if output_dir.name.startswith("output_middle_") and len(parts) >= 3:
        method = f"middle_{parts[2]}"
        ratio = _safe_float(parts[3]) if len(parts) >= 5 and parts[3] != "auto" else np.nan
        return method, ratio
    if output_dir.name.startswith("output_") and len(parts) >= 2:
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
    focus_root = _pgd_focus_root(outputs_root, dataset)
    if not focus_root.exists():
        logging.warning("PGD focus root is missing for %s: %s", dataset, focus_root)
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
    }
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
        logging.warning("No student channel-analysis candidates found under %s", focus_root)
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
    table = _read_csv(dataset_dir / "table2_pruning.csv")
    if table.empty:
        _placeholder(dataset_dir / "figure13_search_time_comparison.pdf", "No pruning table available for search-time figure.")
        return
    table = table[~table.get("Group", pd.Series([""] * len(table))).astype(str).eq("Reference")].copy()
    if table.empty:
        _placeholder(dataset_dir / "figure13_search_time_comparison.pdf", "No pruning rows available for search-time figure.")
        return
    labels = table["Method"].astype(str).tolist()
    fig_width = max(7.5, min(13.5, 0.65 * len(labels) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    values = pd.to_numeric(table["Search Time (s)"], errors="coerce").fillna(0.0)
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
    base_root = outputs_root / "pgd_unet" / dataset / PGD_TEACHER_DIR
    rows = []
    for label, relative_dir in FIGURE15_METHOD_DIRS:
        row = _best_test_row(base_root / relative_dir / "metrics_summary.csv")
        if row is None:
            continue
        rows.append(
            {
                "Method": label,
                "Params (M)": _params_to_millions(row.get("params")),
                "Dice": _safe_float(row.get("dice")),
            }
        )
    frame = pd.DataFrame(rows, columns=["Method", "Params (M)", "Dice"]).dropna(subset=["Params (M)", "Dice"])
    if frame.empty:
        _placeholder(path, f"No valid Params/Dice rows found for Figure 15 ({dataset}).")
        return
    frame = frame.sort_values(["Params (M)", "Dice"], ascending=[True, False]).reset_index(drop=True)

    x = frame["Params (M)"].to_numpy(dtype=float)
    labels = [f"{params:.2f}\n({method})" for params, method in zip(frame["Params (M)"], frame["Method"].astype(str))]
    fig_width = max(8.5, min(13.5, 0.72 * len(labels) + 4.0))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.plot(x, frame["Dice"], marker="o", linewidth=1.9, color="#4c78a8")
    ax.set_xlabel("Params (M) (method)")
    ax.set_ylabel("Dice")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.grid(alpha=0.25)
    for params, dice in zip(frame["Params (M)"], frame["Dice"]):
        ax.annotate(f"{dice:.4f}", (params, dice), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
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
        _run_figure("figure6_accuracy_efficiency_tradeoff", dataset_dir / "figure6_accuracy_efficiency_tradeoff.pdf", figure6_tradeoff, dataset_dir)
        _run_figure("figure7_training_curve", dataset_dir / "figure7_training_curve.pdf", figure7_training, outputs_root, dataset, dataset_dir)
        _run_figure("figure8_visual_comparison", dataset_dir / "figure8_visual_comparison.pdf", figure8_visual, outputs_root, dataset, dataset_dir)
        _run_figure("figure9_failure_cases", dataset_dir / "figure9_failure_cases.pdf", figure9_failures, outputs_root, dataset, dataset_dir, topk=args.failure_topk)
        _run_figure("figure10_boundary_comparison", dataset_dir / "figure10_boundary_comparison.pdf", figure10_boundary, outputs_root, dataset, dataset_dir)
        _run_figure("figure11_12_channel_analysis", dataset_dir / "figure11/figure12 channel-analysis PDFs", figure11_12, outputs_root, dataset, dataset_dir)
        _run_figure("figure13_search_time_comparison", dataset_dir / "figure13_search_time_comparison.pdf", figure13_search, dataset_dir)
        _run_figure("figure14_computational_cost_breakdown", dataset_dir / "figure14_computational_cost_breakdown.pdf", figure14_cost, dataset_dir)

    _run_figure("figure15_params_dice_tradeoff", save_root / "figure15_params_dice_tradeoff.pdf", figure15, outputs_root, save_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
