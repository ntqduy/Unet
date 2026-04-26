from __future__ import annotations

import argparse
import logging
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


def _save_pdf(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _placeholder(path: Path, message: str) -> None:
    logging.warning("%s", message)
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save_pdf(fig, path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as error:
        logging.warning("Cannot read CSV %s | %s", path, error)
        return pd.DataFrame()


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


def _datasets(outputs_root: Path, save_root: Path) -> List[str]:
    names = {path.name for path in save_root.iterdir() if path.is_dir() and path.name != "paper_figures"} if save_root.exists() else set()
    if outputs_root.exists():
        for csv_path in outputs_root.rglob("*.csv"):
            names.add(_dataset_from_path(csv_path, outputs_root))
    return sorted(name for name in names if name and name != "unknown")


def _find_dataset_files(outputs_root: Path, dataset: str, filename: str) -> List[Path]:
    if not outputs_root.exists():
        return []
    return [path for path in outputs_root.rglob(filename) if dataset in path.parts]


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
    frame = _concat_csvs(_find_dataset_files(outputs_root, dataset, "channel_level_detail.csv") + _find_dataset_files(outputs_root, dataset, "channel_importance.csv"))
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
    layer = str(details["layer_name"].dropna().astype(str).iloc[0])
    values = pd.to_numeric(details[details["layer_name"].astype(str).eq(layer)]["importance"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(values, bins=35, alpha=0.65, color="#9ecae1")
    if not summaries.empty and "pruning_threshold" in summaries.columns:
        for method, group in summaries.groupby(summaries.get("prune_method", pd.Series(["Method"] * len(summaries))).astype(str)):
            threshold = pd.to_numeric(group["pruning_threshold"], errors="coerce").dropna()
            if not threshold.empty:
                ax.axvline(float(threshold.iloc[0]), linestyle="--", linewidth=1.2, label=method)
    ax.set_xlabel("Channel importance")
    ax.set_ylabel("Number of channels")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    _save_pdf(fig, path)


def figure5_layerwise(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    path = dataset_dir / "figure5_layerwise_pruning_ratio.pdf"
    frame = _concat_csvs(_find_dataset_files(outputs_root, dataset, "pruning_summary.csv"))
    if frame.empty or "actual_prune_ratio" not in frame.columns:
        _placeholder(path, f"No pruning summary rows found for {dataset}.")
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for method, group in frame.groupby(frame.get("prune_method", pd.Series(["Method"] * len(frame))).astype(str)):
        ratios = pd.to_numeric(group["actual_prune_ratio"], errors="coerce").dropna().to_numpy()
        ax.plot(np.arange(1, len(ratios) + 1), ratios, marker="o", label=method)
    ax.set_xlabel("Layer index (1, 2, 3, ...)")
    ax.set_ylabel("Pruning ratio")
    ax.set_xticks([])
    ax.grid(alpha=0.25)
    ax.legend()
    _save_pdf(fig, path)


def figure6_tradeoff(dataset_dir: Path) -> None:
    path = dataset_dir / "figure6_accuracy_efficiency_tradeoff.pdf"
    frames = [_read_csv(dataset_dir / name) for name in ("table1_baseline.csv", "table2_pruning.csv", "table3_loss.csv", "table4_ablation.csv")]
    frames = [frame.rename(columns={"Model": "Method", "Component": "Method"}) for frame in frames if not frame.empty]
    if not frames:
        _placeholder(path, "No table metrics available for trade-off figure.")
        return
    frame = pd.concat(frames, ignore_index=True)
    x_col = "FLOPs" if "FLOPs" in frame.columns and pd.to_numeric(frame["FLOPs"], errors="coerce").notna().any() else "Params"
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for _, row in frame.iterrows():
        x = pd.to_numeric(pd.Series([row.get(x_col)]), errors="coerce").iloc[0]
        y = pd.to_numeric(pd.Series([row.get("Dice")]), errors="coerce").iloc[0]
        if pd.notna(x) and pd.notna(y):
            ax.scatter(x, y)
            ax.annotate(str(row.get("Method", ""))[:18], (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Dice")
    ax.grid(alpha=0.25)
    _save_pdf(fig, path)


def figure7_training(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    frame = _concat_csvs(_find_dataset_files(outputs_root, dataset, "student_epoch_diagnostics.csv") + _find_dataset_files(outputs_root, dataset, "train_log.csv"))
    if frame.empty:
        _placeholder(dataset_dir / "figure7_training_curve.pdf", f"No training log found for {dataset}.")
        return
    epoch = pd.to_numeric(frame.get("epoch", pd.Series(range(1, len(frame) + 1))), errors="coerce")
    if "train_total_loss" in frame.columns:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(epoch, pd.to_numeric(frame["train_total_loss"], errors="coerce"), label="Training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.grid(alpha=0.25)
        ax.legend()
        _save_pdf(fig, dataset_dir / "figure7a_training_loss_curve.pdf")
    if "val_macro_dice" in frame.columns:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(epoch, pd.to_numeric(frame["val_macro_dice"], errors="coerce"), label="Validation Dice")
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
    gt = _read_image(row.get("mask_path"))
    pred = _read_image(row.get("prediction_path"))
    if image is None or gt is None or pred is None:
        logging.warning("Skip figure10 for %s because boundary masks cannot be loaded.", dataset)
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.imshow(image, cmap="gray")
    ax.contour(np.squeeze(gt), levels=[0.5], colors="lime", linewidths=1)
    ax.contour(np.squeeze(pred), levels=[0.5], colors="red", linewidths=1)
    ax.axis("off")
    _save_pdf(fig, dataset_dir / "figure10_boundary_comparison.pdf")


def figure11_12(outputs_root: Path, dataset: str, dataset_dir: Path) -> None:
    frame = _concat_csvs(_find_dataset_files(outputs_root, dataset, "student_final_channel_summary.csv") + _find_dataset_files(outputs_root, dataset, "channel_summary.csv"))
    if frame.empty:
        _placeholder(dataset_dir / "figure11_output_channels_per_layer_pruned_student.pdf", f"No channel summary found for {dataset}.")
        _placeholder(dataset_dir / "figure12_mean_channel_importance_per_layer_pruned_student.pdf", f"No channel summary found for {dataset}.")
        return
    x = np.arange(1, len(frame) + 1)
    for filename, column, ylabel in (
        ("figure11_output_channels_per_layer_pruned_student.pdf", "out_channels", "Output channels"),
        ("figure12_mean_channel_importance_per_layer_pruned_student.pdf", "importance_mean", "Mean channel importance"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(x, pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else np.zeros(len(frame)))
        ax.set_xlabel("Layer index (1, 2, 3, ...)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([])
        ax.grid(alpha=0.25, axis="y")
        _save_pdf(fig, dataset_dir / filename)


def figure13_search(dataset_dir: Path) -> None:
    table = _read_csv(dataset_dir / "table2_pruning.csv")
    if table.empty:
        _placeholder(dataset_dir / "figure13_search_time_comparison.pdf", "No pruning table available for search-time figure.")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = table["Method"].astype(str).tolist()
    ax.bar(np.arange(len(labels)), pd.to_numeric(table["Search Time (s)"], errors="coerce").fillna(0.0))
    ax.set_xlabel("Method")
    ax.set_ylabel("Search Time (s)")
    if len(labels) > 6:
        ax.set_xticks([])
    else:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(alpha=0.25, axis="y")
    _save_pdf(fig, dataset_dir / "figure13_search_time_comparison.pdf")


def figure14_cost(dataset_dir: Path) -> None:
    table = _read_csv(dataset_dir / "table5_computational_cost.csv")
    if table.empty:
        _placeholder(dataset_dir / "figure14_computational_cost_breakdown.pdf", "No timing table available for computational-cost figure.")
        return
    labels = table["Method"].astype(str).tolist()
    components = ["Pruning Time (s)", "Search Time (s)", "Training Time (s)", "Inference Time (s)"]
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for component in components:
        values = pd.to_numeric(table.get(component, pd.Series([0] * len(table))), errors="coerce").fillna(0.0).to_numpy()
        ax.bar(x, values, bottom=bottom, label=component.replace(" (s)", ""))
        bottom += values
    ax.set_xlabel("Method")
    ax.set_ylabel("Time (s)")
    ax.set_xticks([] if len(labels) > 6 else x)
    if len(labels) <= 6:
        ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    _save_pdf(fig, dataset_dir / "figure14_computational_cost_breakdown.pdf")


def figure15(save_root: Path) -> None:
    table = _read_csv(save_root / "table_mean_std_across_datasets.csv")
    path = save_root / "figure15_mean_performance_across_datasets.pdf"
    if table.empty:
        _placeholder(path, "No mean/std table available.")
        return
    labels = table["Method"].astype(str).tolist()
    mean = pd.to_numeric(table["Mean Dice"], errors="coerce")
    std = pd.to_numeric(table["Std Dice"], errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(np.arange(len(labels)), mean, yerr=std, capsize=3)
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean Dice")
    if len(labels) > 8:
        ax.set_xticks([])
    else:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(alpha=0.25, axis="y")
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

    # Figure 1 is intentionally not exported by default because the pipeline
    # diagram is prepared manually. Other paper figures are generated as usual.
    figure4(save_root)

    for dataset in _datasets(outputs_root, save_root):
        dataset_dir = save_root / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        figure2_importance(outputs_root, dataset, dataset_dir)
        figure3_thresholds(outputs_root, dataset, dataset_dir)
        figure5_layerwise(outputs_root, dataset, dataset_dir)
        figure6_tradeoff(dataset_dir)
        figure7_training(outputs_root, dataset, dataset_dir)
        figure8_visual(outputs_root, dataset, dataset_dir)
        figure9_failures(outputs_root, dataset, dataset_dir, topk=args.failure_topk)
        figure10_boundary(outputs_root, dataset, dataset_dir)
        figure11_12(outputs_root, dataset, dataset_dir)
        figure13_search(dataset_dir)
        figure14_cost(dataset_dir)

    figure15(save_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
