from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from utils.visualization import colorize_mask


def _normalize_image_for_plot(image) -> np.ndarray:
    image = image.detach().cpu().float()
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3]
    image = image.permute(1, 2, 0).numpy()
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max - image_min > 1e-8:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = np.zeros_like(image)
    return image


def write_metrics_rows(rows: Iterable[Mapping], csv_path: Path | str) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("Cannot export an empty metrics CSV.")

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def save_loss_pdf(history: Mapping[str, Sequence[float]], pdf_path: Path | str, *, title: str = "Loss Curves") -> Path:
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        for series_name, values in history.items():
            if not values:
                continue
            ax.plot(range(1, len(values) + 1), values, label=series_name)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    return pdf_path


def save_visualization_pdf(samples: Sequence[Mapping], pdf_path: Path | str, *, title: str = "Segmentation Visualizations") -> Path:
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for sample in samples:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(_normalize_image_for_plot(sample["image"]))
            axes[0].set_title("Image")
            axes[1].imshow(colorize_mask(sample["label"]).permute(1, 2, 0).numpy())
            axes[1].set_title("Ground Truth")
            axes[2].imshow(colorize_mask(sample["prediction"]).permute(1, 2, 0).numpy())
            dice_value = sample.get("dice")
            axes[2].set_title("Prediction" if dice_value is None else f"Prediction | Dice {dice_value:.4f}")
            for axis in axes:
                axis.axis("off")
            case_name = sample.get("case", "unknown")
            fig.suptitle(f"{title} | {case_name}", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path


def save_performance_pdf(
    rows: Sequence[Mapping],
    pdf_path: Path | str,
    *,
    title: str = "Performance Report",
) -> Path:
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]

    with PdfPages(pdf_path) as pdf:
        if rows:
            headers = list(rows[0].keys())
            table_values = [[row.get(column, "") for column in headers] for row in rows]

            fig, ax = plt.subplots(figsize=(14, 4 + 0.35 * len(rows)))
            ax.axis("off")
            table = ax.table(cellText=table_values, colLabels=headers, loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.3)
            ax.set_title(title)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            metric_candidates = [column for column in ("dice", "iou", "hd95", "params", "fps", "inference_time_seconds") if column in headers]
            if metric_candidates:
                fig, axes = plt.subplots(len(metric_candidates), 1, figsize=(10, 3 * len(metric_candidates)))
                if len(metric_candidates) == 1:
                    axes = [axes]
                labels = [str(row.get("split", row.get("phase", f"row_{index}"))) for index, row in enumerate(rows)]
                for axis, metric_name in zip(axes, metric_candidates):
                    axis.bar(labels, [row.get(metric_name, 0.0) or 0.0 for row in rows])
                    axis.set_title(metric_name)
                    axis.grid(alpha=0.25, axis="y")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    return pdf_path
