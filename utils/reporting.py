from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torchvision.utils import save_image

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
        if not samples:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.axis("off")
            ax.text(0.5, 0.5, "No visualization samples available.", ha="center", va="center")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            return pdf_path

        num_rows = len(samples)
        fig_height = max(4.8, 3.8 * num_rows)
        fig, axes = plt.subplots(num_rows, 3, figsize=(12, fig_height))
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_index, sample in enumerate(samples):
            row_axes = axes[row_index]
            case_name = sample.get("case", f"sample_{row_index}")
            row_axes[0].imshow(_normalize_image_for_plot(sample["image"]))
            row_axes[0].set_title(f"Image ({case_name})", fontsize=10, pad=6)
            row_axes[1].imshow(colorize_mask(sample["label"]).permute(1, 2, 0).numpy())
            row_axes[1].set_title("GT", fontsize=10, pad=6)
            row_axes[2].imshow(colorize_mask(sample["prediction"]).permute(1, 2, 0).numpy())
            dice_value = sample.get("dice")
            prediction_title = "PR" if dice_value is None else f"PR | Dice {dice_value:.4f}"
            row_axes[2].set_title(prediction_title, fontsize=10, pad=6)
            for axis in row_axes:
                axis.axis("off")

        fig.tight_layout(h_pad=1.0, w_pad=0.8)
        pdf.savefig(fig)
        plt.close(fig)
    return pdf_path


def save_visualization_overview_image(
    samples: Sequence[Mapping],
    image_path: Path | str,
) -> Path:
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    for sample in samples:
        image = sample["image"].detach().cpu().float()
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3]
        image_min = image.min()
        image_max = image.max()
        if float(image_max - image_min) > 1e-8:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = torch.zeros_like(image)
        label = colorize_mask(sample["label"])
        prediction = colorize_mask(sample["prediction"])
        panels.append(torch.cat([image, label, prediction], dim=2))
    if not panels:
        blank = torch.zeros(3, 64, 192)
        save_image(blank, image_path)
        return image_path
    save_image(torch.cat(panels, dim=1), image_path)
    return image_path


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
            metric_candidates = [column for column in ("dice", "iou", "hd95", "params", "fps", "inference_time_seconds", "evaluation_time_seconds") if column in headers]
            labels = [str(row.get("split", row.get("phase", f"row_{index}"))) for index, row in enumerate(rows)]

            def _numeric(value) -> float:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0

            if not metric_candidates:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.axis("off")
                ax.text(0.5, 0.5, "No plottable metrics available.", ha="center", va="center")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                return pdf_path

            for start_index in range(0, len(metric_candidates), 4):
                chunk = metric_candidates[start_index : start_index + 4]
                fig, axes = plt.subplots(2, 2, figsize=(14, 8))
                axes = axes.flatten()
                for axis, metric_name in zip(axes, chunk):
                    axis.bar(labels, [_numeric(row.get(metric_name, 0.0)) for row in rows])
                    axis.grid(alpha=0.25, axis="y")
                    axis.tick_params(axis="x", rotation=25)
                for axis in axes[len(chunk) :]:
                    axis.axis("off")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
    return pdf_path


def _save_table_page(pdf: PdfPages, rows: Sequence[Mapping], title: str, *, max_rows: int = 20) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        return

    headers = list(rows[0].keys())
    for start_index in range(0, len(rows), max_rows):
        chunk = rows[start_index : start_index + max_rows]
        values = [[row.get(column, "") for column in headers] for row in chunk]
        fig, ax = plt.subplots(figsize=(14, 4 + 0.35 * len(chunk)))
        ax.axis("off")
        table = ax.table(cellText=values, colLabels=headers, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.25)
        suffix = "" if len(rows) <= max_rows else f" ({start_index + 1}-{start_index + len(chunk)})"
        ax.set_title(f"{title}{suffix}")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _save_empty_report_page(pdf: PdfPages, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def save_channel_analysis_pdf(
    report: Mapping,
    pdf_path: Path | str,
    *,
    title: str = "Channel Analysis Report",
) -> Dict[str, Path]:
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    tables_pdf_path = pdf_path.with_name(f"{pdf_path.stem}_tables.pdf")
    charts_pdf_path = pdf_path.with_name(f"{pdf_path.stem}_charts.pdf")

    global_summary = dict(report.get("global_summary", {}))
    layer_summary_rows = report.get("layer_summary_rows", [])
    gate_summary_rows = report.get("gate_summary_rows", [])
    comparison_rows = report.get("comparison_rows", [])
    pruning_summary_rows = report.get("pruning_summary_rows", []) or report.get("teacher_vs_student_rows", [])

    with PdfPages(tables_pdf_path) as pdf:
        has_table_content = False
        global_summary = dict(report.get("global_summary", {}))
        if global_summary:
            summary_rows = [{"key": key, "value": value} for key, value in global_summary.items()]
            _save_table_page(pdf, summary_rows, f"{title} | Global Summary", max_rows=25)
            has_table_content = True

        if layer_summary_rows:
            _save_table_page(pdf, layer_summary_rows, f"{title} | Layer Summary")
            has_table_content = True
        if gate_summary_rows:
            _save_table_page(pdf, gate_summary_rows, f"{title} | Gate Summary")
            has_table_content = True
        if comparison_rows:
            _save_table_page(pdf, comparison_rows, f"{title} | Comparison")
            has_table_content = True
        if pruning_summary_rows:
            _save_table_page(pdf, pruning_summary_rows, f"{title} | Pruning Summary")
            has_table_content = True

        if not has_table_content:
            _save_empty_report_page(pdf, title, "No table content available.")

    with PdfPages(charts_pdf_path) as pdf:
        has_chart_content = False
        if layer_summary_rows:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            labels = [str(row.get("layer_name")) for row in layer_summary_rows]
            axes[0].bar(labels, [row.get("out_channels", 0) or 0 for row in layer_summary_rows])
            axes[0].set_title(f"{title} | Output Channels Per Layer")
            axes[0].grid(alpha=0.25, axis="y")
            axes[0].tick_params(axis="x", rotation=70)
            axes[1].bar(labels, [row.get("importance_mean", 0) or 0 for row in layer_summary_rows])
            axes[1].set_title(f"{title} | Mean Channel Importance Per Layer")
            axes[1].grid(alpha=0.25, axis="y")
            axes[1].tick_params(axis="x", rotation=70)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            has_chart_content = True

        if pruning_summary_rows:
            teacher_key = "teacher_out_channels" if "teacher_out_channels" in pruning_summary_rows[0] else None
            student_key = "student_out_channels" if "student_out_channels" in pruning_summary_rows[0] else None
            ratio_key = "actual_prune_ratio" if "actual_prune_ratio" in pruning_summary_rows[0] else None
            if teacher_key and student_key:
                labels = [str(row.get("layer_name")) for row in pruning_summary_rows]
                x = np.arange(len(labels))
                width = 0.35
                fig, axes = plt.subplots(2, 1, figsize=(14, 8))
                axes[0].bar(x - width / 2, [row.get(teacher_key, 0) or 0 for row in pruning_summary_rows], width=width, label="teacher")
                axes[0].bar(x + width / 2, [row.get(student_key, 0) or 0 for row in pruning_summary_rows], width=width, label="student")
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(labels, rotation=70)
                axes[0].set_title(f"{title} | Teacher vs Student Channels")
                axes[0].grid(alpha=0.25, axis="y")
                axes[0].legend()
                if ratio_key:
                    axes[1].bar(labels, [row.get(ratio_key, 0) or 0 for row in pruning_summary_rows])
                    axes[1].set_title(f"{title} | Prune Ratio Per Layer")
                    axes[1].grid(alpha=0.25, axis="y")
                    axes[1].tick_params(axis="x", rotation=70)
                else:
                    axes[1].axis("off")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                has_chart_content = True

        if not has_chart_content:
            _save_empty_report_page(pdf, title, "No chart content available.")
    return {
        "tables_pdf": tables_pdf_path,
        "charts_pdf": charts_pdf_path,
    }
