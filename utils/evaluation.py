from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from utils.experiment import project_relative_path
from utils.val_2d import test_single_volume
from utils.visualization import save_triplet_visualization


METRIC_NAMES = ("dice", "iou", "hd95")


def sanitize_tag(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._-") or "unknown"


def checkpoint_label(checkpoint_path: Path | str) -> str:
    return sanitize_tag(Path(checkpoint_path).stem)


def build_evaluation_output_dir(
    snapshot_path: Path | str,
    dataset_name: str,
    model_name: str,
    checkpoint_path: Path | str,
    split: str,
) -> Path:
    return Path(snapshot_path) / "evaluations" / sanitize_tag(split)


def _normalize_case_name(sample: Dict) -> str:
    case_name = sample.get("case", "unknown")
    if isinstance(case_name, (list, tuple)):
        case_name = case_name[0]
    return str(case_name)


def _sample_value(sample: Dict, key: str, default: str = "NaN") -> str:
    value = sample.get(key, default)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else default
    return str(value)


def evaluate_segmentation_dataset(
    model,
    dataloader,
    *,
    device,
    num_classes: int,
    patch_size: Sequence[int],
    output_dir: Optional[Path | str] = None,
    save_visualizations: bool = False,
    vis_limit: int = 0,
    sample_validator: Optional[Callable[[torch.Tensor, int, Dict], None]] = None,
    progress_desc: Optional[str] = None,
) -> Dict:
    output_dir = Path(output_dir) if output_dir is not None else None
    model.eval()
    total_metric: List[np.ndarray] = []
    case_metrics: List[Dict] = []
    saved_visualizations = 0
    visualization_samples: List[Dict] = []

    iterator: Iterable = dataloader
    if progress_desc:
        iterator = tqdm(dataloader, desc=progress_desc, leave=False)

    with torch.no_grad():
        for sample in iterator:
            if sample_validator is not None:
                sample_validator(sample["label"], num_classes, sample)

            sample_start = time.perf_counter()
            metric_i, prediction = test_single_volume(
                sample["image"],
                sample["label"],
                model,
                classes=num_classes,
                patch_size=patch_size,
                device=device,
                return_prediction=True,
            )
            sample_elapsed = time.perf_counter() - sample_start
            metric_array = np.array(metric_i, dtype=np.float64)
            total_metric.append(metric_array)

            case_name = _normalize_case_name(sample)
            image_path = _sample_value(sample, "image_path")
            mask_path = _sample_value(sample, "label_path")

            should_save = bool(save_visualizations) and output_dir is not None and (vis_limit < 0 or saved_visualizations < vis_limit)
            prediction_path = "NaN"
            if should_save:
                save_triplet_visualization(
                    image=sample["image"][0],
                    label=sample["label"][0],
                    prediction=prediction[0],
                    output_dir=output_dir,
                    case_name=case_name,
                )
                prediction_path = str((output_dir / "pred" / f"{case_name}.png").resolve())
                per_case_mean = metric_array.mean(axis=0)
                visualization_samples.append(
                    {
                        "case": case_name,
                        "image": sample["image"][0].detach().cpu(),
                        "label": sample["label"][0].detach().cpu(),
                        "prediction": prediction[0].detach().cpu(),
                        "dice": float(per_case_mean[0]),
                        "iou": float(per_case_mean[1]),
                        "hd95": float(per_case_mean[2]),
                    }
                )
                saved_visualizations += 1

            for class_index, metric_row in enumerate(metric_array, start=1):
                case_metrics.append(
                    {
                        "case": case_name,
                        "sample_id": case_name,
                        "class_index": class_index,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "prediction_path": prediction_path,
                        "inference_time_s": float(sample_elapsed),
                        "dice": float(metric_row[0]),
                        "iou": float(metric_row[1]),
                        "hd95": float(metric_row[2]),
                    }
                )

    if not total_metric:
        raise ValueError("Evaluation dataset is empty; cannot compute metrics.")

    average_metric = np.stack(total_metric, axis=0).mean(axis=0)
    return {
        "average_metric": average_metric,
        "case_metrics": case_metrics,
        "num_cases": len({row["case"] for row in case_metrics}),
        "saved_visualizations": saved_visualizations,
        "visualization_samples": visualization_samples,
    }


def _per_class_summary(average_metric: np.ndarray) -> List[Dict]:
    return [
        {
            "class_index": int(class_index),
            "dice": float(row[0]),
            "iou": float(row[1]),
            "hd95": float(row[2]),
        }
        for class_index, row in enumerate(average_metric, start=1)
    ]


def build_evaluation_summary(metadata: Dict, average_metric: np.ndarray, case_metrics: List[Dict]) -> Dict:
    per_class_mean = _per_class_summary(average_metric)
    macro_mean = {
        metric_name: float(np.mean([row[metric_name] for row in per_class_mean])) if per_class_mean else 0.0
        for metric_name in METRIC_NAMES
    }
    return {
        **metadata,
        "metric_names": list(METRIC_NAMES),
        "num_cases": len({row["case"] for row in case_metrics}),
        "num_case_metric_rows": len(case_metrics),
        "class_indices_evaluated": [row["class_index"] for row in per_class_mean],
        "metrics": {
            "per_class_mean": per_class_mean,
            "macro_mean": macro_mean,
        },
        "average_metric": average_metric.tolist(),
    }


def _write_case_metrics_csv(case_metrics: List[Dict], output_dir: Path) -> Path:
    csv_path = output_dir / "case_metrics.csv"
    fieldnames = []
    for row in case_metrics:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["case", "class_index", "dice", "iou", "hd95"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_metrics)
    return csv_path


def _method_metadata(metadata: Dict) -> Dict[str, object]:
    model_info = metadata.get("model_info", {}) if isinstance(metadata.get("model_info"), dict) else {}
    config = model_info.get("config", {}) if isinstance(model_info.get("config"), dict) else {}
    prune_method = (
        metadata.get("prune_method")
        or model_info.get("prune_method")
        or config.get("prune_method")
        or metadata.get("phase")
        or "NaN"
    )
    pruning_ratio = (
        metadata.get("static_prune_ratio")
        or model_info.get("static_prune_ratio")
        or config.get("static_prune_ratio")
        or metadata.get("prune_ratio")
        or model_info.get("prune_ratio")
        or config.get("prune_ratio")
        or "NaN"
    )
    loss_parts = []
    for flag_name, label in (
        ("use_kd_output", "KD"),
        ("use_feature_distill", "Feature"),
        ("use_aux_loss", "Aux"),
        ("use_sparsity", "Sparsity"),
    ):
        value = metadata.get(flag_name, model_info.get(flag_name, config.get(flag_name)))
        try:
            if int(value):
                loss_parts.append(label)
        except (TypeError, ValueError):
            continue
    return {
        "method": metadata.get("method") or model_info.get("student_name") or model_info.get("model_name") or metadata.get("model") or prune_method,
        "pruning_type": prune_method,
        "pruning_ratio": pruning_ratio,
        "threshold_method": prune_method,
        "loss_method": " + ".join(loss_parts) if loss_parts else "NaN",
    }


def _write_sample_metrics_csv(case_metrics: List[Dict], output_dir: Path, metadata: Dict) -> Path:
    grouped: Dict[str, List[Dict]] = {}
    for row in case_metrics:
        grouped.setdefault(str(row.get("sample_id", row.get("case", "unknown"))), []).append(row)

    method_info = _method_metadata(metadata)
    rows = []
    for sample_id, sample_rows in grouped.items():
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": sample_rows[0].get("image_path", "NaN"),
                "mask_path": sample_rows[0].get("mask_path", "NaN"),
                "prediction_path": sample_rows[0].get("prediction_path", "NaN"),
                "dataset": metadata.get("dataset", "NaN"),
                "model": metadata.get("model", metadata.get("architecture", "NaN")),
                "method": method_info["method"],
                "pruning_type": method_info["pruning_type"],
                "pruning_ratio": method_info["pruning_ratio"],
                "threshold_method": method_info["threshold_method"],
                "loss_method": method_info["loss_method"],
                "dice": float(np.mean([float(row.get("dice", np.nan)) for row in sample_rows])),
                "iou": float(np.mean([float(row.get("iou", np.nan)) for row in sample_rows])),
                "hd95": float(np.mean([float(row.get("hd95", np.nan)) for row in sample_rows])),
                "inference_time_s": float(np.mean([float(row.get("inference_time_s", np.nan)) for row in sample_rows])),
                "split": metadata.get("split", "NaN"),
            }
        )

    csv_path = output_dir / "sample_metrics.csv"
    fieldnames = [
        "sample_id",
        "image_path",
        "mask_path",
        "prediction_path",
        "dataset",
        "model",
        "method",
        "pruning_type",
        "pruning_ratio",
        "threshold_method",
        "loss_method",
        "dice",
        "iou",
        "hd95",
        "inference_time_s",
        "split",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _write_metrics_summary_csv(summary: Dict, output_dir: Path) -> Path:
    macro = summary["metrics"]["macro_mean"]
    row = {
        "dataset": summary.get("dataset"),
        "split": summary.get("split"),
        "model": summary.get("model"),
        "phase": summary.get("phase"),
        "dice": macro.get("dice"),
        "iou": macro.get("iou"),
        "hd95": macro.get("hd95"),
        "num_cases": summary.get("num_cases"),
        "checkpoint_name": summary.get("checkpoint_name"),
        "checkpoint_path": summary.get("checkpoint_path"),
    }
    csv_path = output_dir / "metrics_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return csv_path


def _summary_to_markdown(summary: Dict) -> str:
    lines = [
        f"# Evaluation Summary | {summary['dataset']} | {summary['split']}",
        "",
        f"- Experiment: `{summary['experiment']}`",
        f"- Dataset: `{summary['dataset']}`",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Split: `{summary['split']}`",
        f"- Model: `{summary['model']}`",
        f"- Checkpoint: `{summary['checkpoint_name']}`",
        f"- Checkpoint path: `{summary['checkpoint_path']}`",
        f"- Num cases: `{summary['num_cases']}`",
        "",
        "## Metrics",
        "",
    ]
    for row in summary["metrics"]["per_class_mean"]:
        lines.append(
            f"- Class `{row['class_index']}` | dice `{row['dice']:.6f}` | iou `{row['iou']:.6f}` | hd95 `{row['hd95']:.6f}`"
        )
    lines.extend(
        [
            "",
            "## Macro Mean",
            "",
            f"- Dice: `{summary['metrics']['macro_mean']['dice']:.6f}`",
            f"- IoU: `{summary['metrics']['macro_mean']['iou']:.6f}`",
            f"- HD95: `{summary['metrics']['macro_mean']['hd95']:.6f}`",
            "",
        ]
    )
    return "\n".join(lines)


def save_evaluation_artifacts(
    output_dir: Path | str,
    metadata: Dict,
    average_metric: np.ndarray,
    case_metrics: List[Dict],
    *,
    project_root: Path | str | None = None,
) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = _write_case_metrics_csv(case_metrics, output_dir)
    sample_metrics_path = _write_sample_metrics_csv(case_metrics, output_dir, metadata)
    summary = build_evaluation_summary(metadata, average_metric, case_metrics)
    summary["case_metrics_file"] = (
        project_relative_path(csv_path, project_root)
        if project_root is not None
        else str(csv_path)
    )
    summary["sample_metrics_file"] = (
        project_relative_path(sample_metrics_path, project_root)
        if project_root is not None
        else str(sample_metrics_path)
    )
    summary["evaluation_dir"] = (
        project_relative_path(output_dir, project_root)
        if project_root is not None
        else str(output_dir)
    )

    metrics_summary_csv_path = _write_metrics_summary_csv(summary, output_dir)
    summary["metrics_summary_csv"] = (
        project_relative_path(metrics_summary_csv_path, project_root)
        if project_root is not None
        else str(metrics_summary_csv_path)
    )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    legacy_summary_path = output_dir / "metrics_summary.json"
    with legacy_summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    markdown_path = output_dir / "summary.md"
    with markdown_path.open("w", encoding="utf-8") as file:
        file.write(_summary_to_markdown(summary))

    return summary
