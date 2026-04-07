from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.model_output import extract_model_info
from utils.reporting import save_channel_analysis_pdf, write_metrics_rows


CHANNEL_LAYER_TYPES = (nn.Conv2d, nn.ConvTranspose2d)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    return str(value)


def _stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
        }
    return {
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _kernel_size_repr(module: nn.Module) -> Optional[List[int]]:
    kernel_size = getattr(module, "kernel_size", None)
    if kernel_size is None:
        return None
    if isinstance(kernel_size, tuple):
        return [int(value) for value in kernel_size]
    return [int(kernel_size)]


def _weight_shape_repr(module: nn.Module) -> Optional[List[int]]:
    weight = getattr(module, "weight", None)
    if weight is None:
        return None
    return [int(value) for value in weight.shape]


def _module_in_channels(module: nn.Module) -> Optional[int]:
    return int(module.in_channels) if hasattr(module, "in_channels") else None


def _module_out_channels(module: nn.Module) -> Optional[int]:
    return int(module.out_channels) if hasattr(module, "out_channels") else None


def compute_filter_l1_importance(module: nn.Module) -> torch.Tensor:
    if isinstance(module, nn.Conv2d):
        return module.weight.detach().abs().sum(dim=(1, 2, 3))
    if isinstance(module, nn.ConvTranspose2d):
        weight = module.weight.detach().abs()
        if module.groups == 1:
            return weight.sum(dim=(0, 2, 3))
        in_per_group = module.in_channels // module.groups
        out_per_group = module.out_channels // module.groups
        reshaped = weight.view(module.groups, in_per_group, out_per_group, *weight.shape[2:])
        return reshaped.sum(dim=(1, 3, 4)).reshape(-1)
    raise TypeError(f"Unsupported layer type for channel importance: {type(module)!r}")


def find_primary_channel_layer(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, CHANNEL_LAYER_TYPES):
        return module
    for child in module.modules():
        if child is module:
            continue
        if isinstance(child, CHANNEL_LAYER_TYPES):
            return child
    return None


def extract_channel_analysis(
    model,
    *,
    importance_name: str = "filter_l1",
    gate_threshold: float = 0.1,
) -> Dict[str, Any]:
    layer_summary_rows: List[Dict[str, Any]] = []
    channel_importance_rows: List[Dict[str, Any]] = []
    gate_summary_rows: List[Dict[str, Any]] = []
    gate_value_rows: List[Dict[str, Any]] = []

    for module_name, module in model.named_modules():
        if isinstance(module, CHANNEL_LAYER_TYPES):
            importance = compute_filter_l1_importance(module).detach().cpu()
            importance_values = [float(value) for value in importance.tolist()]
            ranked_indices = torch.argsort(importance, descending=True).tolist()
            ranks = {int(index): rank + 1 for rank, index in enumerate(ranked_indices)}
            stats = _stats(importance_values)
            layer_summary_rows.append(
                {
                    "layer_name": module_name,
                    "module_type": type(module).__name__,
                    "in_channels": _module_in_channels(module),
                    "out_channels": _module_out_channels(module),
                    "kernel_size": _kernel_size_repr(module),
                    "weight_shape": _weight_shape_repr(module),
                    "importance_source": importance_name,
                    "importance_min": stats["min"],
                    "importance_max": stats["max"],
                    "importance_mean": stats["mean"],
                    "importance_std": stats["std"],
                }
            )
            for channel_index, importance_value in enumerate(importance_values):
                channel_importance_rows.append(
                    {
                        "layer_name": module_name,
                        "module_type": type(module).__name__,
                        "channel_index": int(channel_index),
                        "importance": float(importance_value),
                        "rank_desc": int(ranks[channel_index]),
                        "importance_source": importance_name,
                    }
                )

        if hasattr(module, "gate_values") and callable(module.gate_values):
            gate_tensor = module.gate_values().detach().cpu().float()
            gate_values = [float(value) for value in gate_tensor.tolist()]
            ranked_indices = torch.argsort(gate_tensor, descending=True).tolist()
            ranks = {int(index): rank + 1 for rank, index in enumerate(ranked_indices)}
            stats = _stats(gate_values)
            gate_summary_rows.append(
                {
                    "layer_name": module_name,
                    "module_type": type(module).__name__,
                    "out_channels": len(gate_values),
                    "gate_min": stats["min"],
                    "gate_max": stats["max"],
                    "gate_mean": stats["mean"],
                    "gate_std": stats["std"],
                    "gate_threshold": float(gate_threshold),
                    "near_off_channels": int(sum(value < gate_threshold for value in gate_values)),
                }
            )
            for channel_index, gate_value in enumerate(gate_values):
                gate_value_rows.append(
                    {
                        "layer_name": module_name,
                        "module_type": type(module).__name__,
                        "channel_index": int(channel_index),
                        "gate_value": float(gate_value),
                        "rank_desc": int(ranks[channel_index]),
                        "near_off": int(gate_value < gate_threshold),
                    }
                )

    global_summary = {
        "num_channel_layers": int(len(layer_summary_rows)),
        "total_output_channels": int(sum(int(row["out_channels"] or 0) for row in layer_summary_rows)),
        "num_gate_layers": int(len(gate_summary_rows)),
        "total_gate_channels": int(sum(int(row["out_channels"] or 0) for row in gate_summary_rows)),
        "importance_source": importance_name,
    }

    return {
        "model_info": _to_serializable(extract_model_info(model)),
        "global_summary": global_summary,
        "layer_summary_rows": layer_summary_rows,
        "channel_importance_rows": channel_importance_rows,
        "gate_summary_rows": gate_summary_rows,
        "gate_value_rows": gate_value_rows,
    }


def build_analysis_comparison(
    before_analysis: Mapping[str, Any],
    after_analysis: Mapping[str, Any],
    *,
    before_label: str = "before",
    after_label: str = "after",
) -> List[Dict[str, Any]]:
    comparison_rows: List[Dict[str, Any]] = []

    before_layers = {row["layer_name"]: row for row in before_analysis.get("layer_summary_rows", [])}
    after_layers = {row["layer_name"]: row for row in after_analysis.get("layer_summary_rows", [])}
    for layer_name in sorted(set(before_layers) | set(after_layers)):
        before_row = before_layers.get(layer_name, {})
        after_row = after_layers.get(layer_name, {})
        comparison_rows.append(
            {
                "analysis_type": "channel",
                "layer_name": layer_name,
                "module_type": before_row.get("module_type", after_row.get("module_type")),
                f"out_channels_{before_label}": before_row.get("out_channels"),
                f"out_channels_{after_label}": after_row.get("out_channels"),
                f"importance_mean_{before_label}": before_row.get("importance_mean"),
                f"importance_mean_{after_label}": after_row.get("importance_mean"),
                f"importance_std_{before_label}": before_row.get("importance_std"),
                f"importance_std_{after_label}": after_row.get("importance_std"),
                "channel_delta": (after_row.get("out_channels") or 0) - (before_row.get("out_channels") or 0),
            }
        )

    before_gates = {row["layer_name"]: row for row in before_analysis.get("gate_summary_rows", [])}
    after_gates = {row["layer_name"]: row for row in after_analysis.get("gate_summary_rows", [])}
    for layer_name in sorted(set(before_gates) | set(after_gates)):
        before_row = before_gates.get(layer_name, {})
        after_row = after_gates.get(layer_name, {})
        comparison_rows.append(
            {
                "analysis_type": "gate",
                "layer_name": layer_name,
                "module_type": before_row.get("module_type", after_row.get("module_type")),
                f"gate_mean_{before_label}": before_row.get("gate_mean"),
                f"gate_mean_{after_label}": after_row.get("gate_mean"),
                f"gate_std_{before_label}": before_row.get("gate_std"),
                f"gate_std_{after_label}": after_row.get("gate_std"),
                f"near_off_channels_{before_label}": before_row.get("near_off_channels"),
                f"near_off_channels_{after_label}": after_row.get("near_off_channels"),
            }
        )

    return comparison_rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_serializable(payload), file, indent=2)
    return path


def save_channel_analysis_artifacts(
    output_dir: Path | str,
    analysis: Mapping[str, Any],
    *,
    prefix: Optional[str] = None,
    title: str = "Channel Analysis",
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_part = f"{prefix}_" if prefix else ""

    paths: Dict[str, Path] = {}
    layer_summary_rows = analysis.get("layer_summary_rows", [])
    channel_importance_rows = analysis.get("channel_importance_rows", [])
    gate_summary_rows = analysis.get("gate_summary_rows", [])
    gate_value_rows = analysis.get("gate_value_rows", [])

    if layer_summary_rows:
        paths["channel_summary_csv"] = write_metrics_rows(layer_summary_rows, output_dir / f"{prefix_part}channel_summary.csv")
    if channel_importance_rows:
        paths["channel_importance_csv"] = write_metrics_rows(channel_importance_rows, output_dir / f"{prefix_part}channel_importance.csv")
    if gate_summary_rows:
        paths["gate_summary_csv"] = write_metrics_rows(gate_summary_rows, output_dir / f"{prefix_part}gate_summary.csv")
    if gate_value_rows:
        paths["gate_values_csv"] = write_metrics_rows(gate_value_rows, output_dir / f"{prefix_part}gate_values.csv")

    paths["channel_analysis_json"] = _write_json(output_dir / f"{prefix_part}channel_analysis.json", analysis)
    paths["channel_analysis_pdf"] = save_channel_analysis_pdf(
        analysis,
        output_dir / f"{prefix_part}channel_analysis.pdf",
        title=title,
    )
    return paths


def save_gating_analysis_artifacts(
    output_dir: Path | str,
    analysis: Mapping[str, Any],
    *,
    prefix: Optional[str] = None,
    title: str = "Gating Analysis",
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_part = f"{prefix}_" if prefix else ""

    gate_summary_rows = list(analysis.get("gate_summary_rows", []))
    gate_value_rows = list(analysis.get("gate_value_rows", []))
    global_summary = dict(analysis.get("global_summary", {}))
    global_summary["analysis_type"] = "gating"
    global_summary["total_near_off_channels"] = int(
        sum(int(row.get("near_off_channels") or 0) for row in gate_summary_rows)
    )
    global_summary["mean_gate_value"] = float(
        np.mean([float(row.get("gate_mean") or 0.0) for row in gate_summary_rows]) if gate_summary_rows else 0.0
    )

    gating_report = {
        "model_info": analysis.get("model_info", {}),
        "global_summary": global_summary,
        "gate_summary_rows": gate_summary_rows,
        "gate_value_rows": gate_value_rows,
    }

    paths: Dict[str, Path] = {}
    if gate_summary_rows:
        paths["gating_summary_csv"] = write_metrics_rows(gate_summary_rows, output_dir / f"{prefix_part}gating_summary.csv")
    if gate_value_rows:
        paths["gating_values_csv"] = write_metrics_rows(gate_value_rows, output_dir / f"{prefix_part}gating_values.csv")
    paths["gating_analysis_json"] = _write_json(output_dir / f"{prefix_part}gating_analysis.json", gating_report)
    paths["gating_analysis_pdf"] = save_channel_analysis_pdf(
        gating_report,
        output_dir / f"{prefix_part}gating_analysis.pdf",
        title=title,
    )
    return paths


def save_comparison_artifacts(
    output_dir: Path | str,
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    title: str,
    extra_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    rows = [dict(row) for row in comparison_rows]
    if rows:
        paths["comparison_csv"] = write_metrics_rows(rows, output_dir / f"{prefix}.csv")
    report_payload = {
        "global_summary": dict(extra_report.get("global_summary", {})) if extra_report else {},
        "comparison_rows": rows,
    }
    if extra_report:
        for key, value in extra_report.items():
            if key == "global_summary":
                continue
            report_payload[key] = value
    paths["comparison_json"] = _write_json(output_dir / f"{prefix}.json", report_payload)
    paths["comparison_pdf"] = save_channel_analysis_pdf(report_payload, output_dir / f"{prefix}.pdf", title=title)
    return paths


def save_pruning_analysis_artifacts(
    output_dir: Path | str,
    pruning_report: Mapping[str, Any],
    *,
    title: str = "Pruning Analysis",
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    teacher_channel_rows = list(pruning_report.get("teacher_channel_analysis", []))
    pruning_summary_rows = list(pruning_report.get("pruning_summary_rows", []))
    teacher_vs_student_rows = list(pruning_report.get("teacher_vs_student_rows", []))
    channel_level_detail = list(pruning_report.get("channel_level_detail", []))
    teacher_summary_rows = [
        {
            "layer_name": row.get("layer_name"),
            "module_type": row.get("module_type"),
            "channel_layer_type": row.get("channel_layer_type"),
            "in_channels": row.get("in_channels"),
            "teacher_out_channels": row.get("teacher_out_channels"),
            "kernel_size": row.get("kernel_size"),
            "weight_shape": row.get("weight_shape"),
            "importance_min": row.get("importance_min"),
            "importance_max": row.get("importance_max"),
            "importance_mean": row.get("importance_mean"),
            "importance_std": row.get("importance_std"),
        }
        for row in teacher_channel_rows
    ]
    pruning_rows_for_pdf = [
        {
            "layer_name": row.get("layer_name"),
            "teacher_out_channels": row.get("teacher_out_channels", row.get("source_out_channels")),
            "student_out_channels": row.get("student_out_channels", row.get("kept_channels")),
            "pruned_channels": row.get("pruned_channels"),
            "actual_prune_ratio": row.get("actual_prune_ratio"),
            "importance_mean": row.get("importance_mean"),
            "importance_std": row.get("importance_std"),
        }
        for row in pruning_summary_rows
    ]

    if teacher_channel_rows:
        paths["teacher_channel_summary_csv"] = write_metrics_rows(teacher_summary_rows, output_dir / "teacher_channel_summary.csv")
    if channel_level_detail:
        paths["teacher_channel_importance_csv"] = write_metrics_rows(channel_level_detail, output_dir / "teacher_channel_importance.csv")
        paths["channel_level_detail_csv"] = write_metrics_rows(channel_level_detail, output_dir / "channel_level_detail.csv")
    if pruning_summary_rows:
        paths["pruning_summary_csv"] = write_metrics_rows(pruning_summary_rows, output_dir / "pruning_summary.csv")
    if teacher_vs_student_rows:
        paths["teacher_vs_student_csv"] = write_metrics_rows(teacher_vs_student_rows, output_dir / "teacher_vs_student_channels.csv")
    if pruning_report.get("global_pruning_summary"):
        paths["global_pruning_summary_csv"] = write_metrics_rows([dict(pruning_report["global_pruning_summary"])], output_dir / "global_pruning_summary.csv")

    pruning_payload = {
        "global_summary": pruning_report.get("global_pruning_summary", {}),
        "layer_summary_rows": teacher_summary_rows,
        "pruning_summary_rows": pruning_rows_for_pdf,
        "teacher_vs_student_rows": teacher_vs_student_rows,
        "channel_level_detail": channel_level_detail,
    }
    paths["pruning_analysis_json"] = _write_json(output_dir / "pruning_analysis.json", pruning_payload)
    paths["pruning_analysis_pdf"] = save_channel_analysis_pdf(pruning_payload, output_dir / "pruning_analysis.pdf", title=title)
    return paths
