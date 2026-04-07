from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    plt = None
    PdfPages = None


PROJECT_ROOT = Path(__file__).resolve().parent
STAGE_ORDER = {
    "basic": 1,
    "teacher": 2,
    "pruned_student": 3,
    "tuned_student": 4,
}
STAGE_LABELS = {
    "basic": "Basic Baseline",
    "teacher": "Teacher",
    "pruned_student": "Pruned Student Blueprint",
    "tuned_student": "Tuned Student",
}
SPLIT_ORDER = {"train": 1, "val": 2, "test": 3}


def sanitize_tag(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._-") or "unknown"


def _project_relative_path(path_value: Path | str | None) -> str:
    if path_value in (None, ""):
        return ""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_from_project(path_value: Path | str | None) -> Optional[Path]:
    if path_value in (None, ""):
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else {"value": payload}


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    return _read_json(path) if path.is_file() else None


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> Optional[Path]:
    rows = [dict(row) for row in rows]
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return path


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    float_value = _coerce_float(value)
    if float_value is None:
        return None
    return int(float_value)


def _metric_value(rows: Sequence[Mapping[str, Any]], split: str, key: str) -> Optional[float]:
    for row in rows:
        if row.get("split") == split:
            value = _coerce_float(row.get(key))
            if value is not None:
                return value
    return None


def _first_metric_value(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    for row in rows:
        value = _coerce_float(row.get(key))
        if value is not None:
            return value
    return None


def _normalize_metrics_rows(stage: str, run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        resolved = dict(row)
        resolved["stage"] = stage
        resolved["stage_label"] = STAGE_LABELS[stage]
        resolved["stage_order"] = STAGE_ORDER[stage]
        resolved["run_dir"] = _project_relative_path(run_dir)
        split = str(resolved.get("split", ""))
        resolved["split_order"] = SPLIT_ORDER.get(split, 99)
        for key in (
            "dice",
            "iou",
            "hd95",
            "params",
            "trainable_params",
            "flops",
            "fps",
            "inference_time_seconds",
            "evaluation_time_seconds",
            "prune_ratio",
        ):
            value = _coerce_float(resolved.get(key))
            if value is not None:
                resolved[key] = value
        normalized.append(resolved)
    normalized.sort(key=lambda row: (row["stage_order"], row["split_order"], str(row.get("split", ""))))
    return normalized


def _stage_overview_row(
    *,
    stage: str,
    run_dir: Path,
    model_info: Mapping[str, Any],
    metrics_rows: Sequence[Mapping[str, Any]],
    global_summary: Optional[Mapping[str, Any]] = None,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    global_summary = dict(global_summary or {})
    row = {
        "stage": stage,
        "stage_label": STAGE_LABELS[stage],
        "stage_order": STAGE_ORDER[stage],
        "run_dir": _project_relative_path(run_dir),
        "branch": model_info.get("branch"),
        "model_name": model_info.get("model_name"),
        "backbone_name": model_info.get("backbone_name"),
        "student_name": model_info.get("student_name"),
        "num_channel_layers": _coerce_int(global_summary.get("num_channel_layers", global_summary.get("num_layers_analyzed"))),
        "total_output_channels": _coerce_int(global_summary.get("total_output_channels", global_summary.get("total_channels_after"))),
        "num_gate_layers": _coerce_int(global_summary.get("num_gate_layers")),
        "total_gate_channels": _coerce_int(global_summary.get("total_gate_channels")),
        "total_channels_before": _coerce_int(global_summary.get("total_channels_before")),
        "total_channels_after": _coerce_int(global_summary.get("total_channels_after")),
        "total_channels_pruned": _coerce_int(global_summary.get("total_channels_pruned")),
        "global_prune_ratio": _coerce_float(global_summary.get("global_prune_ratio")),
        "val_dice": _metric_value(metrics_rows, "val", "dice"),
        "val_iou": _metric_value(metrics_rows, "val", "iou"),
        "val_hd95": _metric_value(metrics_rows, "val", "hd95"),
        "test_dice": _metric_value(metrics_rows, "test", "dice"),
        "test_iou": _metric_value(metrics_rows, "test", "iou"),
        "test_hd95": _metric_value(metrics_rows, "test", "hd95"),
        "params": _first_metric_value(metrics_rows, "params"),
        "trainable_params": _first_metric_value(metrics_rows, "trainable_params"),
        "flops": _first_metric_value(metrics_rows, "flops"),
        "fps": _first_metric_value(metrics_rows, "fps"),
        "inference_time_seconds": _first_metric_value(metrics_rows, "inference_time_seconds"),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _load_basic_stage(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics" / "basic_metrics.csv"
    channel_analysis_path = run_dir / "artifacts" / "channel_analysis" / "channel_analysis.json"
    model_config_path = run_dir / "configs" / "model_config.json"

    metrics_rows = _normalize_metrics_rows("basic", run_dir, _read_csv_rows(metrics_path))
    channel_analysis = _read_json_if_exists(channel_analysis_path) or {}
    model_info = (
        channel_analysis.get("model_info")
        or _read_json_if_exists(model_config_path)
        or {}
    )
    overview_row = _stage_overview_row(
        stage="basic",
        run_dir=run_dir,
        model_info=model_info,
        metrics_rows=metrics_rows,
        global_summary=channel_analysis.get("global_summary", {}),
    )
    return {
        "stage": "basic",
        "metrics_rows": metrics_rows,
        "overview_row": overview_row,
        "channel_analysis": channel_analysis,
        "source_files": {
            "metrics_csv": _project_relative_path(metrics_path) if metrics_path.is_file() else None,
            "channel_analysis_json": _project_relative_path(channel_analysis_path) if channel_analysis_path.is_file() else None,
            "model_config_json": _project_relative_path(model_config_path) if model_config_path.is_file() else None,
        },
    }


def _load_teacher_stage(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics" / "teacher_metrics.csv"
    channel_analysis_path = run_dir / "artifacts" / "channel_analysis" / "channel_analysis.json"
    model_config_path = run_dir / "configs" / "model_config.json"

    metrics_rows = _normalize_metrics_rows("teacher", run_dir, _read_csv_rows(metrics_path))
    channel_analysis = _read_json_if_exists(channel_analysis_path) or {}
    model_info = (
        channel_analysis.get("model_info")
        or _read_json_if_exists(model_config_path)
        or {}
    )
    overview_row = _stage_overview_row(
        stage="teacher",
        run_dir=run_dir,
        model_info=model_info,
        metrics_rows=metrics_rows,
        global_summary=channel_analysis.get("global_summary", {}),
    )
    return {
        "stage": "teacher",
        "metrics_rows": metrics_rows,
        "overview_row": overview_row,
        "channel_analysis": channel_analysis,
        "source_files": {
            "metrics_csv": _project_relative_path(metrics_path) if metrics_path.is_file() else None,
            "channel_analysis_json": _project_relative_path(channel_analysis_path) if channel_analysis_path.is_file() else None,
            "model_config_json": _project_relative_path(model_config_path) if model_config_path.is_file() else None,
        },
    }


def _load_pruned_stage(run_dir: Path) -> Dict[str, Any]:
    blueprint_path = _first_existing(
        (
            run_dir / "artifacts" / "blueprint.json",
            run_dir / "configs" / "pruning_config.json",
            run_dir / "metrics" / "pruning_summary.json",
        )
    )
    pruning_report_path = run_dir / "artifacts" / "pruning_analysis" / "pruning_analysis.json"
    teacher_vs_student_path = run_dir / "artifacts" / "pruning_analysis" / "teacher_vs_student_channels.csv"
    pruning_global_summary_path = run_dir / "artifacts" / "pruning_analysis" / "global_pruning_summary.csv"

    blueprint = _read_json_if_exists(blueprint_path) if blueprint_path else {}
    pruning_report = _read_json_if_exists(pruning_report_path) or {}
    teacher_vs_student_rows = _read_csv_rows(teacher_vs_student_path) or list(pruning_report.get("teacher_vs_student_rows", []))
    pruning_global_summary = dict(pruning_report.get("global_summary") or blueprint.get("global_pruning_summary") or {})

    model_info = {
        "branch": "proposal",
        "model_name": blueprint.get("student_name", "pdg_unet"),
        "backbone_name": blueprint.get("teacher_model"),
        "student_name": blueprint.get("student_name", "pdg_unet"),
    }
    overview_row = _stage_overview_row(
        stage="pruned_student",
        run_dir=run_dir,
        model_info=model_info,
        metrics_rows=[],
        global_summary=pruning_global_summary,
        extra_fields={
            "channel_config": ",".join(str(value) for value in blueprint.get("channel_config", [])),
            "target_modules": ",".join(str(value) for value in blueprint.get("target_modules", [])),
            "criterion": blueprint.get("criterion"),
            "mapping_rule": blueprint.get("mapping_rule"),
            "note": "Architecture-only stage. No segmentation metrics are expected before student tuning.",
        },
    )
    return {
        "stage": "pruned_student",
        "metrics_rows": [],
        "overview_row": overview_row,
        "blueprint": blueprint,
        "pruning_report": pruning_report,
        "teacher_vs_student_rows": teacher_vs_student_rows,
        "pruning_global_summary": pruning_global_summary,
        "source_files": {
            "blueprint_json": _project_relative_path(blueprint_path) if blueprint_path and blueprint_path.is_file() else None,
            "pruning_analysis_json": _project_relative_path(pruning_report_path) if pruning_report_path.is_file() else None,
            "teacher_vs_student_csv": _project_relative_path(teacher_vs_student_path) if teacher_vs_student_path.is_file() else None,
            "global_pruning_summary_csv": _project_relative_path(pruning_global_summary_path) if pruning_global_summary_path.is_file() else None,
        },
    }


def _load_tuned_student_stage(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics" / "student_metrics.csv"
    final_channel_analysis_path = run_dir / "artifacts" / "channel_analysis" / "student_final_channel_analysis.json"
    input_channel_analysis_path = run_dir / "artifacts" / "channel_analysis" / "student_input_channel_analysis.json"
    student_tuning_comparison_path = run_dir / "artifacts" / "channel_analysis" / "student_tuning_comparison.csv"
    model_config_path = run_dir / "configs" / "model_config.json"

    metrics_rows = _normalize_metrics_rows("tuned_student", run_dir, _read_csv_rows(metrics_path))
    final_channel_analysis = _read_json_if_exists(final_channel_analysis_path) or {}
    input_channel_analysis = _read_json_if_exists(input_channel_analysis_path) or {}
    student_tuning_rows = _read_csv_rows(student_tuning_comparison_path)
    model_info = (
        final_channel_analysis.get("model_info")
        or _read_json_if_exists(model_config_path)
        or {}
    )
    overview_row = _stage_overview_row(
        stage="tuned_student",
        run_dir=run_dir,
        model_info=model_info,
        metrics_rows=metrics_rows,
        global_summary=final_channel_analysis.get("global_summary", {}),
        extra_fields={
            "input_total_channels": _coerce_int(input_channel_analysis.get("global_summary", {}).get("total_output_channels")),
            "final_total_channels": _coerce_int(final_channel_analysis.get("global_summary", {}).get("total_output_channels")),
            "input_gate_layers": _coerce_int(input_channel_analysis.get("global_summary", {}).get("num_gate_layers")),
            "final_gate_layers": _coerce_int(final_channel_analysis.get("global_summary", {}).get("num_gate_layers")),
        },
    )
    return {
        "stage": "tuned_student",
        "metrics_rows": metrics_rows,
        "overview_row": overview_row,
        "input_channel_analysis": input_channel_analysis,
        "final_channel_analysis": final_channel_analysis,
        "student_tuning_rows": student_tuning_rows,
        "source_files": {
            "metrics_csv": _project_relative_path(metrics_path) if metrics_path.is_file() else None,
            "student_input_channel_analysis_json": _project_relative_path(input_channel_analysis_path) if input_channel_analysis_path.is_file() else None,
            "student_final_channel_analysis_json": _project_relative_path(final_channel_analysis_path) if final_channel_analysis_path.is_file() else None,
            "student_tuning_comparison_csv": _project_relative_path(student_tuning_comparison_path) if student_tuning_comparison_path.is_file() else None,
            "model_config_json": _project_relative_path(model_config_path) if model_config_path.is_file() else None,
        },
    }


def _resolve_dirs(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    basic_run_dir = _resolve_from_project(args.basic_run_dir)
    teacher_run_dir = _resolve_from_project(args.teacher_run_dir)
    pruning_run_dir = _resolve_from_project(args.pruning_run_dir)
    student_run_dir = _resolve_from_project(args.student_run_dir)
    pipeline_dir = _resolve_from_project(args.pipeline_dir)

    if pipeline_dir:
        pipeline_summary_path = pipeline_dir / "pipeline_summary.json"
        if not pipeline_summary_path.is_file():
            raise FileNotFoundError(f"Missing pipeline summary: {pipeline_summary_path}")
        pipeline_summary = _read_json(pipeline_summary_path)
        teacher_run_dir = teacher_run_dir or _resolve_from_project(pipeline_summary["teacher_run_dir"])
        pruning_run_dir = pruning_run_dir or _resolve_from_project(pipeline_summary["pruning_run_dir"])
        student_run_dir = student_run_dir or _resolve_from_project(pipeline_summary["student_run_dir"])

    return {
        "basic_run_dir": basic_run_dir,
        "pipeline_dir": pipeline_dir,
        "teacher_run_dir": teacher_run_dir,
        "pruning_run_dir": pruning_run_dir,
        "student_run_dir": student_run_dir,
    }


def _save_table_page(pdf: PdfPages, rows: Sequence[Mapping[str, Any]], title: str, *, max_rows: int = 20) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        return
    headers = list(rows[0].keys())
    for start in range(0, len(rows), max_rows):
        chunk = rows[start : start + max_rows]
        values = [[row.get(column, "") for column in headers] for row in chunk]
        fig, ax = plt.subplots(figsize=(15, 4 + 0.35 * len(chunk)))
        ax.axis("off")
        table = ax.table(cellText=values, colLabels=headers, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.25)
        suffix = "" if len(rows) <= max_rows else f" ({start + 1}-{start + len(chunk)})"
        ax.set_title(f"{title}{suffix}")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _save_comparison_report_pdf(report: Mapping[str, Any], pdf_path: Path, *, title: str) -> Path:
    if plt is None or PdfPages is None:
        raise ModuleNotFoundError("matplotlib is required to export comparison_report.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    overview_rows = list(report.get("overview_rows", []))
    metrics_rows = list(report.get("metrics_rows", []))
    teacher_vs_student_rows = list(report.get("teacher_vs_student_rows", []))
    student_tuning_rows = list(report.get("student_tuning_rows", []))
    pruning_global_summary = dict(report.get("pruning_global_summary", {}))
    resolved_inputs = list(report.get("resolved_inputs", []))

    with PdfPages(pdf_path) as pdf:
        if resolved_inputs:
            _save_table_page(pdf, resolved_inputs, f"{title} | Resolved Inputs", max_rows=25)
        if overview_rows:
            _save_table_page(pdf, overview_rows, f"{title} | Stage Overview")
        if metrics_rows:
            _save_table_page(pdf, metrics_rows, f"{title} | Performance Comparison")
        if pruning_global_summary:
            _save_table_page(
                pdf,
                [{"key": key, "value": value} for key, value in pruning_global_summary.items()],
                f"{title} | Global Pruning Summary",
                max_rows=25,
            )
        if teacher_vs_student_rows:
            _save_table_page(pdf, teacher_vs_student_rows, f"{title} | Teacher vs Pruned Student")
        if student_tuning_rows:
            _save_table_page(pdf, student_tuning_rows, f"{title} | Student Tuning Comparison")

        if overview_rows:
            labels = [str(row.get("stage_label", row.get("stage"))) for row in overview_rows]
            channel_values = [float(row.get("total_output_channels") or 0.0) for row in overview_rows]
            gate_values = [float(row.get("num_gate_layers") or 0.0) for row in overview_rows]
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            axes[0].bar(labels, channel_values)
            axes[0].set_title("Total Output Channels by Stage")
            axes[0].grid(alpha=0.25, axis="y")
            axes[0].tick_params(axis="x", rotation=25)
            axes[1].bar(labels, gate_values)
            axes[1].set_title("Gate Layers by Stage")
            axes[1].grid(alpha=0.25, axis="y")
            axes[1].tick_params(axis="x", rotation=25)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if metrics_rows:
            metric_candidates = [metric for metric in ("dice", "iou", "hd95") if any(_coerce_float(row.get(metric)) is not None for row in metrics_rows)]
            if metric_candidates:
                fig, axes = plt.subplots(len(metric_candidates), 1, figsize=(12, 3.5 * len(metric_candidates)))
                if len(metric_candidates) == 1:
                    axes = [axes]
                labels = [f"{row.get('stage')}:{row.get('split')}" for row in metrics_rows]
                for axis, metric_name in zip(axes, metric_candidates):
                    axis.bar(labels, [float(_coerce_float(row.get(metric_name)) or 0.0) for row in metrics_rows])
                    axis.set_title(f"{metric_name.upper()} by Stage and Split")
                    axis.grid(alpha=0.25, axis="y")
                    axis.tick_params(axis="x", rotation=35)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        if teacher_vs_student_rows:
            labels = [str(row.get("layer_name")) for row in teacher_vs_student_rows]
            teacher_channels = [float(_coerce_float(row.get("teacher_out_channels")) or 0.0) for row in teacher_vs_student_rows]
            student_channels = [float(_coerce_float(row.get("student_out_channels")) or 0.0) for row in teacher_vs_student_rows]
            prune_ratio = [float(_coerce_float(row.get("actual_prune_ratio")) or 0.0) for row in teacher_vs_student_rows]
            x = list(range(len(labels)))
            width = 0.35
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            axes[0].bar(x - width / 2, teacher_channels, width=width, label="teacher")
            axes[0].bar(x + width / 2, student_channels, width=width, label="pruned_student")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(labels, rotation=45)
            axes[0].set_title("Teacher vs Pruned Student Channels")
            axes[0].grid(alpha=0.25, axis="y")
            axes[0].legend()
            axes[1].bar(labels, prune_ratio)
            axes[1].set_title("Actual Prune Ratio per Layer")
            axes[1].grid(alpha=0.25, axis="y")
            axes[1].tick_params(axis="x", rotation=45)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if student_tuning_rows:
            channel_rows = [row for row in student_tuning_rows if row.get("analysis_type") == "channel"]
            gate_rows = [row for row in student_tuning_rows if row.get("analysis_type") == "gate"]
            if channel_rows:
                labels = [str(row.get("layer_name")) for row in channel_rows]
                input_channels = [float(_coerce_float(row.get("out_channels_input")) or 0.0) for row in channel_rows]
                final_channels = [float(_coerce_float(row.get("out_channels_final")) or 0.0) for row in channel_rows]
                x = list(range(len(labels)))
                width = 0.35
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.bar(x - width / 2, input_channels, width=width, label="input")
                ax.bar(x + width / 2, final_channels, width=width, label="final")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45)
                ax.set_title("Student Input vs Final Channels")
                ax.grid(alpha=0.25, axis="y")
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            if gate_rows:
                labels = [str(row.get("layer_name")) for row in gate_rows]
                input_gate = [float(_coerce_float(row.get("gate_mean_input")) or 0.0) for row in gate_rows]
                final_gate = [float(_coerce_float(row.get("gate_mean_final")) or 0.0) for row in gate_rows]
                x = list(range(len(labels)))
                width = 0.35
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.bar(x - width / 2, input_gate, width=width, label="input")
                ax.bar(x + width / 2, final_gate, width=width, label="final")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45)
                ax.set_title("Student Gate Mean Before vs After Tuning")
                ax.grid(alpha=0.25, axis="y")
                ax.legend()
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    return pdf_path


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    resolved_dirs = _resolve_dirs(args)
    bundles: List[Dict[str, Any]] = []

    for key, path in resolved_dirs.items():
        if path is not None and not path.exists():
            logging.warning("Skip %s because the path does not exist: %s", key, path)

    if resolved_dirs["basic_run_dir"] and resolved_dirs["basic_run_dir"].is_dir():
        bundles.append(_load_basic_stage(resolved_dirs["basic_run_dir"]))
    if resolved_dirs["teacher_run_dir"] and resolved_dirs["teacher_run_dir"].is_dir():
        bundles.append(_load_teacher_stage(resolved_dirs["teacher_run_dir"]))
    if resolved_dirs["pruning_run_dir"] and resolved_dirs["pruning_run_dir"].is_dir():
        bundles.append(_load_pruned_stage(resolved_dirs["pruning_run_dir"]))
    if resolved_dirs["student_run_dir"] and resolved_dirs["student_run_dir"].is_dir():
        bundles.append(_load_tuned_student_stage(resolved_dirs["student_run_dir"]))

    if not bundles:
        raise ValueError("No valid run directories were resolved. Provide --basic_run_dir and/or proposal phase directories.")

    overview_rows = sorted((bundle["overview_row"] for bundle in bundles), key=lambda row: row["stage_order"])
    metrics_rows: List[Dict[str, Any]] = []
    teacher_vs_student_rows: List[Dict[str, Any]] = []
    student_tuning_rows: List[Dict[str, Any]] = []
    pruning_global_summary: Dict[str, Any] = {}
    source_files: Dict[str, Any] = {}

    for bundle in bundles:
        metrics_rows.extend(bundle.get("metrics_rows", []))
        if bundle["stage"] == "pruned_student":
            teacher_vs_student_rows = list(bundle.get("teacher_vs_student_rows", []))
            pruning_global_summary = dict(bundle.get("pruning_global_summary", {}))
        if bundle["stage"] == "tuned_student":
            student_tuning_rows = list(bundle.get("student_tuning_rows", []))
        source_files[bundle["stage"]] = bundle.get("source_files", {})

    metrics_rows.sort(key=lambda row: (row["stage_order"], row["split_order"], str(row.get("split", ""))))

    resolved_inputs = [
        {
            "input_name": key,
            "path": _project_relative_path(path),
            "exists": int(path is not None and path.exists()),
        }
        for key, path in resolved_dirs.items()
    ]

    return {
        "overview_rows": overview_rows,
        "metrics_rows": metrics_rows,
        "teacher_vs_student_rows": teacher_vs_student_rows,
        "student_tuning_rows": student_tuning_rows,
        "pruning_global_summary": pruning_global_summary,
        "resolved_inputs": resolved_inputs,
        "source_files": source_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a unified report from saved artifacts: basic vs teacher vs pruned student vs tuned student."
    )
    parser.add_argument("--basic_run_dir", type=str, default="", help="Path to a basic branch run directory.")
    parser.add_argument("--pipeline_dir", type=str, default="", help="Optional proposal pipeline directory containing pipeline_summary.json.")
    parser.add_argument("--teacher_run_dir", type=str, default="", help="Path to the proposal teacher run directory.")
    parser.add_argument("--pruning_run_dir", type=str, default="", help="Path to the proposal pruning run directory.")
    parser.add_argument("--student_run_dir", type=str, default="", help="Path to the proposal student run directory.")
    parser.add_argument("--output_dir", type=str, default="", help="Where the aggregated report should be written.")
    parser.add_argument("--comparison_name", type=str, default="artifact_comparison", help="Name used when auto-creating the output directory.")
    parser.add_argument("--title", type=str, default="Unified Model Comparison Report", help="PDF report title.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    report = build_report(args)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        PROJECT_ROOT / "outputs" / "comparisons" / sanitize_tag(args.comparison_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_overview_csv = _write_csv(output_dir / "stage_overview.csv", report["overview_rows"])
    performance_csv = _write_csv(output_dir / "performance_comparison.csv", report["metrics_rows"])
    pruning_global_csv = _write_csv(
        output_dir / "pruning_global_summary.csv",
        [report["pruning_global_summary"]] if report.get("pruning_global_summary") else [],
    )
    teacher_vs_student_csv = _write_csv(output_dir / "teacher_vs_student_channels.csv", report["teacher_vs_student_rows"])
    student_tuning_csv = _write_csv(output_dir / "student_tuning_comparison.csv", report["student_tuning_rows"])
    summary_json = _write_json(output_dir / "comparison_summary.json", report)
    report_pdf = None
    if plt is None or PdfPages is None:
        logging.warning("matplotlib is not installed. Skip comparison_report.pdf and keep CSV/JSON outputs only.")
    else:
        report_pdf = _save_comparison_report_pdf(report, output_dir / "comparison_report.pdf", title=args.title)

    logging.info("Comparison report saved to %s", output_dir)
    logging.info("Generated: %s", summary_json)
    if report_pdf is not None:
        logging.info("Generated: %s", report_pdf)
    for path in (stage_overview_csv, performance_csv, pruning_global_csv, teacher_vs_student_csv, student_tuning_csv):
        if path is not None:
            logging.info("Generated: %s", path)


if __name__ == "__main__":
    main()
