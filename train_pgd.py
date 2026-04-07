from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import Normalize, RandomGenerator, ToTensor, build_dataset, list_available_datasets
from networks.PGD_Unet.gated_unet import PDGUNet
from networks.PGD_Unet.pruning import extract_pruned_blueprint, load_blueprint_artifact, save_blueprint_artifact
from networks.net_factory import get_model_metadata, list_models, net_factory
from utils.channel_analysis import (
    build_analysis_comparison,
    extract_channel_analysis,
    save_channel_analysis_artifacts,
    save_comparison_artifacts,
    save_pruning_analysis_artifacts,
)
from utils.checkpoints import build_checkpoint_metadata, load_checkpoint, load_checkpoint_into_model, resolve_phase_checkpoint, save_checkpoint
from utils.checkpoint_resolver import (
    build_expected_signature,
    find_compatible_checkpoint,
    register_reused_checkpoint,
    resolve_basic_checkpoint,
    resolve_run_checkpoint,
)
from utils.compression_loss import CompressionLoss
from utils.evaluation import build_evaluation_output_dir, evaluate_segmentation_dataset, save_evaluation_artifacts
from utils.experiment import (
    build_pdg_phase_dir,
    build_pdg_root_dir,
    ensure_run_layout,
    normalize_path_string,
    project_relative_path,
    write_model_config,
    write_run_config,
)
from utils.losses import DiceLoss
from utils.model_output import extract_logits, extract_model_info
from utils.profiling import benchmark_inference, count_parameters, maybe_compute_flops
from utils.reporting import save_loss_pdf, save_performance_pdf, save_visualization_overview_image, save_visualization_pdf, write_metrics_rows


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "Kvasir-SEG"


parser = argparse.ArgumentParser(description="Teacher -> Pruning -> Student training for PDG-UNet")
parser.add_argument("--root_path", type=str, default=str(DEFAULT_DATA_ROOT))
parser.add_argument("--dataset", type=str, default="kvasir", choices=list_available_datasets())
parser.add_argument("--exp", type=str, default="pdg_pipeline")
parser.add_argument("--teacher_model", type=str, default="unet_resnet152", choices=list_models())
parser.add_argument("--teacher_checkpoint", type=str, default="")
parser.add_argument("--train_split", type=str, default="train", choices=["train", "val", "test"])
parser.add_argument("--val_split", type=str, default="val", choices=["train", "val", "test"])
parser.add_argument("--max_epochs_teacher", type=int, default=20)
parser.add_argument("--max_epochs_student", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--teacher_lr", type=float, default=0.01)
parser.add_argument("--student_lr", type=float, default=1e-4)
parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256])
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--encoder_pretrained", type=int, default=0)
parser.add_argument("--prune_ratio", type=float, default=0.5)
parser.add_argument("--lambda_distill", type=float, default=0.3)
parser.add_argument("--lambda_sparsity", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--save_visualizations", type=int, default=1)
parser.add_argument("--vis_num_samples", type=int, default=8)
parser.add_argument("--final_eval_splits", nargs="*", default=["train", "val", "test"])
parser.add_argument("--force_retrain_teacher", type=int, default=0)
parser.add_argument("--force_reprune", type=int, default=0)
parser.add_argument("--force_retrain_student", type=int, default=0)
parser.add_argument("--output_root", type=str, default="", help="root directory for all exported outputs; defaults to PROJECT_ROOT/outputs")
parser.add_argument("--save_history_checkpoints", type=int, default=0, help="set to 1 to keep per-epoch checkpoint history in addition to best/last")
args = parser.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _proposal_root_dir() -> Path:
    return build_pdg_root_dir(
        project_root=PROJECT_ROOT,
        dataset=args.dataset,
        teacher_name=args.teacher_model,
        output_root=args.output_root or None,
    )


def _phase_dir(phase: str) -> Path:
    return build_pdg_phase_dir(
        project_root=PROJECT_ROOT,
        dataset=args.dataset,
        teacher_name=args.teacher_model,
        phase=phase,
        output_root=args.output_root or None,
    )


def _configure_logging(log_path: Path) -> None:
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def _teacher_signature(in_channels: int) -> dict:
    return build_expected_signature(
        dataset=args.dataset,
        model_name=args.teacher_model,
        num_classes=args.num_classes,
        in_channels=in_channels,
        patch_size=args.patch_size,
        encoder_pretrained=bool(args.encoder_pretrained) if args.teacher_model == "unet_resnet152" else None,
    )


def _student_signature(channel_config, in_channels: int) -> dict:
    return build_expected_signature(
        dataset=args.dataset,
        model_name="pdg_unet",
        num_classes=args.num_classes,
        in_channels=in_channels,
        patch_size=args.patch_size,
        teacher_model=args.teacher_model,
        channel_config=channel_config,
    )


def _restore_loss_pdf_if_possible(history: dict | None, pdf_path: Path, title: str) -> None:
    if not history:
        return
    if any(values for values in history.values() if isinstance(values, list)):
        save_loss_pdf(history, pdf_path, title=title)


def _copy_exact_matching_weights(source_model, target_model) -> dict:
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    copied_keys = []
    skipped_keys = []

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None or source_tensor.shape != target_tensor.shape:
            skipped_keys.append(key)
            continue
        target_state[key] = source_tensor.detach().clone()
        copied_keys.append(key)

    target_model.load_state_dict(target_state, strict=False)
    total_target_tensors = len(target_state)
    return {
        "copied_tensor_keys": int(len(copied_keys)),
        "skipped_tensor_keys": int(len(skipped_keys)),
        "total_target_tensors": int(total_target_tensors),
        "copy_ratio": float(len(copied_keys) / max(1, total_target_tensors)),
        "copied_key_examples": copied_keys[:20],
        "skipped_key_examples": skipped_keys[:20],
    }


def _aggregate_phase_metrics(*phase_exports: dict) -> list[dict]:
    rows = []
    for export_bundle in phase_exports:
        rows.extend(list(export_bundle.get("metrics_rows", [])))
    return rows


def _build_pipeline_overview_rows(*phase_artifacts: dict) -> list[dict]:
    rows = []
    for artifact in phase_artifacts:
        phase = artifact.get("phase")
        export_bundle = artifact.get("evaluation_bundle", {})
        metrics_rows = list(export_bundle.get("metrics_rows", []))
        split_lookup = {row.get("split"): row for row in metrics_rows}
        row = {
            "phase": phase,
            "run_dir": project_relative_path(artifact.get("run_dir"), PROJECT_ROOT),
            "checkpoint_path": project_relative_path(artifact.get("checkpoint_path"), PROJECT_ROOT) if artifact.get("checkpoint_path") else "",
            "model_name": artifact.get("metadata", {}).get("model_name"),
            "backbone_name": artifact.get("metadata", {}).get("backbone_name"),
            "student_name": artifact.get("metadata", {}).get("student_name"),
            "train_dice": split_lookup.get("train", {}).get("dice"),
            "val_dice": split_lookup.get("val", {}).get("dice"),
            "test_dice": split_lookup.get("test", {}).get("dice"),
            "params": next((metric.get("params") for metric in metrics_rows if metric.get("params") is not None), None),
            "flops": next((metric.get("flops") for metric in metrics_rows if metric.get("flops") is not None), None),
            "fps": next((metric.get("fps") for metric in metrics_rows if metric.get("fps") is not None), None),
        }
        if phase == "pruning":
            row["evaluation_note"] = artifact.get("metadata", {}).get("evaluation_note")
            row["weight_transfer_copy_ratio"] = artifact.get("metadata", {}).get("weight_transfer", {}).get("copy_ratio")
        rows.append(row)
    return rows


def _save_pipeline_outputs(pipeline_dir: Path, *phase_artifacts: dict) -> dict:
    layout = ensure_run_layout(pipeline_dir)
    pipeline_metrics_rows = _aggregate_phase_metrics(*phase_artifacts)
    pipeline_overview_rows = _build_pipeline_overview_rows(*phase_artifacts)

    if pipeline_overview_rows:
        write_metrics_rows(pipeline_overview_rows, layout["metrics_dir"] / "pipeline_stage_overview.csv")
    if pipeline_metrics_rows:
        write_metrics_rows(pipeline_metrics_rows, layout["metrics_dir"] / "pipeline_metrics.csv")
        report_rows = []
        for row in pipeline_metrics_rows:
            report_row = dict(row)
            report_row["split"] = f"{row.get('phase', 'phase')}:{row.get('split', 'split')}"
            report_rows.append(report_row)
        save_performance_pdf(
            report_rows,
            layout["reports_dir"] / "pipeline_performance.pdf",
            title="Pipeline stage performance",
        )

    pipeline_visual_index = {}
    for artifact in phase_artifacts:
        phase = artifact.get("phase")
        export_bundle = artifact.get("evaluation_bundle", {})
        sample_map = export_bundle.get("visualization_samples_by_split", {})
        pipeline_visual_index[phase] = {}
        for split, samples in sample_map.items():
            if not samples:
                continue
            prefixed_samples = []
            for sample in samples:
                cloned = dict(sample)
                cloned["case"] = f"{phase}:{sample.get('case', 'unknown')}"
                prefixed_samples.append(cloned)
            save_visualization_pdf(
                prefixed_samples,
                layout["reports_dir"] / f"pipeline_{phase}_{split}_visualizations.pdf",
                title=f"pipeline | {phase} | {split}",
            )
            save_visualization_overview_image(
                prefixed_samples,
                layout["visualization_dir"] / f"pipeline_{phase}_{split}_visualizations.png",
            )
            pipeline_visual_index[phase][split] = {
                "pdf": project_relative_path(layout["reports_dir"] / f"pipeline_{phase}_{split}_visualizations.pdf", PROJECT_ROOT),
                "image": project_relative_path(layout["visualization_dir"] / f"pipeline_{phase}_{split}_visualizations.png", PROJECT_ROOT),
            }

    summary = {
        "dataset": args.dataset,
        "teacher_model": args.teacher_model,
        "prune_ratio": args.prune_ratio,
        "stages": {
            artifact.get("phase"): {
                "run_dir": project_relative_path(artifact.get("run_dir"), PROJECT_ROOT),
                "checkpoint_path": project_relative_path(artifact.get("checkpoint_path"), PROJECT_ROOT) if artifact.get("checkpoint_path") else "",
                "model_info": artifact.get("metadata", {}),
                "split_summaries": artifact.get("evaluation_bundle", {}).get("split_summaries", {}),
            }
            for artifact in phase_artifacts
        },
        "pipeline_stage_overview_csv": project_relative_path(layout["metrics_dir"] / "pipeline_stage_overview.csv", PROJECT_ROOT) if pipeline_overview_rows else "",
        "pipeline_metrics_csv": project_relative_path(layout["metrics_dir"] / "pipeline_metrics.csv", PROJECT_ROOT) if pipeline_metrics_rows else "",
        "pipeline_performance_pdf": project_relative_path(layout["reports_dir"] / "pipeline_performance.pdf", PROJECT_ROOT) if pipeline_metrics_rows else "",
        "pipeline_visualizations": pipeline_visual_index,
    }
    _write_json(layout["evaluation_dir"] / "pipeline_summary.json", summary)
    markdown_lines = [
        f"# Pipeline Evaluation Summary | {args.dataset} | {args.teacher_model}",
        "",
        f"- Teacher model: `{args.teacher_model}`",
        f"- Prune ratio: `{args.prune_ratio}`",
        "",
        "## Stages",
        "",
    ]
    for row in pipeline_overview_rows:
        markdown_lines.append(
            f"- `{row.get('phase')}` | train dice `{row.get('train_dice')}` | val dice `{row.get('val_dice')}` | test dice `{row.get('test_dice')}` | checkpoint `{row.get('checkpoint_path')}`"
        )
    with (layout["evaluation_dir"] / "pipeline_summary.md").open("w", encoding="utf-8") as file:
        file.write("\n".join(markdown_lines) + "\n")
    return {
        "overview_rows": pipeline_overview_rows,
        "metrics_rows": pipeline_metrics_rows,
        "summary": summary,
    }


def _export_student_final_shortcuts(student_run_dir: Path, proposal_root_dir: Path) -> dict:
    student_final_dir = proposal_root_dir / "student_final"
    student_final_dir.mkdir(parents=True, exist_ok=True)
    source_dir = student_run_dir / "checkpoints"
    mapping = {
        "best.pth": "best_student.pth",
        "last.pth": "last_student.pth",
    }
    exported = {}
    for source_name, target_name in mapping.items():
        source_path = source_dir / source_name
        if not source_path.is_file():
            continue
        target_path = student_final_dir / target_name
        payload = load_checkpoint(source_path)
        torch.save(payload, target_path)
        metadata = build_checkpoint_metadata(
            payload,
            target_path,
            project_root=PROJECT_ROOT,
            extra_fields={
                "shortcut_role": target_name.replace(".pth", ""),
                "source_checkpoint_path": project_relative_path(source_path, PROJECT_ROOT),
                "source_run_dir": project_relative_path(student_run_dir, PROJECT_ROOT),
            },
        )
        metadata_name = target_name.replace(".pth", ".json")
        _write_json(student_final_dir / metadata_name, metadata)
        exported[target_name] = project_relative_path(target_path, PROJECT_ROOT)
    return exported


def _validate_label_batch(label_batch, num_classes, sampled_batch):
    label_batch = label_batch.long()
    invalid_mask = (label_batch < 0) | (label_batch >= num_classes)
    if not invalid_mask.any():
        return
    raise ValueError(
        f"Invalid mask values detected. "
        f"Case={sampled_batch.get('case')} invalid={torch.unique(label_batch[invalid_mask].cpu()).tolist()[:20]}"
    )


def _build_loaders(device: torch.device, image_mode: str):
    train_transform = transforms.Compose([Normalize(), RandomGenerator(args.patch_size)])
    eval_transform = transforms.Compose([Normalize(), ToTensor()])
    db_train = build_dataset(args.dataset, args.root_path, split=args.train_split, transform=train_transform, image_mode=image_mode)
    db_val = build_dataset(args.dataset, args.root_path, split=args.val_split, transform=eval_transform, image_mode=image_mode)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=device.type == "cuda")
    return db_train, db_val, trainloader, valloader


def _supervised_loss(logits: torch.Tensor, target: torch.Tensor, ce_loss, dice_loss) -> torch.Tensor:
    return 0.5 * (ce_loss(logits, target.long()) + dice_loss(torch.softmax(logits, dim=1), target.unsqueeze(1)))


def _compute_supervised_val_loss(model, dataloader, device, ce_loss, dice_loss) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            _validate_label_batch(batch["label"], args.num_classes, batch)
            image = F.interpolate(batch["image"].to(device), size=args.patch_size, mode="bilinear", align_corners=False)
            label = F.interpolate(batch["label"].unsqueeze(1).float().to(device), size=args.patch_size, mode="nearest").squeeze(1).long()
            losses.append(float(_supervised_loss(extract_logits(model(image)), label, ce_loss, dice_loss).item()))
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def _compute_profile(model, device: torch.device) -> dict:
    input_shape = (1, args.in_channels, args.patch_size[0], args.patch_size[1])
    profile = {
        "params": int(count_parameters(model)),
        "trainable_params": int(count_parameters(model, trainable_only=True)),
    }
    profile.update(benchmark_inference(model, input_shape=input_shape, device=device))
    profile["flops"] = maybe_compute_flops(model, input_shape=input_shape, device=device)
    return profile


def _export_model_channel_analysis(
    run_dir: Path,
    model,
    *,
    checkpoint_path: Path | None,
    device: torch.device,
    prefix: str | None,
    title: str,
) -> dict:
    if checkpoint_path is not None:
        load_checkpoint_into_model(checkpoint_path, model, device=device)
    analysis = extract_channel_analysis(model, importance_name="filter_l1")
    save_channel_analysis_artifacts(run_dir / "artifacts" / "channel_analysis", analysis, prefix=prefix, title=title)
    logging.info(
        "%s | analyzed_layers=%d | total_channels=%d | gate_layers=%d",
        title,
        analysis["global_summary"]["num_channel_layers"],
        analysis["global_summary"]["total_output_channels"],
        analysis["global_summary"]["num_gate_layers"],
    )
    return analysis


def _export_phase_outputs(run_dir: Path, model, checkpoint_path: Path, phase: str, profile: dict, extra: dict, device: torch.device, image_mode: str) -> dict:
    load_checkpoint_into_model(checkpoint_path, model, device=device)
    layout = ensure_run_layout(run_dir)
    metrics_rows = []
    split_summaries = {}
    visualization_samples_by_split = {}
    eval_transform = transforms.Compose([Normalize(), ToTensor()])
    model_info = dict(extract_model_info(model))
    model_info.update({key: value for key, value in extra.items() if key in {"model_name", "backbone_name", "student_name", "teacher_model", "prune_ratio"}})
    for split in dict.fromkeys(args.final_eval_splits):
        try:
            dataset = build_dataset(args.dataset, args.root_path, split=split, transform=eval_transform, image_mode=image_mode)
        except Exception as error:
            logging.warning("Skip %s/%s: %s", phase, split, error)
            continue
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=device.type == "cuda")
        output_dir = build_evaluation_output_dir(run_dir, args.dataset, extra["model_name"], checkpoint_path, split)
        start = time.perf_counter()
        result = evaluate_segmentation_dataset(
            model,
            dataloader,
            device=device,
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            output_dir=output_dir,
            save_visualizations=bool(args.save_visualizations),
            vis_limit=args.vis_num_samples,
            sample_validator=_validate_label_batch,
            progress_desc=f"{phase}-{split}",
        )
        elapsed = time.perf_counter() - start
        summary = save_evaluation_artifacts(
            output_dir,
            {
                "experiment": args.exp,
                "dataset": args.dataset,
                "dataset_root": normalize_path_string(args.root_path),
                "split": split,
                "model": extra["model_name"],
                "architecture": extra["model_name"],
                "backbone_name": extra.get("backbone_name"),
                "student_name": extra.get("student_name"),
                "phase": phase,
                "model_info": model_info,
                "num_classes": args.num_classes,
                "in_channels": args.in_channels,
                "patch_size": list(args.patch_size),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_path": project_relative_path(checkpoint_path, PROJECT_ROOT),
            },
            result["average_metric"],
            result["case_metrics"],
            project_root=PROJECT_ROOT,
        )
        split_summaries[split] = summary
        visualization_samples_by_split[split] = list(result["visualization_samples"])
        macro = summary["metrics"]["macro_mean"]
        metrics_rows.append(
            {
                "experiment": args.exp,
                "dataset": args.dataset,
                "split": split,
                "phase": phase,
                "model_name": extra["model_name"],
                "backbone_name": extra.get("backbone_name"),
                "student_name": extra.get("student_name"),
                "dice": macro["dice"],
                "iou": macro["iou"],
                "hd95": macro["hd95"],
                "params": profile.get("params"),
                "trainable_params": profile.get("trainable_params"),
                "flops": profile.get("flops"),
                "fps": profile.get("fps"),
                "inference_time_seconds": profile.get("inference_time_seconds"),
                "evaluation_time_seconds": elapsed,
                "teacher_model": args.teacher_model,
                "prune_ratio": extra.get("prune_ratio"),
                "checkpoint_path": project_relative_path(checkpoint_path, PROJECT_ROOT),
            }
        )
        if result["visualization_samples"]:
            save_visualization_pdf(
                result["visualization_samples"],
                layout["reports_dir"] / f"{phase}_{split}_visualizations.pdf",
                title=f"{phase} | {split}",
            )
            save_visualization_overview_image(
                result["visualization_samples"],
                layout["visualization_dir"] / f"{phase}_{split}_visualizations.png",
            )
    if metrics_rows:
        write_metrics_rows(metrics_rows, layout["metrics_dir"] / f"{phase}_metrics.csv")
        save_performance_pdf(metrics_rows, layout["reports_dir"] / f"{phase}_performance.pdf", title=f"{phase} performance")
    return {
        "metrics_rows": metrics_rows,
        "split_summaries": split_summaries,
        "visualization_samples_by_split": visualization_samples_by_split,
        "phase": phase,
    }


def _build_teacher(in_channels: int):
    kwargs = {"mode": "train"}
    if args.teacher_model == "unetr":
        kwargs["image_size"] = tuple(args.patch_size)
    if args.teacher_model == "unet_resnet152":
        kwargs["encoder_pretrained"] = bool(args.encoder_pretrained)
    return net_factory(net_type=args.teacher_model, in_chns=in_channels, class_num=args.num_classes, **kwargs)


def _run_teacher(device: torch.device, image_mode: str, db_train, trainloader, valloader):
    run_dir = _phase_dir("teacher")
    layout = ensure_run_layout(run_dir)
    write_run_config(run_dir, vars(args))
    metadata = get_model_metadata(args.teacher_model)
    model = _build_teacher(db_train.in_channels).to(device)
    model_info = dict(metadata)
    model_info.update(extract_model_info(model))
    model_info["build_kwargs"] = {
        "net_type": args.teacher_model,
        "in_channels": db_train.in_channels,
        "num_classes": args.num_classes,
        "image_size": list(args.patch_size) if args.teacher_model == "unetr" else None,
        "encoder_pretrained": bool(args.encoder_pretrained) if args.teacher_model == "unet_resnet152" else None,
    }
    write_model_config(run_dir, model_info)
    teacher_signature = _teacher_signature(db_train.in_channels)
    history = {}
    best_path = None
    reused_from = None

    if not bool(args.force_retrain_teacher):
        explicit_match = None
        if args.teacher_checkpoint:
            explicit_match = find_compatible_checkpoint(
                [Path(args.teacher_checkpoint).expanduser().resolve()],
                expected_signature=teacher_signature,
            )
            if explicit_match is None:
                logging.warning("Ignore explicit teacher checkpoint because it is not compatible: %s", args.teacher_checkpoint)

        proposal_match = explicit_match or resolve_run_checkpoint(run_dir, expected_signature=teacher_signature)
        if proposal_match is not None:
            proposal_checkpoint_path = Path(proposal_match["checkpoint_path"])
            try:
                proposal_checkpoint_path.resolve().relative_to(run_dir.resolve())
                best_path = proposal_checkpoint_path
            except ValueError:
                copied = register_reused_checkpoint(
                    source_checkpoint_path=proposal_checkpoint_path,
                    target_run_dir=run_dir,
                    project_root=PROJECT_ROOT,
                    source_branch="external",
                    source_run_dir=proposal_checkpoint_path.parent.parent if proposal_checkpoint_path.parent.name == "checkpoints" else proposal_checkpoint_path.parent,
                    payload_updates={"phase": "teacher"},
                )
                best_path = Path(copied["best"]["checkpoint_path"])
                reused_from = "external"
            payload = load_checkpoint_into_model(best_path, model, device=device)
            history = payload.get("extra_state", {}).get("history", {})
            model_info.update(payload.get("model_info", {}))
            reused_from = reused_from or "proposal"
            logging.info("Loaded teacher checkpoint from %s: %s", reused_from, project_relative_path(best_path, PROJECT_ROOT))
        else:
            basic_match = resolve_basic_checkpoint(
                project_root=PROJECT_ROOT,
                output_root=args.output_root or None,
                model_name=args.teacher_model,
                dataset=args.dataset,
                expected_signature=teacher_signature,
            )
            if basic_match is not None:
                source_checkpoint_path = Path(basic_match["checkpoint_path"])
                copied = register_reused_checkpoint(
                    source_checkpoint_path=source_checkpoint_path,
                    target_run_dir=run_dir,
                    project_root=PROJECT_ROOT,
                    source_branch="basic",
                    source_run_dir=basic_match.get("run_dir"),
                    payload_updates={"phase": "teacher"},
                )
                best_path = Path(copied["best"]["checkpoint_path"])
                payload = load_checkpoint_into_model(best_path, model, device=device)
                history = payload.get("extra_state", {}).get("history", {})
                model_info.update(payload.get("model_info", {}))
                reused_from = "basic"
                logging.info(
                    "Reused teacher checkpoint from basic branch and registered it under proposal outputs: %s",
                    project_relative_path(source_checkpoint_path, PROJECT_ROOT),
                )

    if best_path is None:
        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(n_classes=args.num_classes)
        optimizer = optim.SGD(model.parameters(), lr=args.teacher_lr, momentum=0.9, weight_decay=1e-4)
        history = {"train_total_loss": [], "val_total_loss": [], "val_macro_dice": []}
        best_metric = float("-inf")
        best_path = None
        for epoch in tqdm(range(1, args.max_epochs_teacher + 1), desc="teacher", ncols=90):
            model.train()
            train_losses = []
            for batch in trainloader:
                _validate_label_batch(batch["label"], args.num_classes, batch)
                image = batch["image"].to(device)
                label = batch["label"].to(device)
                loss = _supervised_loss(extract_logits(model(image)), label, ce_loss, dice_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))
            val_result = evaluate_segmentation_dataset(
                model,
                valloader,
                device=device,
                num_classes=args.num_classes,
                patch_size=args.patch_size,
                output_dir=None,
                save_visualizations=False,
                vis_limit=0,
                sample_validator=_validate_label_batch,
            )
            val_dice = float(np.mean(val_result["average_metric"][:, 0]))
            val_loss = _compute_supervised_val_loss(model, valloader, device, ce_loss, dice_loss)
            history["train_total_loss"].append(float(np.mean(train_losses)) if train_losses else 0.0)
            history["val_total_loss"].append(val_loss)
            history["val_macro_dice"].append(val_dice)
            is_best = val_dice > best_metric
            if is_best:
                best_metric = val_dice
            checkpoint_path = save_checkpoint(
                run_dir,
                f"epoch_{epoch:03d}",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                metrics={"val_macro_dice": val_dice, "val_total_loss": val_loss},
                config=vars(args),
                model_info=model_info,
                phase="teacher",
                extra_state={"history": history},
                is_best=is_best,
                save_tagged_checkpoint=bool(args.save_history_checkpoints),
                project_root=PROJECT_ROOT,
            )
            if is_best:
                best_path = checkpoint_path
            logging.info(
                "Teacher epoch %d/%d | train_loss=%.6f | val_loss=%.6f | val_dice=%.6f",
                epoch,
                args.max_epochs_teacher,
                history["train_total_loss"][-1],
                val_loss,
                val_dice,
            )
        if best_path is None:
            best_path = resolve_phase_checkpoint(run_dir, "last")
        save_loss_pdf(history, layout["reports_dir"] / "teacher_loss.pdf", title="Teacher loss")
    else:
        _restore_loss_pdf_if_possible(history, layout["reports_dir"] / "teacher_loss.pdf", "Teacher loss")

    write_model_config(run_dir, model_info)
    profile = _compute_profile(model, device)
    evaluation_bundle = _export_phase_outputs(
        run_dir,
        model,
        best_path,
        "teacher",
        profile,
        {"model_name": metadata["model_name"], "backbone_name": metadata["backbone_name"], "student_name": None, "prune_ratio": None, "teacher_model": args.teacher_model},
        device,
        image_mode,
    )
    _export_model_channel_analysis(
        run_dir,
        model,
        checkpoint_path=best_path,
        device=device,
        prefix=None,
        title=f"Teacher Channel Analysis | {args.teacher_model}",
    )
    return {
        "model": model,
        "run_dir": run_dir,
        "checkpoint_path": best_path,
        "metadata": model_info,
        "history": history,
        "reused_from": reused_from,
        "evaluation_bundle": evaluation_bundle,
        "phase": "teacher",
    }


def _run_pruning(device: torch.device, image_mode: str, db_train, teacher_artifact: dict):
    run_dir = _phase_dir("pruning")
    layout = ensure_run_layout(run_dir)
    write_run_config(run_dir, vars(args))
    blueprint_path = layout["artifacts_dir"] / "blueprint.json"
    reusable_blueprint = None
    if blueprint_path.is_file() and not bool(args.force_reprune):
        candidate_blueprint = load_blueprint_artifact(blueprint_path)
        if (
            candidate_blueprint.get("teacher_model") == args.teacher_model
            and float(candidate_blueprint.get("prune_ratio", -1.0)) == float(args.prune_ratio)
        ):
            reusable_blueprint = candidate_blueprint
    if reusable_blueprint is not None:
        blueprint = reusable_blueprint
        logging.info("Loaded blueprint: %s", project_relative_path(blueprint_path, PROJECT_ROOT))
    else:
        blueprint = extract_pruned_blueprint(teacher_artifact["model"], prune_ratio=args.prune_ratio)
        blueprint.update(
            {
                "teacher_model": args.teacher_model,
                "teacher_checkpoint_path": project_relative_path(teacher_artifact["checkpoint_path"], PROJECT_ROOT),
                "teacher_run_dir": project_relative_path(teacher_artifact["run_dir"], PROJECT_ROOT),
                "student_name": "pdg_unet",
                "mapping_rule": "teacher_encoder -> student_channel_config",
            }
        )
        save_blueprint_artifact(blueprint, layout["artifacts_dir"])
    save_pruning_analysis_artifacts(layout["artifacts_dir"] / "pruning_analysis", blueprint, title="Teacher -> Student Pruning Analysis")
    _write_json(layout["configs_dir"] / "pruning_config.json", blueprint)
    _write_json(layout["metrics_dir"] / "pruning_summary.json", blueprint)
    global_summary = blueprint.get("global_pruning_summary", {})
    logging.info(
        "Global pruning summary | layers=%s | channels_before=%s | channels_after=%s | pruned=%s | ratio=%.4f",
        global_summary.get("num_layers_analyzed"),
        global_summary.get("total_channels_before"),
        global_summary.get("total_channels_after"),
        global_summary.get("total_channels_pruned"),
        global_summary.get("global_prune_ratio", 0.0),
    )
    for row in blueprint.get("teacher_vs_student_rows", []):
        logging.info(
            "Pruning layer %s | %s -> %s | pruned=%s | ratio=%.4f",
            row["layer_name"],
            row["teacher_out_channels"],
            row["student_out_channels"],
            row["channels_pruned"],
            row["actual_prune_ratio"],
        )

    pruned_student = PDGUNet(
        in_channels=db_train.in_channels,
        num_classes=args.num_classes,
        channel_config=tuple(blueprint["channel_config"]),
    ).to(device)
    weight_transfer = _copy_exact_matching_weights(teacher_artifact["model"], pruned_student)
    pruning_model_info = extract_model_info(pruned_student)
    pruning_model_info.update(
        {
            "branch": "proposal",
            "phase_name": "pruning",
            "teacher_model": args.teacher_model,
            "teacher_backbone_name": teacher_artifact["metadata"].get("backbone_name"),
            "student_name": "pruned_student_init",
            "blueprint_path": project_relative_path(blueprint_path, PROJECT_ROOT),
            "blueprint": blueprint,
            "weight_transfer": weight_transfer,
            "evaluation_note": "This phase evaluates the pruned student immediately after pruning/baseline initialization, before student tuning.",
            "build_kwargs": {
                "in_channels": db_train.in_channels,
                "num_classes": args.num_classes,
                "channel_config": list(blueprint["channel_config"]),
            },
        }
    )
    write_model_config(run_dir, pruning_model_info)

    checkpoint_path = save_checkpoint(
        run_dir,
        "pruning_init",
        model=pruned_student,
        epoch=0,
        global_step=0,
        best_metric=None,
        metrics={},
        config=vars(args),
        model_info=pruning_model_info,
        phase="pruning",
        extra_state={"blueprint": blueprint, "weight_transfer": weight_transfer},
        is_best=True,
        save_tagged_checkpoint=False,
        project_root=PROJECT_ROOT,
    )

    profile = _compute_profile(pruned_student, device)
    evaluation_bundle = _export_phase_outputs(
        run_dir,
        pruned_student,
        checkpoint_path,
        "pruning",
        profile,
        {
            "model_name": "pdg_unet",
            "backbone_name": teacher_artifact["metadata"].get("backbone_name"),
            "student_name": "pruned_student_init",
            "prune_ratio": args.prune_ratio,
            "teacher_model": args.teacher_model,
        },
        device,
        image_mode,
    )
    _export_model_channel_analysis(
        run_dir,
        pruned_student,
        checkpoint_path=checkpoint_path,
        device=device,
        prefix="pruned_student",
        title="Pruned Student Before Tuning",
    )
    return {
        "run_dir": run_dir,
        "blueprint": blueprint,
        "blueprint_path": blueprint_path,
        "model": pruned_student,
        "checkpoint_path": checkpoint_path,
        "metadata": pruning_model_info,
        "evaluation_bundle": evaluation_bundle,
        "phase": "pruning",
    }


def _compute_student_val_losses(student, teacher, valloader, device, criterion):
    student.eval()
    teacher.eval()
    tracked = {key: [] for key in ("total_loss", "segmentation_loss", "distillation_loss", "sparsity_loss")}
    with torch.no_grad():
        for batch in valloader:
            _validate_label_batch(batch["label"], args.num_classes, batch)
            image = F.interpolate(batch["image"].to(device), size=args.patch_size, mode="bilinear", align_corners=False)
            label = F.interpolate(batch["label"].unsqueeze(1).float().to(device), size=args.patch_size, mode="nearest").squeeze(1).long()
            loss_dict = criterion(student(image), teacher(image), student.get_gate_tensors(), label)
            for key in tracked:
                tracked[key].append(float(loss_dict[key].item()))
    student.train()
    return {key: float(np.mean(values)) if values else 0.0 for key, values in tracked.items()}


def _run_student(device: torch.device, image_mode: str, db_train, trainloader, valloader, teacher_artifact: dict, pruning_artifact: dict):
    run_dir = _phase_dir("student")
    layout = ensure_run_layout(run_dir)
    write_run_config(run_dir, vars(args))
    teacher = teacher_artifact["model"]
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student = PDGUNet(in_channels=db_train.in_channels, num_classes=args.num_classes, channel_config=tuple(pruning_artifact["blueprint"]["channel_config"])).to(device)
    student_model_info = extract_model_info(student)
    student_model_info.update(
        {
            "branch": "proposal",
            "teacher_model": args.teacher_model,
            "teacher_backbone_name": teacher_artifact["metadata"].get("backbone_name"),
            "student_name": "gated_student",
            "blueprint_path": project_relative_path(pruning_artifact["blueprint_path"], PROJECT_ROOT),
            "blueprint": pruning_artifact["blueprint"],
            "build_kwargs": {
                "in_channels": db_train.in_channels,
                "num_classes": args.num_classes,
                "channel_config": list(pruning_artifact["blueprint"]["channel_config"]),
            },
        }
    )
    write_model_config(run_dir, student_model_info)
    student_input_analysis = _export_model_channel_analysis(
        run_dir,
        student,
        checkpoint_path=None,
        device=device,
        prefix="student_input",
        title="Student Input Architecture Before Tuning",
    )
    history = {}
    reusable_match = None
    if not bool(args.force_retrain_student):
        reusable_match = resolve_run_checkpoint(
            run_dir,
            expected_signature=_student_signature(pruning_artifact["blueprint"]["channel_config"], db_train.in_channels),
        )
    if reusable_match is not None:
        best_path = Path(reusable_match["checkpoint_path"])
        payload = load_checkpoint_into_model(best_path, student, device=device)
        history = payload.get("extra_state", {}).get("history", {})
        student_model_info.update(payload.get("model_info", {}))
        logging.info("Loaded student checkpoint: %s", project_relative_path(best_path, PROJECT_ROOT))
        _restore_loss_pdf_if_possible(history, layout["reports_dir"] / "student_loss.pdf", "Student loss")
    else:
        criterion = CompressionLoss(args.num_classes, args.lambda_distill, args.lambda_sparsity)
        optimizer = optim.AdamW(student.parameters(), lr=args.student_lr, weight_decay=1e-4)
        history = {
            "train_total_loss": [],
            "train_segmentation_loss": [],
            "train_distillation_loss": [],
            "train_sparsity_loss": [],
            "val_total_loss": [],
            "val_segmentation_loss": [],
            "val_distillation_loss": [],
            "val_sparsity_loss": [],
            "val_macro_dice": [],
        }
        best_metric = float("-inf")
        best_path = None
        for epoch in tqdm(range(1, args.max_epochs_student + 1), desc="student", ncols=90):
            student.train()
            tracked = {key: [] for key in ("total_loss", "segmentation_loss", "distillation_loss", "sparsity_loss")}
            for batch in trainloader:
                _validate_label_batch(batch["label"], args.num_classes, batch)
                image = batch["image"].to(device)
                label = batch["label"].to(device)
                with torch.no_grad():
                    teacher_output = teacher(image)
                loss_dict = criterion(student(image), teacher_output, student.get_gate_tensors(), label)
                optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                optimizer.step()
                for key in tracked:
                    tracked[key].append(float(loss_dict[key].item()))
            val_metrics = evaluate_segmentation_dataset(
                student,
                valloader,
                device=device,
                num_classes=args.num_classes,
                patch_size=args.patch_size,
                output_dir=None,
                save_visualizations=False,
                vis_limit=0,
                sample_validator=_validate_label_batch,
            )
            val_losses = _compute_student_val_losses(student, teacher, valloader, device, criterion)
            val_dice = float(np.mean(val_metrics["average_metric"][:, 0]))
            history["train_total_loss"].append(float(np.mean(tracked["total_loss"])) if tracked["total_loss"] else 0.0)
            history["train_segmentation_loss"].append(float(np.mean(tracked["segmentation_loss"])) if tracked["segmentation_loss"] else 0.0)
            history["train_distillation_loss"].append(float(np.mean(tracked["distillation_loss"])) if tracked["distillation_loss"] else 0.0)
            history["train_sparsity_loss"].append(float(np.mean(tracked["sparsity_loss"])) if tracked["sparsity_loss"] else 0.0)
            history["val_total_loss"].append(val_losses["total_loss"])
            history["val_segmentation_loss"].append(val_losses["segmentation_loss"])
            history["val_distillation_loss"].append(val_losses["distillation_loss"])
            history["val_sparsity_loss"].append(val_losses["sparsity_loss"])
            history["val_macro_dice"].append(val_dice)
            is_best = val_dice > best_metric
            if is_best:
                best_metric = val_dice
            checkpoint_path = save_checkpoint(
                run_dir,
                f"epoch_{epoch:03d}",
                model=student,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_metric,
                metrics={"val_macro_dice": val_dice, "val_total_loss": val_losses["total_loss"]},
                config=vars(args),
                model_info=student_model_info,
                phase="student",
                extra_state={"history": history, "blueprint": pruning_artifact["blueprint"]},
                is_best=is_best,
                save_tagged_checkpoint=bool(args.save_history_checkpoints),
                project_root=PROJECT_ROOT,
            )
            if is_best:
                best_path = checkpoint_path
            logging.info(
                "Student epoch %d/%d | train_total=%.6f | val_total=%.6f | val_dice=%.6f",
                epoch,
                args.max_epochs_student,
                history["train_total_loss"][-1],
                val_losses["total_loss"],
                val_dice,
            )
        if best_path is None:
            best_path = resolve_phase_checkpoint(run_dir, "last")
        save_loss_pdf(history, layout["reports_dir"] / "student_loss.pdf", title="Student loss")
    write_model_config(run_dir, student_model_info)
    profile = _compute_profile(student, device)
    evaluation_bundle = _export_phase_outputs(
        run_dir,
        student,
        best_path,
        "student",
        profile,
        {"model_name": "pdg_unet", "backbone_name": teacher_artifact["metadata"].get("backbone_name"), "student_name": "gated_student", "prune_ratio": args.prune_ratio, "teacher_model": args.teacher_model},
        device,
        image_mode,
    )
    student_final_analysis = _export_model_channel_analysis(
        run_dir,
        student,
        checkpoint_path=best_path,
        device=device,
        prefix="student_final",
        title="Student Final Architecture After Tuning",
    )
    student_comparison_rows = build_analysis_comparison(
        student_input_analysis,
        student_final_analysis,
        before_label="input",
        after_label="final",
    )
    save_comparison_artifacts(
        run_dir / "artifacts" / "channel_analysis",
        student_comparison_rows,
        prefix="student_tuning_comparison",
        title="Student Tuning Comparison",
        extra_report={
            "global_summary": {
                "input_channel_layers": student_input_analysis["global_summary"]["num_channel_layers"],
                "final_channel_layers": student_final_analysis["global_summary"]["num_channel_layers"],
                "input_total_channels": student_input_analysis["global_summary"]["total_output_channels"],
                "final_total_channels": student_final_analysis["global_summary"]["total_output_channels"],
                "input_gate_layers": student_input_analysis["global_summary"]["num_gate_layers"],
                "final_gate_layers": student_final_analysis["global_summary"]["num_gate_layers"],
            }
        },
    )
    student_final_exports = _export_student_final_shortcuts(run_dir, _proposal_root_dir())
    return {
        "model": student,
        "run_dir": run_dir,
        "checkpoint_path": best_path,
        "metadata": student_model_info,
        "student_input_analysis": student_input_analysis,
        "student_final_analysis": student_final_analysis,
        "student_final_exports": student_final_exports,
        "evaluation_bundle": evaluation_bundle,
        "phase": "student",
    }


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_mode = "grayscale" if args.in_channels == 1 else "rgb"

    proposal_root_dir = _proposal_root_dir()
    proposal_root_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir = _phase_dir("pipeline")
    ensure_run_layout(pipeline_dir)
    write_run_config(pipeline_dir, vars(args))
    _configure_logging(pipeline_dir / "run.log")
    logging.info("PDG pipeline args: %s", args)

    db_train, db_val, trainloader, valloader = _build_loaders(device, image_mode)
    logging.info("Dataset summary | train=%d | val=%d", len(db_train), len(db_val))

    teacher_artifact = _run_teacher(device, image_mode, db_train, trainloader, valloader)
    pruning_artifact = _run_pruning(device, image_mode, db_train, teacher_artifact)
    student_artifact = _run_student(device, image_mode, db_train, trainloader, valloader, teacher_artifact, pruning_artifact)
    pipeline_export = _save_pipeline_outputs(pipeline_dir, teacher_artifact, pruning_artifact, student_artifact)

    _write_json(
        pipeline_dir / "pipeline_summary.json",
        {
            "proposal_root_dir": project_relative_path(proposal_root_dir, PROJECT_ROOT),
            "teacher_checkpoint": project_relative_path(teacher_artifact["checkpoint_path"], PROJECT_ROOT),
            "teacher_model_info": teacher_artifact["metadata"],
            "pruning_blueprint_path": project_relative_path(pruning_artifact["blueprint_path"], PROJECT_ROOT),
            "pruning_blueprint": pruning_artifact["blueprint"],
            "pruning_checkpoint": project_relative_path(pruning_artifact["checkpoint_path"], PROJECT_ROOT),
            "pruning_model_info": pruning_artifact["metadata"],
            "student_checkpoint": project_relative_path(student_artifact["checkpoint_path"], PROJECT_ROOT),
            "student_model_info": student_artifact["metadata"],
            "teacher_run_dir": project_relative_path(teacher_artifact["run_dir"], PROJECT_ROOT),
            "pruning_run_dir": project_relative_path(pruning_artifact["run_dir"], PROJECT_ROOT),
            "student_run_dir": project_relative_path(student_artifact["run_dir"], PROJECT_ROOT),
            "student_final_exports": student_artifact.get("student_final_exports", {}),
            "teacher_reused_from": teacher_artifact.get("reused_from"),
            "pipeline_evaluation": pipeline_export.get("summary", {}),
        },
    )
    logging.info("PDG pipeline completed.")
