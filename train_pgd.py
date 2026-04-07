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
from utils.checkpoints import load_checkpoint_into_model, resolve_phase_checkpoint, save_checkpoint
from utils.compression_loss import CompressionLoss
from utils.evaluation import build_evaluation_output_dir, evaluate_segmentation_dataset, save_evaluation_artifacts
from utils.experiment import build_run_dir, normalize_path_string, project_relative_path, ensure_run_layout, write_model_config, write_run_config
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


def _phase_dir(phase: str, variant: str) -> Path:
    return build_run_dir(
        project_root=PROJECT_ROOT,
        experiment=args.exp,
        dataset=args.dataset,
        model_name="pdg_unet",
        phase=f"_{phase}",
        variant=variant,
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


def _export_phase_outputs(run_dir: Path, model, checkpoint_path: Path, phase: str, profile: dict, extra: dict, device: torch.device, image_mode: str) -> None:
    load_checkpoint_into_model(checkpoint_path, model, device=device)
    layout = ensure_run_layout(run_dir)
    metrics_rows = []
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


def _build_teacher(in_channels: int):
    kwargs = {"mode": "train"}
    if args.teacher_model == "unetr":
        kwargs["image_size"] = tuple(args.patch_size)
    if args.teacher_model == "unet_resnet152":
        kwargs["encoder_pretrained"] = bool(args.encoder_pretrained)
    return net_factory(net_type=args.teacher_model, in_chns=in_channels, class_num=args.num_classes, **kwargs)


def _run_teacher(device: torch.device, image_mode: str, db_train, trainloader, valloader):
    run_dir = _phase_dir("teacher", args.teacher_model)
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
    reusable = Path(args.teacher_checkpoint).expanduser().resolve() if args.teacher_checkpoint else resolve_phase_checkpoint(run_dir, "best")
    if reusable and reusable.is_file() and not bool(args.force_retrain_teacher):
        payload = load_checkpoint_into_model(reusable, model, device=device)
        history = payload.get("extra_state", {}).get("history", {})
        logging.info("Loaded teacher checkpoint: %s", project_relative_path(reusable, PROJECT_ROOT))
        best_path = Path(reusable)
    else:
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
    profile = _compute_profile(model, device)
    _export_phase_outputs(
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
    return {"model": model, "run_dir": run_dir, "checkpoint_path": best_path, "metadata": model_info}


def _run_pruning(teacher_artifact: dict):
    run_dir = _phase_dir("pruning", f"{args.teacher_model}_ratio_{args.prune_ratio:.2f}")
    layout = ensure_run_layout(run_dir)
    write_run_config(run_dir, vars(args))
    blueprint_path = layout["artifacts_dir"] / "blueprint.json"
    if blueprint_path.is_file() and not bool(args.force_reprune):
        blueprint = load_blueprint_artifact(blueprint_path)
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
    return {"run_dir": run_dir, "blueprint": blueprint, "blueprint_path": blueprint_path}


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
    run_dir = _phase_dir("student", f"{args.teacher_model}_ratio_{args.prune_ratio:.2f}")
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
    reusable = resolve_phase_checkpoint(run_dir, "best")
    if reusable and reusable.is_file() and not bool(args.force_retrain_student):
        payload = load_checkpoint_into_model(reusable, student, device=device)
        history = payload.get("extra_state", {}).get("history", {})
        logging.info("Loaded student checkpoint: %s", project_relative_path(reusable, PROJECT_ROOT))
        best_path = Path(reusable)
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
    profile = _compute_profile(student, device)
    _export_phase_outputs(
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
    return {
        "model": student,
        "run_dir": run_dir,
        "checkpoint_path": best_path,
        "metadata": student_model_info,
        "student_input_analysis": student_input_analysis,
        "student_final_analysis": student_final_analysis,
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

    pipeline_dir = _phase_dir("pipeline", args.teacher_model)
    ensure_run_layout(pipeline_dir)
    write_run_config(pipeline_dir, vars(args))
    _configure_logging(pipeline_dir / "run.log")
    logging.info("PDG pipeline args: %s", args)

    db_train, db_val, trainloader, valloader = _build_loaders(device, image_mode)
    logging.info("Dataset summary | train=%d | val=%d", len(db_train), len(db_val))

    teacher_artifact = _run_teacher(device, image_mode, db_train, trainloader, valloader)
    pruning_artifact = _run_pruning(teacher_artifact)
    student_artifact = _run_student(device, image_mode, db_train, trainloader, valloader, teacher_artifact, pruning_artifact)

    _write_json(
        pipeline_dir / "pipeline_summary.json",
        {
            "teacher_checkpoint": project_relative_path(teacher_artifact["checkpoint_path"], PROJECT_ROOT),
            "teacher_model_info": teacher_artifact["metadata"],
            "pruning_blueprint_path": project_relative_path(pruning_artifact["blueprint_path"], PROJECT_ROOT),
            "pruning_blueprint": pruning_artifact["blueprint"],
            "student_checkpoint": project_relative_path(student_artifact["checkpoint_path"], PROJECT_ROOT),
            "student_model_info": student_artifact["metadata"],
            "teacher_run_dir": project_relative_path(teacher_artifact["run_dir"], PROJECT_ROOT),
            "pruning_run_dir": project_relative_path(pruning_artifact["run_dir"], PROJECT_ROOT),
            "student_run_dir": project_relative_path(student_artifact["run_dir"], PROJECT_ROOT),
        },
    )
    logging.info("PDG pipeline completed.")
