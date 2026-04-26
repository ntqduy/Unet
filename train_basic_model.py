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
from networks.net_factory import get_model_metadata, list_models, net_factory
from utils.checkpoints import load_checkpoint_into_model, save_checkpoint
from utils.checkpoint_resolver import build_expected_signature, register_reused_checkpoint, resolve_basic_checkpoint
from utils import losses, val_2d
from utils.channel_analysis import extract_channel_analysis, save_channel_analysis_artifacts
from utils.evaluation import (
    build_evaluation_output_dir,
    evaluate_segmentation_dataset,
    save_evaluation_artifacts,
)
from utils.experiment import build_basic_run_dir, ensure_run_layout, normalize_path_string, project_relative_path, sanitize_tag, write_model_config, write_run_config
from utils.model_output import extract_logits, extract_model_info
from utils.profiling import benchmark_inference, count_parameters, maybe_compute_flops
from utils.reporting import save_loss_pdf, save_performance_pdf, save_visualization_overview_image, save_visualization_pdf, write_metrics_rows
from utils.visualization import save_triplet_visualization


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "Kvasir-SEG"



parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=str(DEFAULT_DATA_ROOT), help="dataset root path")
parser.add_argument("--dataset", type=str, default="kvasir_seg", choices=list_available_datasets(), help="dataset name")
parser.add_argument("--exp", type=str, default="supervised", help="experiment name")
parser.add_argument("--model", type=str, default="unet", choices=list_models(), help="model name")
parser.add_argument("--train_split", type=str, default="train", choices=["train", "val", "test"], help="training split")
parser.add_argument("--val_split", type=str, default="val", choices=["train", "val", "test"], help="evaluation split")
parser.add_argument("--max_epochs", type=int, default=None, help="number of training epochs; overrides max_iterations when set")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum training iterations for legacy iteration-based training")
parser.add_argument("--eval_interval", type=int, default=20, help="validation interval in iterations for iteration-based training")
parser.add_argument("--eval_interval_epochs", type=int, default=1, help="validation interval in epochs for epoch-based training")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256])
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--in_channels", type=int, default=3, help="number of image channels to load")
parser.add_argument("--encoder_pretrained", type=int, default=1, help="only used by unet_resnet152; defaults to 1")
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--save_visualizations", type=int, default=1)
parser.add_argument("--vis_num_samples", type=int, default=5)
parser.add_argument("--output_root", type=str, default="", help="root directory for all exported outputs; defaults to PROJECT_ROOT/outputs")
parser.add_argument("--save_history_checkpoints", type=int, default=0, help="set to 1 to keep epoch/iteration checkpoint history in addition to best.pth")
parser.add_argument("--save_last_checkpoint", type=int, default=1, help="set to 1 to keep an overwritten last.pth checkpoint in addition to best.pth")
parser.add_argument("--save_optimizer_state", type=int, default=0, help="set to 1 to include optimizer/scheduler/scaler states in saved checkpoints")
parser.add_argument("--force_retrain", type=int, default=0, help="set to 1 to ignore existing compatible checkpoints and train again")
parser.add_argument(
    "--final_eval_splits",
    nargs="*",
    default=["train", "val", "test"],
    help="splits to evaluate after training with the best checkpoint",
)

args = parser.parse_args()
for _flag_name in ("save_history_checkpoints", "save_last_checkpoint", "save_optimizer_state"):
    if int(getattr(args, _flag_name)) not in (0, 1):
        parser.error(f"--{_flag_name} must be 0 or 1.")
    setattr(args, _flag_name, int(getattr(args, _flag_name)))
dice_loss = losses.DiceLoss(n_classes=args.num_classes)


def save_validation_visualizations(vis_samples, output_dir):
    output_dir = Path(output_dir)
    for sample in vis_samples:
        save_triplet_visualization(
            image=sample["image"],
            label=sample["label"],
            prediction=sample["prediction"],
            output_dir=output_dir,
            case_name=sample["case"],
        )


def validate_label_batch(label_batch, num_classes, sampled_batch):
    label_batch = label_batch.long()
    invalid_mask = (label_batch < 0) | (label_batch >= num_classes)
    if not invalid_mask.any():
        return

    unique_values = torch.unique(label_batch.cpu()).tolist()
    invalid_values = torch.unique(label_batch[invalid_mask].cpu()).tolist()
    case_names = sampled_batch.get("case", [])
    label_paths = sampled_batch.get("label_path", [])

    if not isinstance(case_names, (list, tuple)):
        case_names = [case_names]
    if not isinstance(label_paths, (list, tuple)):
        label_paths = [label_paths]

    raise ValueError(
        f"Found mask values outside [0, {num_classes - 1}] in the current batch. "
        f"Invalid values: {invalid_values[:20]}. "
        f"Unique label values: {unique_values[:20]}. "
        f"Cases: {[str(value) for value in case_names[:4]]}. "
        f"Label paths: {[str(value) for value in label_paths[:2]]}. "
        "This usually means binary masks were not normalized to 0/1 before training."
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _write_timing_sidecars(run_dir: Path, timing_rows: list[dict]) -> None:
    if not timing_rows:
        return
    layout = ensure_run_layout(run_dir)
    write_metrics_rows(timing_rows, layout["metrics_dir"] / "timing_summary.csv")
    write_metrics_rows(timing_rows, run_dir / "timing_summary.csv")
    payload = {
        "timings": timing_rows,
        "total_time_seconds": float(sum(float(row.get("total_time_seconds") or 0.0) for row in timing_rows)),
    }
    _write_json(layout["metrics_dir"] / "timing_summary.json", payload)
    _write_json(run_dir / "timing_summary.json", payload)


def _write_inference_summary(run_dir: Path, metrics_rows: list[dict]) -> None:
    if not metrics_rows:
        return
    inference_rows = [
        {
            "dataset": row.get("dataset"),
            "split": row.get("split"),
            "phase": row.get("phase"),
            "model_name": row.get("model_name"),
            "fps": row.get("fps"),
            "inference_time_seconds": row.get("inference_time_seconds"),
            "evaluation_time_seconds": row.get("evaluation_time_seconds"),
            "checkpoint_path": row.get("checkpoint_path"),
        }
        for row in metrics_rows
    ]
    write_metrics_rows(inference_rows, run_dir / "inference_summary.csv")
    write_metrics_rows(inference_rows, run_dir / "metrics" / "inference_summary.csv")


def _expected_basic_checkpoint_signature(in_channels: int) -> dict:
    return build_expected_signature(
        dataset=args.dataset,
        model_name=args.model,
        num_classes=args.num_classes,
        in_channels=in_channels,
        patch_size=args.patch_size,
        encoder_pretrained=bool(args.encoder_pretrained) if args.model == "unet_resnet152" else None,
    )


def _restore_loss_report_from_history(snapshot_path: Path, history: dict | None) -> None:
    if not history:
        return
    if any(values for values in history.values() if isinstance(values, list)):
        save_loss_pdf(history, snapshot_path / "reports" / "basic_loss.pdf", title="Basic model loss")


def _compute_model_profile(model, device) -> dict:
    input_shape = (1, args.in_channels, args.patch_size[0], args.patch_size[1])
    profile = {
        "params": int(count_parameters(model)),
        "trainable_params": int(count_parameters(model, trainable_only=True)),
    }
    profile.update(benchmark_inference(model, input_shape=input_shape, device=device))
    profile["flops"] = maybe_compute_flops(model, input_shape=input_shape, device=device)
    return profile


def _unique_preserve_order(values):
    ordered = []
    seen = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _build_evaluation_metadata(split: str, checkpoint_path: Path, model_info: dict | None = None) -> dict:
    resolved_model_info = dict(model_info or get_model_metadata(args.model))
    return {
        "experiment": args.exp,
        "dataset": args.dataset,
        "dataset_root": normalize_path_string(args.root_path),
        "split": split,
        "model": args.model,
        "architecture": args.model,
        "backbone_name": resolved_model_info.get("backbone_name"),
        "student_name": resolved_model_info.get("student_name"),
        "model_info": resolved_model_info,
        "num_classes": args.num_classes,
        "in_channels": args.in_channels,
        "patch_size": list(args.patch_size),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_path": project_relative_path(checkpoint_path, PROJECT_ROOT),
    }


def _write_evaluation_overview(snapshot_path: Path, checkpoint_path: Path, split_summaries: dict, model_info: dict | None = None) -> None:
    if not split_summaries:
        return

    checkpoint_root = build_evaluation_output_dir(snapshot_path, args.dataset, args.model, checkpoint_path, next(iter(split_summaries))).parent
    overview = {
        "experiment": args.exp,
        "dataset": args.dataset,
        "dataset_root": normalize_path_string(args.root_path),
        "model": args.model,
        "model_info": model_info or get_model_metadata(args.model),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_path": project_relative_path(checkpoint_path, PROJECT_ROOT),
        "splits": split_summaries,
    }
    _write_json(checkpoint_root / "evaluation_overview.json", overview)

    lines = [
        f"# Final Evaluation Overview | {args.dataset} | {args.model}",
        "",
        f"- Experiment: `{args.exp}`",
        f"- Dataset root: `{normalize_path_string(args.root_path)}`",
        f"- Checkpoint: `{checkpoint_path.name}`",
        f"- Checkpoint path: `{project_relative_path(checkpoint_path, PROJECT_ROOT)}`",
        "",
        "## Splits",
        "",
    ]
    for split_name, summary in split_summaries.items():
        macro = summary["metrics"]["macro_mean"]
        lines.append(
            f"- `{split_name}` | dice `{macro['dice']:.6f}` | iou `{macro['iou']:.6f}` | hd95 `{macro['hd95']:.6f}` | cases `{summary['num_cases']}`"
        )
    with (checkpoint_root / "evaluation_overview.md").open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def run_validation(args, model, valloader, device, ce_loss):
    model.eval()
    metric_list = []
    vis_samples = []
    val_losses = []

    with torch.no_grad():
        for sampled_val in valloader:
            validate_label_batch(sampled_val["label"], args.num_classes, sampled_val)
            resized_image = F.interpolate(sampled_val["image"].to(device), size=args.patch_size, mode="bilinear", align_corners=False)
            resized_label = F.interpolate(
                sampled_val["label"].unsqueeze(1).float().to(device),
                size=args.patch_size,
                mode="nearest",
            ).squeeze(1).long()
            logits = extract_logits(model(resized_image))
            loss_ce = ce_loss(logits, resized_label.long())
            outputs_soft = torch.softmax(logits, dim=1)
            loss_dice = dice_loss(outputs_soft, resized_label.unsqueeze(1))
            val_losses.append(float((0.5 * (loss_ce + loss_dice)).item()))

            need_prediction = bool(args.save_visualizations) and len(vis_samples) < args.vis_num_samples
            metric_output = val_2d.test_single_volume(
                sampled_val["image"],
                sampled_val["label"],
                model,
                classes=args.num_classes,
                patch_size=args.patch_size,
                device=device,
                return_prediction=need_prediction,
            )

            if need_prediction:
                metric_i, prediction = metric_output
                case_name = sampled_val["case"][0] if isinstance(sampled_val["case"], (list, tuple)) else str(sampled_val["case"])
                vis_samples.append(
                    {
                        "case": case_name,
                        "image": sampled_val["image"][0],
                        "label": sampled_val["label"][0],
                        "prediction": prediction[0],
                    }
                )
            else:
                metric_i = metric_output

            metric_list.append(np.array(metric_i))

    if not metric_list:
        raise ValueError("Validation dataset is empty; cannot compute metrics.")

    metric_array = np.stack(metric_list, axis=0).mean(axis=0)
    performance = float(np.mean(metric_array[:, 0]))
    return performance, float(np.mean(val_losses)) if val_losses else 0.0, vis_samples


def _run_final_evaluations(snapshot_path: Path, checkpoint_path: Path, model, device, image_mode: str, profile: dict, model_info: dict, training_time_seconds: float = 0.0) -> list[dict]:
    logging.info("Running final evaluation with best checkpoint: %s", project_relative_path(checkpoint_path, PROJECT_ROOT))
    load_checkpoint_into_model(checkpoint_path, model, device=device)
    model.eval()

    eval_transform = transforms.Compose([Normalize(), ToTensor()])
    split_summaries = {}
    metrics_rows = []
    for split in _unique_preserve_order(args.final_eval_splits):
        try:
            dataset = build_dataset(
                dataset_name=args.dataset,
                base_dir=args.root_path,
                split=split,
                transform=eval_transform,
                image_mode=image_mode,
            )
        except Exception as error:
            logging.warning("Skipping final evaluation for split '%s': %s", split, error)
            continue

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        output_dir = build_evaluation_output_dir(snapshot_path, args.dataset, args.model, checkpoint_path, split)
        start_time = time.perf_counter()
        result = evaluate_segmentation_dataset(
            model,
            dataloader,
            device=device,
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            output_dir=output_dir,
            save_visualizations=bool(args.save_visualizations),
            vis_limit=args.vis_num_samples,
            sample_validator=validate_label_batch,
            progress_desc=f"final-{split}",
        )
        elapsed = time.perf_counter() - start_time
        summary = save_evaluation_artifacts(
            output_dir,
            _build_evaluation_metadata(split, checkpoint_path, model_info=model_info),
            result["average_metric"],
            result["case_metrics"],
            project_root=PROJECT_ROOT,
        )
        split_summaries[split] = summary
        macro = summary["metrics"]["macro_mean"]
        metrics_rows.append(
            {
                "experiment": args.exp,
                "dataset": args.dataset,
                "split": split,
                "phase": "basic",
                "model_name": args.model,
                "backbone_name": get_model_metadata(args.model)["backbone_name"],
                "student_name": None,
                "dice": macro["dice"],
                "iou": macro["iou"],
                "hd95": macro["hd95"],
                "params": profile.get("params"),
                "trainable_params": profile.get("trainable_params"),
                "flops": profile.get("flops"),
                "fps": profile.get("fps"),
                "inference_time_seconds": profile.get("inference_time_seconds"),
                "Inf (s)": profile.get("inference_time_seconds"),
                "evaluation_time_seconds": elapsed,
                "training_time_seconds": float(training_time_seconds),
                "search_time_seconds": 0.0,
                "Search Time (s)": 0.0,
                "checkpoint_path": project_relative_path(checkpoint_path, PROJECT_ROOT),
            }
        )
        logging.info(
            "Final evaluation | split=%s | checkpoint=%s | macro_dice=%.6f | macro_iou=%.6f | macro_hd95=%.6f | cases=%d",
            split,
            checkpoint_path.name,
            macro["dice"],
            macro["iou"],
            macro["hd95"],
            summary["num_cases"],
        )
        if result["visualization_samples"]:
            save_visualization_pdf(
                result["visualization_samples"],
                snapshot_path / "reports" / f"basic_{split}_visualizations.pdf",
                title=f"basic | {split}",
            )
            save_visualization_overview_image(
                result["visualization_samples"],
                snapshot_path / "artifacts" / "visualizations" / f"basic_{split}_visualizations.png",
            )

    _write_evaluation_overview(snapshot_path, checkpoint_path, split_summaries, model_info)
    if metrics_rows:
        write_metrics_rows(metrics_rows, snapshot_path / "metrics" / "basic_metrics.csv")
        write_metrics_rows(metrics_rows, snapshot_path / "metrics_summary.csv")
        _write_inference_summary(snapshot_path, metrics_rows)
        save_performance_pdf(metrics_rows, snapshot_path / "reports" / "basic_performance.pdf", title="Basic model performance")
    return metrics_rows


def _export_basic_channel_analysis(snapshot_path: Path, checkpoint_path: Path, model, device) -> None:
    load_checkpoint_into_model(checkpoint_path, model, device=device)
    analysis = extract_channel_analysis(model, importance_name="filter_l1")
    save_channel_analysis_artifacts(
        snapshot_path / "artifacts" / "channel_analysis",
        analysis,
        title=f"Basic Channel Analysis | {args.model}",
    )
    logging.info(
        "Basic channel analysis saved | layers=%d | total_channels=%d",
        analysis["global_summary"]["num_channel_layers"],
        analysis["global_summary"]["total_output_channels"],
    )


def train(args, snapshot_path):
    phase_start_time = time.perf_counter()
    snapshot_path = Path(snapshot_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_mode = "grayscale" if args.in_channels == 1 else "rgb"

    train_transform = transforms.Compose([Normalize(), RandomGenerator(args.patch_size)])
    eval_transform = transforms.Compose([Normalize(), ToTensor()])

    db_train = build_dataset(
        dataset_name=args.dataset,
        base_dir=args.root_path,
        split=args.train_split,
        transform=train_transform,
        image_mode=image_mode,
    )
    db_val = build_dataset(
        dataset_name=args.dataset,
        base_dir=args.root_path,
        split=args.val_split,
        transform=eval_transform,
        image_mode=image_mode,
    )
    logging.info(
        "Dataset sizes | train=%d (%s) | val=%d (%s)",
        len(db_train),
        args.train_split,
        len(db_val),
        args.val_split,
    )
    logging.info(
        "Split manifests | train=%s | val=%s",
        project_relative_path(getattr(db_train, "manifest_path", None), PROJECT_ROOT),
        project_relative_path(getattr(db_val, "manifest_path", None), PROJECT_ROOT),
    )
    logging.info(
        "Binary mask normalization | train: %s | val: %s",
        getattr(db_train, "force_binary_masks", False),
        getattr(db_val, "force_binary_masks", False),
    )

    model_kwargs = {"mode": "train"}
    if args.model == "unetr":
        model_kwargs["image_size"] = tuple(args.patch_size)
    if args.model == "unet_resnet152":
        model_kwargs["encoder_pretrained"] = bool(args.encoder_pretrained)
    model = net_factory(
        net_type=args.model,
        in_chns=db_train.in_channels,
        class_num=args.num_classes,
        **model_kwargs,
    ).to(device)
    model_metadata = get_model_metadata(args.model)
    model_info = dict(model_metadata)
    model_info.update(extract_model_info(model))
    model_info["build_kwargs"] = {
        "net_type": args.model,
        "in_channels": db_train.in_channels,
        "num_classes": args.num_classes,
        **{key: value for key, value in model_kwargs.items() if key != "mode"},
    }
    write_model_config(snapshot_path, model_info)

    if not bool(args.force_retrain):
        reusable_match = resolve_basic_checkpoint(
            project_root=PROJECT_ROOT,
            output_root=args.output_root or None,
            model_name=args.model,
            dataset=args.dataset,
            expected_signature=_expected_basic_checkpoint_signature(db_train.in_channels),
        )
        if reusable_match is not None:
            reusable_checkpoint_path = Path(reusable_match["checkpoint_path"])
            registered_checkpoint_path = reusable_checkpoint_path
            try:
                reusable_checkpoint_path.resolve().relative_to((snapshot_path / "checkpoints").resolve())
            except ValueError:
                copied = register_reused_checkpoint(
                    source_checkpoint_path=reusable_checkpoint_path,
                    target_run_dir=snapshot_path,
                    project_root=PROJECT_ROOT,
                    source_branch="basic",
                    source_run_dir=reusable_match.get("run_dir"),
                    payload_updates={"phase": "basic"},
                )
                registered_checkpoint_path = Path(copied["best"]["checkpoint_path"])
            payload = load_checkpoint_into_model(registered_checkpoint_path, model, device=device)
            history = payload.get("extra_state", {}).get("history", {})
            logging.info(
                "Found compatible checkpoint for basic branch. Skip training and reuse: %s",
                project_relative_path(registered_checkpoint_path, PROJECT_ROOT),
            )
            _restore_loss_report_from_history(snapshot_path, history)
            profile = _compute_model_profile(model, device)
            metrics_rows = _run_final_evaluations(snapshot_path, registered_checkpoint_path, model, device, image_mode, profile, model_info, training_time_seconds=0.0)
            _write_timing_sidecars(
                snapshot_path,
                [
                    {
                        "phase": "basic",
                        "dataset": args.dataset,
                        "method": args.model,
                        "pruning_time_seconds": 0.0,
                        "search_time_seconds": 0.0,
                        "training_time_seconds": 0.0,
                        "inference_time_seconds": profile.get("inference_time_seconds"),
                        "evaluation_time_seconds": float(sum(float(row.get("evaluation_time_seconds") or 0.0) for row in metrics_rows)),
                        "total_time_seconds": float(time.perf_counter() - phase_start_time),
                        "reused_from": "basic",
                    }
                ],
            )
            _export_basic_channel_analysis(snapshot_path, registered_checkpoint_path, model, device)
            return

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
    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    iterations_per_epoch = len(trainloader)
    if iterations_per_epoch == 0:
        raise ValueError("Training dataset is empty; cannot start training.")

    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    ce_loss = CrossEntropyLoss()

    if args.max_epochs is not None and args.max_epochs <= 0:
        raise ValueError("--max_epochs must be a positive integer.")
    if args.max_epochs is None and args.max_iterations <= 0:
        raise ValueError("--max_iterations must be a positive integer when --max_epochs is not set.")
    if args.eval_interval <= 0:
        raise ValueError("--eval_interval must be a positive integer.")
    if args.eval_interval_epochs <= 0:
        raise ValueError("--eval_interval_epochs must be a positive integer.")

    logging.info("Start training")
    logging.info("%d iterations per epoch", iterations_per_epoch)
    training_start_time = time.perf_counter()
    logging.info("Train split: %s | Val split: %s", args.train_split, args.val_split)

    model.train()
    iter_num = 0
    best_performance = float("-inf")
    best_checkpoint_path = None
    history = {"train_total_loss": [], "val_total_loss": [], "val_macro_dice": []}

    def evaluate_and_checkpoint(metric_prefix, log_prefix):
        nonlocal best_performance, best_checkpoint_path

        performance, val_loss, vis_samples = run_validation(args, model, valloader, device, ce_loss)
        history["val_total_loss"].append(val_loss)
        history["val_macro_dice"].append(performance)
        is_best = performance > best_performance
        if is_best:
            best_performance = performance

        checkpoint_path = save_checkpoint(
            snapshot_path,
            sanitize_tag(metric_prefix),
            model=model,
            optimizer=optimizer,
            epoch=len(history["train_total_loss"]),
            global_step=iter_num,
            best_metric=best_performance,
            metrics={"val_macro_dice": performance, "val_total_loss": val_loss},
            config=vars(args),
            model_info=model_info,
            phase="basic",
            extra_state={"history": history},
            is_best=is_best,
            save_tagged_checkpoint=bool(args.save_history_checkpoints),
            save_last_checkpoint=bool(args.save_last_checkpoint),
            include_optimizer_state=bool(args.save_optimizer_state),
            project_root=PROJECT_ROOT,
        )

        if is_best:
            best_checkpoint_path = checkpoint_path
            logging.info("Updated best checkpoint: %s", project_relative_path(checkpoint_path, PROJECT_ROOT))
            if args.save_visualizations and vis_samples:
                vis_dir = build_evaluation_output_dir(snapshot_path, args.dataset, args.model, checkpoint_path, args.val_split)
                save_validation_visualizations(vis_samples, vis_dir)

        logging.info("%s : val_loss : %f | val_macro_dice : %f", log_prefix, val_loss, performance)
        model.train()

    if args.max_epochs is not None:
        logging.info("Training mode: epoch-based")
        logging.info("Max epochs: %d | Validation every %d epoch(s)", args.max_epochs, args.eval_interval_epochs)

        for epoch_num in tqdm(range(1, args.max_epochs + 1), ncols=70):
            epoch_losses = []

            for sampled_batch in trainloader:
                label_batch = sampled_batch["label"]
                validate_label_batch(label_batch, args.num_classes, sampled_batch)
                image_batch = sampled_batch["image"].to(device)
                label_batch = label_batch.to(device)

                logits = extract_logits(model(image_batch))
                loss_ce = ce_loss(logits, label_batch.long())
                outputs_soft = torch.softmax(logits, dim=1)
                loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
                loss = 0.5 * (loss_ce + loss_dice)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_num += 1
                epoch_losses.append(loss.item())
                logging.info("epoch %d/%d iteration %d : loss : %f", epoch_num, args.max_epochs, iter_num, loss.item())

            epoch_loss = float(np.mean(epoch_losses))
            history["train_total_loss"].append(epoch_loss)
            logging.info("epoch %d/%d : mean_loss : %f", epoch_num, args.max_epochs, epoch_loss)

            should_eval = (epoch_num % args.eval_interval_epochs == 0) or (epoch_num == args.max_epochs)
            if should_eval:
                evaluate_and_checkpoint(
                    metric_prefix=f"epoch_{epoch_num}_iter_{iter_num}",
                    log_prefix=f"epoch {epoch_num}/{args.max_epochs} iteration {iter_num}",
                )
    else:
        max_epoch = (args.max_iterations + iterations_per_epoch - 1) // iterations_per_epoch
        logging.info("Training mode: iteration-based")
        logging.info("Max iterations: %d | Validation every %d iteration(s)", args.max_iterations, args.eval_interval)

        for epoch_num in tqdm(range(1, max_epoch + 1), ncols=70):
            epoch_losses = []

            for sampled_batch in trainloader:
                label_batch = sampled_batch["label"]
                validate_label_batch(label_batch, args.num_classes, sampled_batch)
                image_batch = sampled_batch["image"].to(device)
                label_batch = label_batch.to(device)

                logits = extract_logits(model(image_batch))
                loss_ce = ce_loss(logits, label_batch.long())
                outputs_soft = torch.softmax(logits, dim=1)
                loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
                loss = 0.5 * (loss_ce + loss_dice)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_num += 1
                epoch_losses.append(loss.item())
                logging.info(
                    "epoch %d/%d iteration %d/%d : loss : %f",
                    epoch_num,
                    max_epoch,
                    iter_num,
                    args.max_iterations,
                    loss.item(),
                )

                should_eval = (iter_num % args.eval_interval == 0) or (iter_num == args.max_iterations)
                if should_eval:
                    evaluate_and_checkpoint(
                        metric_prefix=f"iter_{iter_num}",
                        log_prefix=f"epoch {epoch_num}/{max_epoch} iteration {iter_num}/{args.max_iterations}",
                    )

                if iter_num >= args.max_iterations:
                    break

            epoch_loss = float(np.mean(epoch_losses))
            history["train_total_loss"].append(epoch_loss)
            logging.info("epoch %d/%d : mean_loss : %f", epoch_num, max_epoch, epoch_loss)

            if iter_num >= args.max_iterations:
                break

    if best_checkpoint_path is None:
        fallback_checkpoint_path = save_checkpoint(
            snapshot_path,
            "final",
            model=model,
            optimizer=optimizer,
            epoch=len(history["train_total_loss"]),
            global_step=iter_num,
            best_metric=-1.0,
            metrics={"val_macro_dice": -1.0},
            config=vars(args),
            model_info=model_info,
            phase="basic",
            extra_state={"history": history},
            is_best=True,
            save_tagged_checkpoint=bool(args.save_history_checkpoints),
            save_last_checkpoint=bool(args.save_last_checkpoint),
            include_optimizer_state=bool(args.save_optimizer_state),
            project_root=PROJECT_ROOT,
        )
        best_checkpoint_path = fallback_checkpoint_path
        logging.warning("No validation checkpoint was selected; using the last model state at %s", project_relative_path(fallback_checkpoint_path, PROJECT_ROOT))

    training_time_seconds = time.perf_counter() - training_start_time
    save_loss_pdf(history, Path(snapshot_path) / "reports" / "basic_loss.pdf", title="Basic model loss")
    profile = _compute_model_profile(model, device)
    metrics_rows = _run_final_evaluations(Path(snapshot_path), best_checkpoint_path, model, device, image_mode, profile, model_info, training_time_seconds=training_time_seconds)
    _write_timing_sidecars(
        Path(snapshot_path),
        [
            {
                "phase": "basic",
                "dataset": args.dataset,
                "method": args.model,
                "pruning_time_seconds": 0.0,
                "search_time_seconds": 0.0,
                "training_time_seconds": float(training_time_seconds),
                "inference_time_seconds": profile.get("inference_time_seconds"),
                "evaluation_time_seconds": float(sum(float(row.get("evaluation_time_seconds") or 0.0) for row in metrics_rows)),
                "total_time_seconds": float(time.perf_counter() - phase_start_time),
            }
        ],
    )
    _export_basic_channel_analysis(Path(snapshot_path), best_checkpoint_path, model, device)


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    snapshot_path = build_basic_run_dir(
        project_root=PROJECT_ROOT,
        dataset=args.dataset,
        model_name=args.model,
        output_root=args.output_root or None,
    )
    snapshot_path.mkdir(parents=True, exist_ok=True)
    ensure_run_layout(snapshot_path)
    write_run_config(snapshot_path, vars(args))

    logging.basicConfig(
        filename=str(snapshot_path / "run.log"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, str(snapshot_path))
