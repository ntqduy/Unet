import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import Normalize, RandomGenerator, ToTensor, build_dataset, list_available_datasets
from networks.net_factory import list_models, net_factory
from utils import losses, val_2d
from utils.evaluation import (
    build_evaluation_output_dir,
    checkpoint_label,
    evaluate_segmentation_dataset,
    sanitize_tag,
    save_evaluation_artifacts,
)
from utils.visualization import save_triplet_visualization


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "Kvasir-SEG"


def extract_logits(model_output):
    if isinstance(model_output, (list, tuple)):
        tensor_candidates = [item for item in model_output if torch.is_tensor(item) and item.ndim >= 4]
        if not tensor_candidates:
            raise ValueError("Model output does not contain segmentation logits.")
        return tensor_candidates[0]
    return model_output


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=str(DEFAULT_DATA_ROOT), help="dataset root path")
parser.add_argument("--dataset", type=str, default="kvasir", choices=list_available_datasets(), help="dataset name")
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
parser.add_argument("--encoder_pretrained", type=int, default=0, help="only used by unet_resnet152")
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--save_visualizations", type=int, default=1)
parser.add_argument("--vis_num_samples", type=int, default=5)
parser.add_argument(
    "--final_eval_splits",
    nargs="*",
    default=["train", "val", "test"],
    help="splits to evaluate after training with the best checkpoint",
)

args = parser.parse_args()
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
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _unique_preserve_order(values):
    ordered = []
    seen = set()
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _run_prefix() -> str:
    return f"{sanitize_tag(args.dataset)}_{sanitize_tag(args.model)}"


def _best_checkpoint_manifest_path(snapshot_path: Path) -> Path:
    return snapshot_path / "best_checkpoint.json"


def _build_checkpoint_manifest(checkpoint_path: Path, alias_path: Path, legacy_path: Path, metric_prefix: str, metric_value: float) -> dict:
    return {
        "experiment": args.exp,
        "dataset": args.dataset,
        "dataset_root": str(Path(args.root_path).expanduser().resolve()),
        "model": args.model,
        "architecture": args.model,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "metric_name": "val_macro_dice",
        "metric_value": float(metric_value),
        "metric_prefix": metric_prefix,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_alias_path": str(alias_path.resolve()),
        "legacy_checkpoint_path": str(legacy_path.resolve()),
        "checkpoint_label": checkpoint_label(checkpoint_path),
    }


def _build_evaluation_metadata(split: str, checkpoint_path: Path) -> dict:
    return {
        "experiment": args.exp,
        "dataset": args.dataset,
        "dataset_root": str(Path(args.root_path).expanduser().resolve()),
        "split": split,
        "model": args.model,
        "architecture": args.model,
        "num_classes": args.num_classes,
        "in_channels": args.in_channels,
        "patch_size": list(args.patch_size),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_label": checkpoint_label(checkpoint_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }


def _write_evaluation_overview(snapshot_path: Path, checkpoint_path: Path, split_summaries: dict) -> None:
    if not split_summaries:
        return

    checkpoint_root = build_evaluation_output_dir(snapshot_path, args.dataset, args.model, checkpoint_path, next(iter(split_summaries))).parent
    overview = {
        "experiment": args.exp,
        "dataset": args.dataset,
        "dataset_root": str(Path(args.root_path).expanduser().resolve()),
        "model": args.model,
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "splits": split_summaries,
    }
    _write_json(checkpoint_root / "evaluation_overview.json", overview)

    lines = [
        f"# Final Evaluation Overview | {args.dataset} | {args.model}",
        "",
        f"- Experiment: `{args.exp}`",
        f"- Dataset root: `{Path(args.root_path).expanduser().resolve()}`",
        f"- Checkpoint: `{checkpoint_path.name}`",
        f"- Checkpoint path: `{checkpoint_path.resolve()}`",
        "",
        "## Splits",
        "",
    ]
    for split_name, summary in split_summaries.items():
        macro = summary["metrics"]["macro_mean"]
        lines.append(
            f"- `{split_name}` | dice `{macro['dice']:.6f}` | hd95 `{macro['hd95']:.6f}` | cases `{summary['num_cases']}`"
        )
    with (checkpoint_root / "evaluation_overview.md").open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def run_validation(args, model, valloader, device):
    model.eval()
    metric_list = []
    vis_samples = []

    with torch.no_grad():
        for sampled_val in valloader:
            validate_label_batch(sampled_val["label"], args.num_classes, sampled_val)
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
    return performance, vis_samples


def _run_final_evaluations(snapshot_path: Path, checkpoint_path: Path, model, device, image_mode: str) -> None:
    logging.info("Running final evaluation with best checkpoint: %s", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    eval_transform = transforms.Compose([Normalize(), ToTensor()])
    split_summaries = {}
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

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=device.type == "cuda")
        output_dir = build_evaluation_output_dir(snapshot_path, args.dataset, args.model, checkpoint_path, split)
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
        summary = save_evaluation_artifacts(
            output_dir,
            _build_evaluation_metadata(split, checkpoint_path),
            result["average_metric"],
            result["case_metrics"],
        )
        split_summaries[split] = summary
        macro = summary["metrics"]["macro_mean"]
        logging.info(
            "Final evaluation | split=%s | checkpoint=%s | macro_dice=%.6f | macro_hd95=%.6f | cases=%d",
            split,
            checkpoint_path.name,
            macro["dice"],
            macro["hd95"],
            summary["num_cases"],
        )

    _write_evaluation_overview(snapshot_path, checkpoint_path, split_summaries)


def train(args, snapshot_path):
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
        num_workers=1,
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
    logging.info("Train split: %s | Val split: %s", args.train_split, args.val_split)

    model.train()
    iter_num = 0
    best_performance = float("-inf")
    weights_dir = Path(snapshot_path) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = None

    def evaluate_and_checkpoint(metric_prefix, log_prefix):
        nonlocal best_performance, best_checkpoint_path

        performance, vis_samples = run_validation(args, model, valloader, device)

        if performance > best_performance:
            best_performance = performance
            metric_prefix_safe = sanitize_tag(metric_prefix)
            unique_checkpoint_path = weights_dir / f"{_run_prefix()}_{metric_prefix_safe}_valdice_{best_performance:.4f}.pth"
            alias_checkpoint_path = weights_dir / f"{_run_prefix()}_best.pth"
            legacy_checkpoint_path = Path(snapshot_path) / f"{args.model}_best_model.pth"

            torch.save(model.state_dict(), unique_checkpoint_path)
            torch.save(model.state_dict(), alias_checkpoint_path)
            torch.save(model.state_dict(), legacy_checkpoint_path)
            best_checkpoint_path = unique_checkpoint_path

            manifest = _build_checkpoint_manifest(
                unique_checkpoint_path,
                alias_checkpoint_path,
                legacy_checkpoint_path,
                metric_prefix=metric_prefix,
                metric_value=best_performance,
            )
            _write_json(_best_checkpoint_manifest_path(Path(snapshot_path)), manifest)
            logging.info("Saved best model to %s", unique_checkpoint_path)
            logging.info("Updated stable best aliases: %s | %s", alias_checkpoint_path, legacy_checkpoint_path)

            if args.save_visualizations and vis_samples:
                vis_dir = build_evaluation_output_dir(snapshot_path, args.dataset, args.model, unique_checkpoint_path, args.val_split)
                save_validation_visualizations(vis_samples, vis_dir)

        logging.info("%s : val_macro_dice : %f", log_prefix, performance)
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
            logging.info("epoch %d/%d : mean_loss : %f", epoch_num, max_epoch, epoch_loss)

            if iter_num >= args.max_iterations:
                break

    if best_checkpoint_path is None:
        fallback_checkpoint_path = weights_dir / f"{_run_prefix()}_last.pth"
        alias_checkpoint_path = weights_dir / f"{_run_prefix()}_best.pth"
        legacy_checkpoint_path = Path(snapshot_path) / f"{args.model}_best_model.pth"
        torch.save(model.state_dict(), fallback_checkpoint_path)
        torch.save(model.state_dict(), alias_checkpoint_path)
        torch.save(model.state_dict(), legacy_checkpoint_path)
        best_checkpoint_path = fallback_checkpoint_path
        manifest = _build_checkpoint_manifest(
            fallback_checkpoint_path,
            alias_checkpoint_path,
            legacy_checkpoint_path,
            metric_prefix="last",
            metric_value=-1.0,
        )
        _write_json(_best_checkpoint_manifest_path(Path(snapshot_path)), manifest)
        logging.warning("No validation checkpoint was selected; using the last model state at %s", fallback_checkpoint_path)

    _run_final_evaluations(Path(snapshot_path), best_checkpoint_path, model, device, image_mode)


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    snapshot_path = PROJECT_ROOT / "logs" / "model" / "supervised" / args.exp
    snapshot_path.mkdir(parents=True, exist_ok=True)

    shutil.copy(Path(__file__).resolve(), snapshot_path / "train2d.py")
    _write_json(snapshot_path / "run_config.json", vars(args))

    logging.basicConfig(
        filename=str(snapshot_path / "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, str(snapshot_path))