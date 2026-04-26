import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloaders.dataset import Normalize, ToTensor, build_dataset, list_available_datasets
from networks.net_factory import get_model_metadata, list_models, net_factory
from utils.checkpoints import load_checkpoint_into_model
from utils.evaluation import (
    build_evaluation_output_dir,
    checkpoint_label,
    evaluate_segmentation_dataset,
    sanitize_tag,
    save_evaluation_artifacts,
)
from utils.experiment import build_basic_run_dir, normalize_path_string, project_relative_path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "Kvasir-SEG"


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=str(DEFAULT_DATA_ROOT), help="dataset root")
parser.add_argument("--dataset", type=str, default="kvasir_seg", choices=list_available_datasets(), help="dataset name")
parser.add_argument("--exp", type=str, default="supervised", help="experiment name")
parser.add_argument("--model", type=str, default="unet", choices=list_models(), help="model name")
parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="evaluation split")
parser.add_argument("--checkpoint_path", type=str, default="", help="optional checkpoint path; defaults to the best checkpoint under the experiment")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--encoder_pretrained", type=int, default=1, help="only used by unet_resnet152")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256])
parser.add_argument("--save_visualizations", type=int, default=1)
parser.add_argument("--vis_limit", type=int, default=200, help="-1 means save all")
parser.add_argument("--output_root", type=str, default="", help="root directory for exported outputs; defaults to PROJECT_ROOT/outputs")

FLAGS = parser.parse_args()


def _resolve_requested_splits():
    if FLAGS.split == "all":
        return ["train", "val", "test"]
    return [FLAGS.split]


def _resolve_checkpoint_path(snapshot_path: Path) -> Path:
    if FLAGS.checkpoint_path:
        checkpoint_path = Path(FLAGS.checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")
        return checkpoint_path

    manifest_path = snapshot_path / "checkpoints" / "metadata" / "best.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        candidate = (PROJECT_ROOT / manifest.get("checkpoint_path", "")).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    legacy_manifest_path = snapshot_path / "best_checkpoint.json"
    if legacy_manifest_path.is_file():
        with legacy_manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        candidate = (PROJECT_ROOT / manifest.get("checkpoint_path", "")).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    weights_dir = snapshot_path / "weights"
    candidates = [
        snapshot_path / "checkpoints" / "best.pth",
        weights_dir / f"{sanitize_tag(FLAGS.dataset)}_{sanitize_tag(FLAGS.model)}_best.pth",
        snapshot_path / f"{FLAGS.model}_best_model.pth",
    ]
    if weights_dir.is_dir():
        candidates.extend(sorted(weights_dir.glob(f"*{sanitize_tag(FLAGS.model)}*best*.pth")))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve a checkpoint for exp='{FLAGS.exp}', model='{FLAGS.model}'. "
        "Provide --checkpoint_path explicitly or ensure a best checkpoint exists."
    )


def _build_summary_metadata(split: str, checkpoint_path: Path, model_info: dict | None = None) -> dict:
    resolved_model_info = dict(model_info or get_model_metadata(FLAGS.model))
    return {
        "experiment": FLAGS.exp,
        "dataset": FLAGS.dataset,
        "dataset_root": normalize_path_string(FLAGS.root_path),
        "split": split,
        "model": FLAGS.model,
        "architecture": FLAGS.model,
        "backbone_name": resolved_model_info.get("backbone_name"),
        "student_name": resolved_model_info.get("student_name"),
        "model_info": resolved_model_info,
        "num_classes": FLAGS.num_classes,
        "in_channels": FLAGS.in_channels,
        "patch_size": list(FLAGS.patch_size),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_label": checkpoint_label(checkpoint_path),
        "checkpoint_path": project_relative_path(checkpoint_path, PROJECT_ROOT),
    }


def _write_overview(checkpoint_root: Path, split_summaries: dict) -> None:
    overview = {
        "experiment": FLAGS.exp,
        "dataset": FLAGS.dataset,
        "dataset_root": normalize_path_string(FLAGS.root_path),
        "model": FLAGS.model,
        "checkpoint_name": next(iter(split_summaries.values()))["checkpoint_name"],
        "checkpoint_path": next(iter(split_summaries.values()))["checkpoint_path"],
        "splits": split_summaries,
    }
    with (checkpoint_root / "evaluation_overview.json").open("w", encoding="utf-8") as file:
        json.dump(overview, file, indent=2)

    lines = [
        f"# Evaluation Overview | {FLAGS.dataset} | {FLAGS.model}",
        "",
        f"- Experiment: `{FLAGS.exp}`",
        f"- Dataset root: `{normalize_path_string(FLAGS.root_path)}`",
        f"- Checkpoint: `{overview['checkpoint_name']}`",
        f"- Checkpoint path: `{overview['checkpoint_path']}`",
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


def test_calculate_metric():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot_path = build_basic_run_dir(
        project_root=PROJECT_ROOT,
        dataset=FLAGS.dataset,
        model_name=FLAGS.model,
        output_root=FLAGS.output_root or None,
    )
    legacy_basic_snapshot = PROJECT_ROOT / "outputs" / sanitize_tag(FLAGS.model) / sanitize_tag(FLAGS.dataset) / "basic"
    legacy_snapshot_path = PROJECT_ROOT / "logs" / "model" / "supervised" / FLAGS.exp
    if not snapshot_path.exists() and legacy_basic_snapshot.exists():
        snapshot_path = legacy_basic_snapshot
    elif not snapshot_path.exists() and legacy_snapshot_path.exists():
        snapshot_path = legacy_snapshot_path
    image_mode = "grayscale" if FLAGS.in_channels == 1 else "rgb"
    checkpoint_path = _resolve_checkpoint_path(snapshot_path)

    model_kwargs = {"mode": "test"}
    if FLAGS.model == "unetr":
        model_kwargs["image_size"] = tuple(FLAGS.patch_size)
    if FLAGS.model == "unet_resnet152":
        model_kwargs["encoder_pretrained"] = bool(FLAGS.encoder_pretrained)

    reference_dataset = build_dataset(
        dataset_name=FLAGS.dataset,
        base_dir=FLAGS.root_path,
        split="train",
        transform=transforms.Compose([Normalize(), ToTensor()]),
        image_mode=image_mode,
    )
    model = net_factory(
        net_type=FLAGS.model,
        in_chns=reference_dataset.in_channels,
        class_num=FLAGS.num_classes,
        **model_kwargs,
    ).to(device)
    checkpoint_payload = load_checkpoint_into_model(checkpoint_path, model, device=device)
    checkpoint_model_info = checkpoint_payload.get("model_info", {})
    model.eval()

    split_summaries = {}
    requested_splits = _resolve_requested_splits()
    for split in requested_splits:
        dataset = build_dataset(
            dataset_name=FLAGS.dataset,
            base_dir=FLAGS.root_path,
            split=split,
            transform=transforms.Compose([Normalize(), ToTensor()]),
            image_mode=image_mode,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=device.type == "cuda")
        output_dir = build_evaluation_output_dir(snapshot_path, FLAGS.dataset, FLAGS.model, checkpoint_path, split)
        result = evaluate_segmentation_dataset(
            model,
            dataloader,
            device=device,
            num_classes=FLAGS.num_classes,
            patch_size=FLAGS.patch_size,
            output_dir=output_dir,
            save_visualizations=bool(FLAGS.save_visualizations),
            vis_limit=FLAGS.vis_limit,
            progress_desc=f"eval-{split}",
        )
        summary = save_evaluation_artifacts(
            output_dir,
            _build_summary_metadata(split, checkpoint_path, model_info=checkpoint_model_info),
            result["average_metric"],
            result["case_metrics"],
            project_root=PROJECT_ROOT,
        )
        split_summaries[split] = summary
        macro_mean = summary["metrics"]["macro_mean"]
        print(
            f"[{split}] checkpoint={summary['checkpoint_name']} | dataset={summary['dataset']} | "
            f"model={summary['model']} | dice={macro_mean['dice']:.6f} | hd95={macro_mean['hd95']:.6f}"
        )

    if split_summaries:
        checkpoint_root = build_evaluation_output_dir(snapshot_path, FLAGS.dataset, FLAGS.model, checkpoint_path, requested_splits[0]).parent
        _write_overview(checkpoint_root, split_summaries)

    return split_summaries


if __name__ == "__main__":
    summaries = test_calculate_metric()
    print(json.dumps({split: summary["metrics"]["macro_mean"] for split, summary in summaries.items()}, indent=2))
