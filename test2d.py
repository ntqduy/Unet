import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import Normalize, ToTensor, build_dataset, list_available_datasets
from networks.net_factory import list_models, net_factory
from utils.val_2d import test_single_volume
from utils.visualization import save_triplet_visualization


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "Kvasir-SEG"


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default=str(DEFAULT_DATA_ROOT), help="dataset root")
parser.add_argument("--dataset", type=str, default="kvasir", choices=list_available_datasets(), help="dataset name")
parser.add_argument("--exp", type=str, default="supervised", help="experiment name")
parser.add_argument("--model", type=str, default="unet", choices=list_models(), help="model name")
parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="evaluation split")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--in_channels", type=int, default=3)
parser.add_argument("--encoder_pretrained", type=int, default=0, help="only used by unet_resnet152")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256])
parser.add_argument("--save_visualizations", type=int, default=1)
parser.add_argument("--vis_limit", type=int, default=200, help="-1 means save all")

FLAGS = parser.parse_args()


def _write_case_metrics(case_metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "case_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "class_index", "dice", "hd95"])
        for row in case_metrics:
            writer.writerow(row)


def test_calculate_metric():
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot_path = PROJECT_ROOT / "logs" / "model" / "supervised" / FLAGS.exp
    prediction_path = snapshot_path / "predictions" / FLAGS.split
    image_mode = "grayscale" if FLAGS.in_channels == 1 else "rgb"

    dataset = build_dataset(
        dataset_name=FLAGS.dataset,
        base_dir=FLAGS.root_path,
        split=FLAGS.split,
        transform=transforms.Compose([Normalize(), ToTensor()]),
        image_mode=image_mode,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=device.type == "cuda")

    model_kwargs = {"mode": "test"}
    if FLAGS.model == "unetr":
        model_kwargs["image_size"] = tuple(FLAGS.patch_size)
    if FLAGS.model == "unet_resnet152":
        model_kwargs["encoder_pretrained"] = bool(FLAGS.encoder_pretrained)
    model = net_factory(
        net_type=FLAGS.model,
        in_chns=dataset.in_channels,
        class_num=FLAGS.num_classes,
        **model_kwargs,
    ).to(device)

    save_model_path = snapshot_path / f"{FLAGS.model}_best_model.pth"
    model.load_state_dict(torch.load(save_model_path, map_location=device))
    model.eval()

    total_metric = []
    case_metrics = []
    saved_visualizations = 0

    for sample in tqdm(dataloader):
        metric_i, prediction = test_single_volume(
            sample["image"],
            sample["label"],
            model,
            classes=FLAGS.num_classes,
            patch_size=FLAGS.patch_size,
            device=device,
            return_prediction=True,
        )
        metric_i = np.array(metric_i)
        total_metric.append(metric_i)

        case_name = sample["case"][0] if isinstance(sample["case"], (list, tuple)) else str(sample["case"])
        for class_index, (dice_value, hd95_value) in enumerate(metric_i, start=1):
            case_metrics.append([case_name, class_index, float(dice_value), float(hd95_value)])

        should_save = bool(FLAGS.save_visualizations) and (FLAGS.vis_limit < 0 or saved_visualizations < FLAGS.vis_limit)
        if should_save:
            save_triplet_visualization(
                image=sample["image"][0],
                label=sample["label"][0],
                prediction=prediction[0],
                output_dir=prediction_path,
                case_name=case_name,
            )
            saved_visualizations += 1

    avg_metric = np.stack(total_metric, axis=0).mean(axis=0)
    prediction_path.mkdir(parents=True, exist_ok=True)
    _write_case_metrics(case_metrics, prediction_path)

    summary = {
        "dataset": FLAGS.dataset,
        "split": FLAGS.split,
        "model": FLAGS.model,
        "num_cases": len(case_metrics) // max(FLAGS.num_classes - 1, 1),
        "average_metric": avg_metric.tolist(),
    }
    with (prediction_path / "metrics_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Average metric:", avg_metric)
    return avg_metric


if __name__ == "__main__":
    metric = test_calculate_metric()
    print(metric)
