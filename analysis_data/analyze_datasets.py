from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
from torchvision.io import ImageReadMode, read_image

from dataloaders.dataset import find_manifest_path, list_existing_splits, scan_dataset_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
REPORT_ROOT = PROJECT_ROOT / "analysis_data" / "reports"


def _read_case_ids(manifest_path: Path) -> List[str]:
    case_ids = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            token = token.split(",")[0].strip()
            case_ids.append(token)
    return case_ids


def _top_counter_items(counter: Counter, limit: int = 10) -> List[Dict]:
    return [{"value": str(key), "count": int(value)} for key, value in counter.most_common(limit)]


def analyze_dataset(dataset_name: str, dataset_root: Path) -> Dict:
    records = scan_dataset_records(dataset_root)
    size_counter = Counter()
    pixel_hist_r = torch.zeros(256, dtype=torch.long)
    pixel_hist_g = torch.zeros(256, dtype=torch.long)
    pixel_hist_b = torch.zeros(256, dtype=torch.long)
    mask_value_counter = Counter()
    foreground_ratios = []
    image_sum = torch.zeros(3, dtype=torch.float64)
    image_sum_sq = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    sample_files = []
    for index, record in enumerate(records):
        image = read_image(str(record.image_path), mode=ImageReadMode.RGB)
        mask = read_image(str(record.mask_path), mode=ImageReadMode.GRAY)

        height, width = int(image.shape[1]), int(image.shape[2])
        size_counter[f"{width}x{height}"] += 1

        image_float = image.float() / 255.0
        flat_image = image.reshape(3, -1)
        image_sum += image_float.reshape(3, -1).sum(dim=1, dtype=torch.float64)
        image_sum_sq += (image_float.reshape(3, -1) ** 2).sum(dim=1, dtype=torch.float64)
        total_pixels += image.shape[1] * image.shape[2]

        pixel_hist_r += torch.bincount(flat_image[0], minlength=256)
        pixel_hist_g += torch.bincount(flat_image[1], minlength=256)
        pixel_hist_b += torch.bincount(flat_image[2], minlength=256)

        unique_values = torch.unique(mask)
        for value in unique_values.tolist():
            mask_value_counter[int(value)] += 1
        foreground_ratios.append(float((mask > 0).float().mean().item()))

        if index < 10:
            sample_files.append(
                {
                    "case_id": record.case_id,
                    "image_path": str(record.image_path),
                    "mask_path": str(record.mask_path),
                    "image_size": [width, height],
                    "mask_unique_values": [int(value) for value in unique_values.tolist()],
                }
            )

    channel_mean_tensor = image_sum / total_pixels
    channel_std_tensor = torch.sqrt((image_sum_sq / total_pixels) - channel_mean_tensor ** 2)

    split_counts = {}
    for split_name in list_existing_splits(dataset_root):
        manifest_path = find_manifest_path(dataset_root, split_name)
        if manifest_path is not None:
            split_counts[split_name] = len(_read_case_ids(manifest_path))

    return {
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "num_pairs": len(records),
        "num_images": len(records),
        "num_masks": len(records),
        "available_splits": sorted(split_counts.keys()),
        "split_counts": split_counts,
        "image_size_top10": _top_counter_items(size_counter, limit=10),
        "channel_mean_rgb": channel_mean_tensor.tolist(),
        "channel_std_rgb": channel_std_tensor.tolist(),
        "foreground_ratio": {
            "min": min(foreground_ratios) if foreground_ratios else 0.0,
            "max": max(foreground_ratios) if foreground_ratios else 0.0,
            "mean": sum(foreground_ratios) / len(foreground_ratios) if foreground_ratios else 0.0,
        },
        "mask_value_presence": {str(key): int(value) for key, value in sorted(mask_value_counter.items())},
        "pixel_histogram_rgb": {
            "r": pixel_hist_r.tolist(),
            "g": pixel_hist_g.tolist(),
            "b": pixel_hist_b.tolist(),
        },
        "sample_files": sample_files,
    }


def summary_to_markdown(summary: Dict) -> str:
    lines = [
        f"# {summary['dataset_name']}",
        "",
        f"- Root: `{summary['dataset_root']}`",
        f"- Image/mask pairs: `{summary['num_pairs']}`",
        f"- Available splits: `{', '.join(summary['available_splits']) if summary['available_splits'] else 'none'}`",
        "",
        "## Split Counts",
        "",
    ]
    if summary["split_counts"]:
        for split_name, count in sorted(summary["split_counts"].items()):
            lines.append(f"- `{split_name}`: `{count}`")
    else:
        lines.append("- No manifest split found.")

    lines.extend(
        [
            "",
            "## Image Statistics",
            "",
            f"- Channel mean RGB: `{summary['channel_mean_rgb']}`",
            f"- Channel std RGB: `{summary['channel_std_rgb']}`",
            f"- Foreground ratio: `{summary['foreground_ratio']}`",
            "",
            "## Common Image Sizes",
            "",
        ]
    )
    for item in summary["image_size_top10"]:
        lines.append(f"- `{item['value']}`: `{item['count']}`")

    lines.extend(["", "## Mask Value Presence", ""])
    for mask_value, count in summary["mask_value_presence"].items():
        lines.append(f"- Value `{mask_value}` appears in `{count}` sample(s)")

    lines.extend(["", "## Sample Files", ""])
    for sample in summary["sample_files"]:
        lines.append(
            f"- `{sample['case_id']}` | size `{sample['image_size']}` | values `{sample['mask_unique_values']}` | "
            f"image `{sample['image_path']}` | mask `{sample['mask_path']}`"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "cvc", "kvasir"], help="dataset to analyze")
    args = parser.parse_args()

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = []
    if args.dataset in {"all", "cvc"}:
        datasets.append(("cvc_clinicdb", DATA_ROOT / "CVC-ClinicDB"))
    if args.dataset in {"all", "kvasir"}:
        datasets.append(("kvasir_seg", DATA_ROOT / "Kvasir-SEG"))

    combined = {}
    for dataset_name, dataset_root in datasets:
        summary = analyze_dataset(dataset_name, dataset_root)
        combined[dataset_name] = summary
        with (REPORT_ROOT / f"{dataset_name}_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        with (REPORT_ROOT / f"{dataset_name}_summary.md").open("w", encoding="utf-8") as file:
            file.write(summary_to_markdown(summary))

    with (REPORT_ROOT / "dataset_overview.json").open("w", encoding="utf-8") as file:
        json.dump(combined, file, indent=2)

    print(json.dumps({key: value["num_pairs"] for key, value in combined.items()}, indent=2))


if __name__ == "__main__":
    main()
