from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path
from typing import Dict, List


MAIN_FIGURES = [
    ("figure3", "{dataset}/figure3_thresholding_methods.pdf"),
    ("figure4", "paper_figures/figure4_block_pruning_strategy.pdf"),
    ("figure5", "{dataset}/figure5_layerwise_pruning_ratio.pdf"),
    ("figure6", "{dataset}/figure6_accuracy_efficiency_tradeoff.pdf"),
    ("figure8", "{dataset}/figure8_visual_comparison.pdf"),
    ("figure15", "figure15_params_dice_tradeoff.pdf"),
    ("figure16", "figure16_performance_clinicdb.pdf"),
]

APPENDIX_FIGURES = [
    ("figure2", "{dataset}/figure2_importance_distribution.pdf"),
    ("figure7", "{dataset}/figure7_training_curve.pdf"),
    ("figure7a", "{dataset}/figure7a_training_loss_curve.pdf"),
    ("figure7b", "{dataset}/figure7b_validation_dice_curve.pdf"),
    ("figure9", "{dataset}/figure9_failure_cases.pdf"),
    ("figure10", "{dataset}/figure10_boundary_comparison.pdf"),
    ("figure11", "{dataset}/figure11_output_channels_per_layer_pruned_student.pdf"),
    ("figure12", "{dataset}/figure12_mean_channel_importance_per_layer_pruned_student.pdf"),
    ("figure13", "{dataset}/figure13_search_time_comparison.pdf"),
    ("figure14", "{dataset}/figure14_computational_cost_breakdown.pdf"),
]


def _copy_figure(statistics_root: Path, save_root: Path, dataset: str, section: str, figure_id: str, template: str) -> Dict[str, str]:
    relative = Path(template.format(dataset=dataset))
    source = statistics_root / relative
    target_dir = save_root / section
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{figure_id}_{source.name}"
    status = "missing"
    logging.info("Collecting figure: %s dataset=%s section=%s source=%s target=%s", figure_id, dataset, section, source, target)
    if source.is_file():
        shutil.copy2(source, target)
        status = "copied"
        logging.info("Copied figure: %s -> %s", source, target)
    else:
        logging.warning("Missing figure for paper collection: %s", source)
    return {
        "figure_id": figure_id,
        "source_path": str(source),
        "target_path": str(target),
        "dataset": dataset,
        "section": section,
        "status": status,
    }


def _write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Writing paper-ready manifest rows=%d -> %s", len(rows), path)
    fieldnames = ["figure_id", "source_path", "target_path", "dataset", "section", "status"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect generated paper figures into main/appendix folders.")
    parser.add_argument("--dataset-main", type=str, default="cvc_300")
    parser.add_argument("--outputs-root", type=str, default="outputs", help="Accepted for CLI symmetry; figures are copied from statistics root.")
    parser.add_argument("--statistics-root", type=str, default="statistics/outputs")
    parser.add_argument("--save-root", type=str, default="statistics/paper_ready")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    statistics_root = Path(args.statistics_root)
    save_root = Path(args.save_root)
    rows: List[Dict[str, str]] = []
    for figure_id, template in MAIN_FIGURES:
        rows.append(_copy_figure(statistics_root, save_root, args.dataset_main, "main", figure_id, template))
    for figure_id, template in APPENDIX_FIGURES:
        rows.append(_copy_figure(statistics_root, save_root, args.dataset_main, "appendix", figure_id, template))
    _write_manifest(save_root / "manifest.csv", rows)
    logging.info("Saved paper-ready manifest: %s", save_root / "manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
