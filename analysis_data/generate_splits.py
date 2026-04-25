from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow direct execution via `python analysis_data/generate_splits.py`.
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dataloaders.dataset import scan_dataset_records
except ModuleNotFoundError:
    scan_dataset_records = None

DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_HINTS = ("mask", "masks", "ground truth", "ground_truth", "groundtruth", "label", "labels", "annotation", "annotations")
IMAGE_HINTS = ("image", "images", "original")
IGNORE_HINTS = ("bbox", "bounding")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    aliases: Tuple[str, ...]
    root_name: str
    archive_name: str


@dataclass(frozen=True)
class FallbackSampleRecord:
    case_id: str
    image_path: Path
    mask_path: Path


DATASET_SPECS: Tuple[DatasetSpec, ...] = (
    DatasetSpec("cvc_clinicdb", ("cvc", "clinicdb", "cvc-clinicdb"), "CVC-ClinicDB", "CVC-ClinicDB.zip"),
    DatasetSpec("kvasir_seg", ("kvasir", "kvasir-seg"), "Kvasir-SEG", "Kvasir-SEG.zip"),
    DatasetSpec("etis", ("etis_larib", "etis-larib"), "ETIS", "ETIS.zip"),
    DatasetSpec("cvc_colondb", ("cvc_colon_db", "cvc-colondb", "colondb"), "CVC-ColonDB", "CVC-ColonDB.zip"),
    DatasetSpec("cvc_300", ("cvc300", "cvc-300"), "CVC-300", "CVC-300.zip"),
)


def _path_parts_lower(path: Path) -> Tuple[str, ...]:
    return tuple(part.lower() for part in path.parts)


def _contains_any(parts: Iterable[str], hints: Sequence[str]) -> bool:
    return any(any(hint in part for hint in hints) for part in parts)


def _score_candidate(path: Path, kind: str) -> int:
    parts = _path_parts_lower(path.parent)
    score = 0
    if kind == "image":
        if _contains_any(parts, ("images",)):
            score += 6
        if _contains_any(parts, ("original",)):
            score += 5
    else:
        if _contains_any(parts, ("masks", "mask")):
            score += 6
        if _contains_any(parts, ("ground truth", "ground_truth", "groundtruth", "labels", "label")):
            score += 5
    if path.suffix.lower() == ".png":
        score += 3
    elif path.suffix.lower() in {".jpg", ".jpeg"}:
        score += 2
    elif path.suffix.lower() in {".tif", ".tiff"}:
        score += 1
    return score


def _classify_asset(path: Path) -> Optional[str]:
    parts = _path_parts_lower(path)
    if _contains_any(parts, IGNORE_HINTS):
        return None
    if _contains_any(parts, MASK_HINTS):
        return "mask"
    if _contains_any(parts, IMAGE_HINTS):
        return "image"
    return None


def _fallback_scan_dataset_records(dataset_root: Path) -> List[FallbackSampleRecord]:
    image_candidates: Dict[str, List[Path]] = {}
    mask_candidates: Dict[str, List[Path]] = {}
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        asset_type = _classify_asset(path)
        if asset_type == "image":
            image_candidates.setdefault(path.stem, []).append(path)
        elif asset_type == "mask":
            mask_candidates.setdefault(path.stem, []).append(path)

    records = []
    for case_id in sorted(set(image_candidates) & set(mask_candidates), key=_natural_key):
        records.append(
            FallbackSampleRecord(
                case_id=case_id,
                image_path=max(image_candidates[case_id], key=lambda item: _score_candidate(item, "image")),
                mask_path=max(mask_candidates[case_id], key=lambda item: _score_candidate(item, "mask")),
            )
        )
    if not records:
        raise RuntimeError(f"Could not find any 2D image/mask pairs under '{dataset_root}'.")
    return records


def _spec_by_name(name: str) -> DatasetSpec:
    normalized = name.strip().lower()
    for spec in DATASET_SPECS:
        names = {spec.key, spec.root_name.lower(), *spec.aliases}
        if normalized in names:
            return spec
    available = ", ".join(spec.key for spec in DATASET_SPECS)
    raise ValueError(f"Unknown dataset '{name}'. Available: all, {available}")


def _natural_key(value: str) -> Tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _write_manifest(case_ids: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for case_id in case_ids:
            file.write(f"{case_id}\n")


def _extract_archive_if_needed(spec: DatasetSpec, dataset_root: Path, extract: bool) -> None:
    if dataset_root.exists():
        return
    archive_path = DATA_ROOT / spec.archive_name
    if not extract:
        raise FileNotFoundError(
            f"Dataset root '{dataset_root}' does not exist. "
            f"Extract '{archive_path}' first, or rerun with --extract."
        )
    if not archive_path.is_file():
        raise FileNotFoundError(f"Cannot extract missing archive '{archive_path}'.")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = [entry.filename for entry in archive.infolist() if entry.filename and not entry.filename.endswith("/")]
        has_dataset_root = any(Path(name).parts and Path(name).parts[0] == spec.root_name for name in names)
        extract_root = DATA_ROOT if has_dataset_root else dataset_root
        extract_root.mkdir(parents=True, exist_ok=True)
        archive.extractall(extract_root)


def split_case_ids(
    case_ids: Sequence[str],
    seed: int,
    split_ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> Dict[str, List[str]]:
    train_ratio, val_ratio, test_ratio = split_ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    normalized_train_ratio = train_ratio / total_ratio
    normalized_val_ratio = val_ratio / total_ratio
    shuffled_ids = list(case_ids)
    random.Random(seed).shuffle(shuffled_ids)

    train_end = int(len(shuffled_ids) * normalized_train_ratio)
    val_end = train_end + int(len(shuffled_ids) * normalized_val_ratio)
    splits = {
        "train": shuffled_ids[:train_end],
        "val": shuffled_ids[train_end:val_end],
        "test": shuffled_ids[val_end:],
    }
    return {
        split: sorted(values, key=_natural_key)
        for split, values in splits.items()
    }


def generate_split_for_dataset(
    spec: DatasetSpec,
    seed: int,
    split_ratios: Tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
    extract: bool = False,
    overwrite: bool = False,
) -> Dict:
    dataset_root = DATA_ROOT / spec.root_name
    _extract_archive_if_needed(spec, dataset_root, extract=extract)

    split_dir = dataset_root / "splits"
    split_paths = {split: split_dir / f"{split}.txt" for split in ("train", "val", "test")}
    if not overwrite and all(path.is_file() for path in split_paths.values()):
        return {
            "dataset": spec.key,
            "dataset_root": str(dataset_root),
            "skipped": True,
            "reason": "split manifests already exist; use --overwrite to regenerate them",
        }

    records = scan_dataset_records(dataset_root) if scan_dataset_records is not None else _fallback_scan_dataset_records(dataset_root)
    case_ids = sorted((record.case_id for record in records), key=_natural_key)
    splits = split_case_ids(case_ids, seed=seed, split_ratios=split_ratios)

    for split, ids in splits.items():
        _write_manifest(ids, split_paths[split])

    summary = {
        "dataset": spec.key,
        "dataset_root": str(dataset_root),
        "seed": seed,
        "split_ratios": {
            "train": split_ratios[0],
            "val": split_ratios[1],
            "test": split_ratios[2],
        },
        "total_cases": len(case_ids),
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "manifests": {split: str(path) for split, path in split_paths.items()},
        "note": "Stable train/val/test manifests generated with the same slicing rule as dataloaders.dataset.",
    }
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate stable 80/10/10 train/val/test manifests for polyp datasets.")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=["all", *(spec.key for spec in DATASET_SPECS), "cvc", "kvasir", "etis_larib", "cvc_colon_db", "cvc300"],
        help="dataset to split",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--extract", action="store_true", help="extract data/<dataset>.zip if the dataset directory is missing")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing split manifests")
    args = parser.parse_args()

    selected_specs = DATASET_SPECS if args.dataset == "all" else (_spec_by_name(args.dataset),)
    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    summaries = [
        generate_split_for_dataset(
            spec,
            seed=args.seed,
            split_ratios=split_ratios,
            extract=args.extract,
            overwrite=args.overwrite,
        )
        for spec in selected_specs
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
