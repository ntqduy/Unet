from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

from dataloaders.dataset import scan_dataset_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


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


def _write_manifest(case_ids: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for case_id in case_ids:
            file.write(f"{case_id}\n")


def generate_cvc_split(dataset_root: Path, seed: int, val_ratio: float, test_ratio: float) -> Dict:
    records = scan_dataset_records(dataset_root)
    case_ids = sorted(record.case_id for record in records)
    rng = random.Random(seed)
    shuffled_ids = list(case_ids)
    rng.shuffle(shuffled_ids)

    val_count = max(1, int(round(len(shuffled_ids) * val_ratio)))
    test_count = max(1, int(round(len(shuffled_ids) * test_ratio)))
    if val_count + test_count >= len(shuffled_ids):
        raise ValueError("For CVC-ClinicDB, val_ratio + test_ratio must leave at least one training sample.")

    test_ids = sorted(shuffled_ids[:test_count], key=lambda value: int(value) if value.isdigit() else value)
    val_ids = sorted(
        shuffled_ids[test_count:test_count + val_count],
        key=lambda value: int(value) if value.isdigit() else value,
    )
    train_ids = sorted(
        shuffled_ids[test_count + val_count:],
        key=lambda value: int(value) if value.isdigit() else value,
    )

    split_dir = dataset_root / "splits"
    _write_manifest(train_ids, split_dir / "train.txt")
    _write_manifest(val_ids, split_dir / "val.txt")
    _write_manifest(test_ids, split_dir / "test.txt")

    summary = {
        "dataset": "cvc_clinicdb",
        "seed": seed,
        "total_cases": len(case_ids),
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "note": "CVC-ClinicDB is written as a stable train/val/test split.",
    }
    with (split_dir / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def generate_kvasir_split(dataset_root: Path, seed: int, val_ratio_from_train: float) -> Dict:
    official_train = _read_case_ids(dataset_root / "train.txt")
    official_val = _read_case_ids(dataset_root / "val.txt")

    rng = random.Random(seed)
    shuffled_train = list(official_train)
    rng.shuffle(shuffled_train)

    val_count = max(1, int(round(len(shuffled_train) * val_ratio_from_train)))
    generated_val = sorted(shuffled_train[:val_count])
    generated_train = sorted(shuffled_train[val_count:])
    generated_test = sorted(official_val)

    split_dir = dataset_root / "splits"
    _write_manifest(generated_train, split_dir / "train.txt")
    _write_manifest(generated_val, split_dir / "val.txt")
    _write_manifest(generated_test, split_dir / "test.txt")

    summary = {
        "dataset": "kvasir_seg",
        "seed": seed,
        "official_train_count": len(official_train),
        "official_val_count": len(official_val),
        "generated_train_count": len(generated_train),
        "generated_val_count": len(generated_val),
        "generated_test_count": len(generated_test),
        "val_ratio_from_official_train": val_ratio_from_train,
        "note": "Official val.txt is promoted to test.txt. A new validation split is sampled from official train.txt.",
    }
    with (split_dir / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "cvc", "kvasir"], help="dataset to split")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cvc_val_ratio", type=float, default=0.1)
    parser.add_argument("--cvc_test_ratio", type=float, default=0.2)
    parser.add_argument("--kvasir_val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    summaries = []
    if args.dataset in {"all", "cvc"}:
        summaries.append(
            generate_cvc_split(
                DATA_ROOT / "CVC-ClinicDB",
                seed=args.seed,
                val_ratio=args.cvc_val_ratio,
                test_ratio=args.cvc_test_ratio,
            )
        )
    if args.dataset in {"all", "kvasir"}:
        summaries.append(generate_kvasir_split(DATA_ROOT / "Kvasir-SEG", seed=args.seed, val_ratio_from_train=args.kvasir_val_ratio))

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
