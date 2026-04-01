from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_HINTS = ("mask", "masks", "ground truth", "ground_truth", "groundtruth", "label", "labels", "annotation", "annotations")
IMAGE_HINTS = ("image", "images", "original")
IGNORE_HINTS = ("bbox", "bounding")
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
MANIFEST_SEARCH_SUBDIRS = ("splits", "")
BINARY_MASK_THRESHOLD = 127
BINARY_MASK_DATASETS = {"cvc", "cvc_clinicdb", "kvasir", "kvasir_seg"}
BINARY_MASK_PATH_HINTS = ("kvasir", "cvc-clinicdb", "cvc_clinicdb")


@dataclass(frozen=True)
class SampleRecord:
    case_id: str
    image_path: Path
    mask_path: Path


def _ensure_hw(size: Sequence[int]) -> Tuple[int, int]:
    if len(size) != 2:
        raise ValueError(f"Expected an output size of length 2, got {size}.")
    return int(size[0]), int(size[1])


def _resize_tensor(image: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    return F.interpolate(image.unsqueeze(0), size=size, mode=mode, align_corners=False if mode != "nearest" else None).squeeze(0)


def _resize_sample(image: torch.Tensor, label: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    image = _resize_tensor(image, size, mode="bilinear")
    label = _resize_tensor(label.unsqueeze(0).float(), size, mode="nearest").squeeze(0).long()
    return image, label


def dataset_uses_binary_masks(dataset_name: str) -> bool:
    return dataset_name.lower() in BINARY_MASK_DATASETS


def dataset_root_uses_binary_masks(base_dir: Union[str, Path]) -> bool:
    base_dir = Path(base_dir).expanduser().resolve()
    lowered_parts = tuple(part.lower() for part in base_dir.parts)
    return any(any(hint in part for hint in BINARY_MASK_PATH_HINTS) for part in lowered_parts)


def normalize_mask(mask: torch.Tensor, force_binary: bool = False) -> torch.Tensor:
    mask = mask.squeeze(0).long()
    unique_values = torch.unique(mask)
    if unique_values.numel() == 0:
        return mask
    if force_binary:
        if unique_values.max().item() <= 1:
            return mask
        # Recover binary labels from masks saved with JPEG artifacts or anti-aliased edges.
        return (mask >= BINARY_MASK_THRESHOLD).long()
    if unique_values.numel() <= 2 and unique_values.max().item() > 1:
        return (mask > 0).long()
    contiguous = torch.arange(unique_values.numel(), device=mask.device, dtype=unique_values.dtype)
    if unique_values.min().item() == 0 and torch.equal(unique_values, contiguous):
        return mask
    remapped = torch.zeros_like(mask)
    for class_index, value in enumerate(unique_values.tolist()):
        remapped[mask == value] = class_index
    return remapped.long()


def _normalize_mask(mask: torch.Tensor, force_binary: bool = False) -> torch.Tensor:
    return normalize_mask(mask, force_binary=force_binary)


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


def _scan_records(base_dir: Path) -> List[SampleRecord]:
    image_candidates: Dict[str, List[Path]] = {}
    mask_candidates: Dict[str, List[Path]] = {}

    for path in sorted(base_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        asset_type = _classify_asset(path)
        if asset_type is None:
            continue
        stem = path.stem
        if asset_type == "image":
            image_candidates.setdefault(stem, []).append(path)
        else:
            mask_candidates.setdefault(stem, []).append(path)

    records: List[SampleRecord] = []
    for case_id in sorted(set(image_candidates) & set(mask_candidates)):
        image_path = max(image_candidates[case_id], key=lambda path: _score_candidate(path, "image"))
        mask_path = max(mask_candidates[case_id], key=lambda path: _score_candidate(path, "mask"))
        records.append(SampleRecord(case_id=case_id, image_path=image_path, mask_path=mask_path))

    if not records:
        raise RuntimeError(
            f"Could not find any 2D image/mask pairs under '{base_dir}'. "
            "Expected folders such as 'images/masks' or 'Original/Ground Truth'."
        )
    return records


def scan_dataset_records(base_dir: Union[str, Path]) -> List[SampleRecord]:
    return _scan_records(Path(base_dir).expanduser().resolve())


def _parse_manifest(manifest_path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for raw_row in csv.reader(file):
            row = [item.strip() for item in raw_row if item.strip()]
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            rows.append(row)
    return rows


def find_manifest_path(base_dir: Union[str, Path], split: str) -> Optional[Path]:
    base_dir = Path(base_dir).expanduser().resolve()
    for subdir in MANIFEST_SEARCH_SUBDIRS:
        candidate = base_dir / subdir / f"{split}.txt" if subdir else base_dir / f"{split}.txt"
        if candidate.is_file():
            return candidate
    return None


def list_existing_splits(base_dir: Union[str, Path]) -> List[str]:
    return [split for split in ("train", "val", "test") if find_manifest_path(base_dir, split) is not None]


def _resolve_path(base_dir: Path, raw_path: str) -> Optional[Path]:
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    joined = (base_dir / candidate).resolve()
    return joined if joined.is_file() else None


def _resolve_record_from_manifest(row: List[str], pair_index: Dict[str, SampleRecord], base_dir: Path) -> Optional[SampleRecord]:
    header_tokens = {"case", "case_id", "image", "images", "id", "mask", "label"}
    if len(row) == 1:
        token = row[0]
        if token.lower() in header_tokens:
            return None
        return pair_index.get(Path(token).stem)

    image_path = _resolve_path(base_dir, row[0])
    mask_path = _resolve_path(base_dir, row[1])
    if image_path is not None and mask_path is not None:
        return SampleRecord(case_id=image_path.stem, image_path=image_path, mask_path=mask_path)

    image_key = Path(row[0]).stem
    mask_key = Path(row[1]).stem
    return pair_index.get(image_key) or pair_index.get(mask_key)


def _slice_records_for_split(records: List[SampleRecord], split: str, split_ratios: Tuple[float, float, float]) -> List[SampleRecord]:
    train_ratio, val_ratio, test_ratio = split_ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    train_ratio /= total_ratio
    val_ratio /= total_ratio
    test_ratio /= total_ratio

    total = len(records)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    if split == "train":
        return records[:train_end]
    if split == "val":
        return records[train_end:val_end]
    if split == "test":
        return records[val_end:]
    raise ValueError(f"Unsupported split '{split}'.")


def _resolve_records_for_split(base_dir: Path, split: str, split_ratios: Tuple[float, float, float]) -> List[SampleRecord]:
    all_records = _scan_records(base_dir)
    pair_index = {record.case_id: record for record in all_records}

    manifest_path = find_manifest_path(base_dir, split)
    if manifest_path is not None:
        rows = _parse_manifest(manifest_path)
        records = [
            record
            for row in rows
            for record in [_resolve_record_from_manifest(row, pair_index, base_dir)]
            if record is not None
        ]
        if records:
            return records

    available_manifest_names = ("train", "val", "test")
    manifests = {name: find_manifest_path(base_dir, name) for name in available_manifest_names}
    manifests = {name: path for name, path in manifests.items() if path is not None}
    if manifests:
        used_case_ids = set()
        for name, path in manifests.items():
            if name == split:
                continue
            for row in _parse_manifest(path):
                record = _resolve_record_from_manifest(row, pair_index, base_dir)
                if record is not None:
                    used_case_ids.add(record.case_id)
        if split == "test":
            leftover = [record for record in all_records if record.case_id not in used_case_ids]
            if leftover:
                return leftover

    sliced_records = _slice_records_for_split(all_records, split, split_ratios)
    if not sliced_records:
        raise RuntimeError(
            f"No samples found for split '{split}' in '{base_dir}'. "
            "Provide a split manifest or adjust the split ratios."
        )
    return sliced_records


class SegmentationDataset2D(Dataset):
    def __init__(
        self,
        base_dir: str,
        split: str = "train",
        transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        image_mode: str = "rgb",
        split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
        force_binary_masks: bool = False,
    ) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.split = split.lower()
        self.transform = transform or ToTensor()
        self.image_mode = image_mode.lower()
        self.split_ratios = tuple(float(value) for value in split_ratios)
        self.force_binary_masks = force_binary_masks or dataset_root_uses_binary_masks(self.base_dir)

        if self.image_mode not in {"rgb", "grayscale"}:
            raise ValueError("image_mode must be either 'rgb' or 'grayscale'.")
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Dataset root '{self.base_dir}' does not exist.")

        self.records = _resolve_records_for_split(self.base_dir, self.split, self.split_ratios)
        self.manifest_path = find_manifest_path(self.base_dir, self.split)
        self.in_channels = 3 if self.image_mode == "rgb" else 1

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        read_mode = ImageReadMode.RGB if self.image_mode == "rgb" else ImageReadMode.GRAY
        image = read_image(str(image_path), mode=read_mode).float() / 255.0
        return image

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        mask = read_image(str(mask_path), mode=ImageReadMode.GRAY)
        return normalize_mask(mask, force_binary=self.force_binary_masks)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        sample = {
            "image": self._load_image(record.image_path),
            "label": self._load_mask(record.mask_path),
            "case": record.case_id,
            "image_path": str(record.image_path),
            "label_path": str(record.mask_path),
        }
        return self.transform(sample)


class CVCClinicDB(SegmentationDataset2D):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("force_binary_masks", True)
        super().__init__(*args, **kwargs)


class KvasirSEG(SegmentationDataset2D):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("force_binary_masks", True)
        super().__init__(*args, **kwargs)


class Cyst2D(SegmentationDataset2D):
    pass


CVC_ClinicDB = CVCClinicDB
Kvasir_SEG = KvasirSEG


def list_available_datasets() -> List[str]:
    return ["cvc", "cvc_clinicdb", "kvasir", "kvasir_seg", "cyst2d", "generic"]


def build_dataset(
    dataset_name: str,
    base_dir: str,
    split: str,
    transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    image_mode: str = "rgb",
    split_ratios: Sequence[float] = DEFAULT_SPLIT_RATIOS,
) -> SegmentationDataset2D:
    normalized_name = dataset_name.lower()
    dataset_cls = {
        "cvc": CVCClinicDB,
        "cvc_clinicdb": CVCClinicDB,
        "kvasir": KvasirSEG,
        "kvasir_seg": KvasirSEG,
        "cyst2d": Cyst2D,
        "generic": SegmentationDataset2D,
    }.get(normalized_name, SegmentationDataset2D)
    return dataset_cls(
        base_dir=base_dir,
        split=split,
        transform=transform,
        image_mode=image_mode,
        split_ratios=split_ratios,
    )


class Normalize:
    def __init__(self, mean: Optional[Sequence[float]] = None, std: Optional[Sequence[float]] = None, eps: float = 1e-6) -> None:
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = sample["image"].float()
        if self.mean is None or self.std is None:
            mean = image.mean(dim=(1, 2), keepdim=True)
            std = image.std(dim=(1, 2), keepdim=True)
        else:
            mean = torch.tensor(self.mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
            std = torch.tensor(self.std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
        sample["image"] = (image - mean) / (std + self.eps)
        return sample


class Resize:
    def __init__(self, output_size: Sequence[int]) -> None:
        self.output_size = _ensure_hw(output_size)

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image, label = _resize_sample(sample["image"], sample["label"], self.output_size)
        sample["image"] = image
        sample["label"] = label
        return sample


class RandomGenerator:
    def __init__(
        self,
        output_size: Sequence[int],
        p_horizontal_flip: float = 0.5,
        p_vertical_flip: float = 0.5,
        p_rot90: float = 0.5,
        p_rotate: float = 0.25,
        max_rotate_degree: float = 20.0,
    ) -> None:
        self.output_size = _ensure_hw(output_size)
        self.p_horizontal_flip = p_horizontal_flip
        self.p_vertical_flip = p_vertical_flip
        self.p_rot90 = p_rot90
        self.p_rotate = p_rotate
        self.max_rotate_degree = max_rotate_degree

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = sample["image"]
        label = sample["label"]

        if random.random() < self.p_horizontal_flip:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])
        if random.random() < self.p_vertical_flip:
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[0])
        if random.random() < self.p_rot90:
            k = random.randint(0, 3)
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            label = TF.rotate(label.unsqueeze(0).float(), angle, interpolation=InterpolationMode.NEAREST).squeeze(0).long()

        image, label = _resize_sample(image, label, self.output_size)
        sample["image"] = image.float()
        sample["label"] = label.long()
        return sample


class RandomNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.05, p: float = 0.5) -> None:
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            noise = torch.randn_like(sample["image"]) * self.std + self.mean
            sample["image"] = torch.clamp(sample["image"] + noise, min=0.0, max=1.0)
        return sample


class RandomBlur:
    def __init__(self, kernel_size: int = 5, sigma: Tuple[float, float] = (0.1, 2.0), p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma)
            sample["image"] = TF.gaussian_blur(sample["image"], kernel_size=self.kernel_size, sigma=sigma)
        return sample


class RandomGamma:
    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5), p: float = 0.5) -> None:
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            sample["image"] = torch.clamp(sample["image"], min=0.0).pow(gamma)
        return sample


class CreateOnehotLabel:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        label = sample["label"].long()
        sample["onehot_label"] = F.one_hot(label, num_classes=self.num_classes).permute(2, 0, 1).float()
        return sample


class ToTensor:
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample["image"] = sample["image"].float()
        sample["label"] = sample["label"].long()
        if "onehot_label" in sample:
            sample["onehot_label"] = sample["onehot_label"].float()
        return sample
