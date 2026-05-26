from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from networks.PGD_Unet.middle_pruned_unet_plus_plus import build_middle_pruned_unet_plus_plus
from networks.net_factory import net_factory
from utils.model_output import extract_logits


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_IMAGE_PATH = Path(
    "/data2/Medical/StrokeCT-MultiModal/Multi_Modal/Stroke_MultiModal_Predict/"
    "segmentation/Unet/outputs/pgd_unet/cvc_colondb/unet_resnet152_teacher/"
    "loss_seg_kd/output_s6_middle_kneedle_auto_no/3_student/evaluations/test/image/10.png"
)

DEFAULT_STUDENT_CHECKPOINT = Path(
    "/data2/Medical/StrokeCT-MultiModal/Multi_Modal/Stroke_MultiModal_Predict/"
    "segmentation/Unet/outputs/pgd_unet/cvc_colondb/unet_resnet152_teacher/"
    "loss_seg_kd/output_s6_middle_kneedle_auto_no/3_student/checkpoints/best.pth"
)

DEFAULT_TEACHER_CHECKPOINT = Path(
    "/data2/Medical/StrokeCT-MultiModal/Multi_Modal/Stroke_MultiModal_Predict/"
    "segmentation/Unet/outputs/pgd_unet/cvc_colondb/unet_resnet152_teacher/"
    "1_teacher/checkpoints/best.pth"
)


def load_checkpoint_payload(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"model_state_dict": payload}


def clean_state_dict(payload: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    state_dict = payload.get("model_state_dict", payload)
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a valid state_dict.")

    cleaned = {}
    for key, value in state_dict.items():
        if key.rsplit(".", 1)[-1] in {"total_ops", "total_params"}:
            continue
        cleaned[str(key)] = value

    if cleaned and all(key.startswith("module.") for key in cleaned):
        cleaned = {key[len("module.") :]: value for key, value in cleaned.items()}
    return cleaned


def infer_int(payload: Mapping[str, Any], key: str, default: int) -> int:
    config = dict(payload.get("config", {}) or {})
    model_info = dict(payload.get("model_info", {}) or {})
    architecture_config = dict(model_info.get("architecture_config", {}) or {})

    for source in (config, architecture_config, model_info):
        value = source.get(key)
        if value is not None:
            return int(value)
    return int(default)


def infer_patch_size(payloads: Sequence[Mapping[str, Any]], default: Sequence[int] = (256, 256)) -> tuple[int, int]:
    for payload in payloads:
        config = dict(payload.get("config", {}) or {})
        value = config.get("patch_size")
        if value is not None and len(value) >= 2:
            return int(value[0]), int(value[1])
    return int(default[0]), int(default[1])


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_blueprint(student_checkpoint: Path, student_payload: Mapping[str, Any]) -> Dict[str, Any]:
    extra_state = dict(student_payload.get("extra_state", {}) or {})
    if isinstance(extra_state.get("blueprint"), Mapping):
        return dict(extra_state["blueprint"])

    model_info = dict(student_payload.get("model_info", {}) or {})
    if isinstance(model_info.get("blueprint"), Mapping):
        return dict(model_info["blueprint"])

    candidates = []
    try:
        output_dir = student_checkpoint.parents[2]
        candidates.extend(
            [
                output_dir / "2_pruning" / "artifacts" / "blueprint.json",
                output_dir / "2_pruning" / "configs" / "pruning_config.json",
                output_dir / "2_pruning" / "metrics" / "pruning_summary.json",
            ]
        )
    except IndexError:
        pass

    for candidate in candidates:
        blueprint = load_json(candidate)
        if blueprint is not None:
            return blueprint

    raise FileNotFoundError(
        "Could not find middle-kneedle blueprint. Expected it inside the student checkpoint "
        "extra_state/model_info, or under 2_pruning/artifacts/blueprint.json."
    )


def build_teacher_model(payload: Mapping[str, Any], in_channels: int, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = net_factory(
        net_type="unet_resnet152",
        in_chns=in_channels,
        class_num=num_classes,
        mode="test",
        encoder_pretrained=False,
    )
    model.load_state_dict(clean_state_dict(payload), strict=True)
    model.to(device)
    model.eval()
    return model


def build_middle_kneedle_student(
    payload: Mapping[str, Any],
    checkpoint_path: Path,
    in_channels: int,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    blueprint = find_blueprint(checkpoint_path, payload)
    model = build_middle_pruned_unet_plus_plus(
        in_channels=in_channels,
        num_classes=num_classes,
        blueprint=blueprint,
    )
    model.load_state_dict(clean_state_dict(payload), strict=True)
    model.to(device)
    model.eval()
    return model


def load_image_tensor(image_path: Path, in_channels: int) -> tuple[torch.Tensor, tuple[int, int]]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mode = "RGB" if in_channels == 3 else "L"
    image = Image.open(image_path).convert(mode)
    array = np.asarray(image).astype(np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]

    tensor = torch.from_numpy(array).permute(2, 0, 1).float()
    mean = tensor.mean(dim=(1, 2), keepdim=True)
    std = tensor.std(dim=(1, 2), keepdim=True)
    tensor = (tensor - mean) / (std + 1e-6)
    return tensor.unsqueeze(0), (image.height, image.width)


@torch.no_grad()
def foreground_logit(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: tuple[int, int],
    original_size: tuple[int, int],
    class_index: int,
    device: torch.device,
) -> torch.Tensor:
    resized = F.interpolate(image.to(device), size=patch_size, mode="bilinear", align_corners=False)
    logits = extract_logits(model(resized))

    if logits.shape[1] == 1:
        logit = logits[:, 0:1]
    else:
        if class_index >= logits.shape[1]:
            raise ValueError(f"class_index={class_index} is invalid for logits with {logits.shape[1]} channels.")
        logit = logits[:, class_index : class_index + 1]

    logit = F.interpolate(logit, size=original_size, mode="bilinear", align_corners=False)
    return logit[0, 0].detach().cpu()


def save_heatmap(logit: torch.Tensor, output_path: Path, *, vmin: float, vmax: float, cmap: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, logit.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save teacher/student foreground-logit heatmaps for one image.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_STUDENT_CHECKPOINT)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "logit_vis")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--in-channels", type=int, default=None)
    parser.add_argument("--patch-size", nargs=2, type=int, default=None)
    parser.add_argument("--class-index", type=int, default=1, help="Foreground class index for 2-class logits.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cmap", type=str, default="magma")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    student_payload = load_checkpoint_payload(args.student_checkpoint, device)
    teacher_payload = load_checkpoint_payload(args.teacher_checkpoint, device)

    in_channels = int(args.in_channels or infer_int(student_payload, "in_channels", 3))
    num_classes = int(args.num_classes or infer_int(student_payload, "num_classes", 2))
    patch_size = tuple(args.patch_size) if args.patch_size is not None else infer_patch_size((student_payload, teacher_payload))

    image, original_size = load_image_tensor(args.image_path, in_channels)

    teacher = build_teacher_model(teacher_payload, in_channels, num_classes, device)
    student = build_middle_kneedle_student(student_payload, args.student_checkpoint, in_channels, num_classes, device)

    teacher_logit = foreground_logit(teacher, image, patch_size, original_size, args.class_index, device)
    student_logit = foreground_logit(student, image, patch_size, original_size, args.class_index, device)

    vmin = float(min(teacher_logit.min().item(), student_logit.min().item()))
    vmax = float(max(teacher_logit.max().item(), student_logit.max().item()))
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1e-8

    stem = args.image_path.stem
    teacher_path = args.output_dir / f"{stem}_teacher_unet_resnet152_logit.png"
    student_path = args.output_dir / f"{stem}_student_middle_kneedle_logit.png"

    save_heatmap(teacher_logit, teacher_path, vmin=vmin, vmax=vmax, cmap=args.cmap)
    save_heatmap(student_logit, student_path, vmin=vmin, vmax=vmax, cmap=args.cmap)

    print(f"Saved teacher logit: {teacher_path}")
    print(f"Saved student logit: {student_path}")


if __name__ == "__main__":
    main()
