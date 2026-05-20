from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image


DEFAULT_PALETTE = (
    (0, 0, 0),
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
)


def _normalize_image(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu().float()
    if image.ndim == 2:
        image = image.unsqueeze(0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3]
    image_min = image.min()
    image_max = image.max()
    if (image_max - image_min) > 1e-8:
        image = (image - image_min) / (image_max - image_min)
    else:
        image = torch.zeros_like(image)
    return image.clamp(0.0, 1.0)


def colorize_mask(mask: torch.Tensor, palette: Sequence[Sequence[int]] = DEFAULT_PALETTE) -> torch.Tensor:
    mask = mask.detach().cpu().long()
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    color = torch.zeros(3, mask.shape[-2], mask.shape[-1], dtype=torch.float32)
    for class_index in torch.unique(mask).tolist():
        rgb = palette[class_index % len(palette)]
        for channel, value in enumerate(rgb):
            color[channel][mask == class_index] = value / 255.0
    return color


def _label_strip(width: int, height: int = 28, labels: Sequence[str] = ("Image", "GT", "PR")) -> Image.Image:
    strip = Image.new("RGB", (int(width), int(height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(strip)
    font = ImageFont.load_default()
    column_width = int(width) / max(1, len(labels))
    for index, label in enumerate(labels):
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(label, font=font)
        x = int(index * column_width + (column_width - text_width) / 2)
        y = int((height - text_height) / 2)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    return strip


def _save_labeled_triplet_panel(panel: torch.Tensor, output_path: Path) -> None:
    panel_image = to_pil_image(panel.detach().cpu().clamp(0.0, 1.0))
    strip = _label_strip(panel_image.width)
    canvas = Image.new("RGB", (panel_image.width, panel_image.height + strip.height), color=(255, 255, 255))
    canvas.paste(strip, (0, 0))
    canvas.paste(panel_image.convert("RGB"), (0, strip.height))
    canvas.save(output_path)


def save_triplet_visualization(
    image: torch.Tensor,
    label: torch.Tensor,
    prediction: torch.Tensor,
    output_dir: Path,
    case_name: str,
    palette: Sequence[Sequence[int]] = DEFAULT_PALETTE,
) -> None:
    output_dir = Path(output_dir)
    image_dir = output_dir / "image"
    gt_dir = output_dir / "gt"
    pred_dir = output_dir / "pred"
    panel_dir = output_dir / "panel"
    for directory in (image_dir, gt_dir, pred_dir, panel_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_vis = _normalize_image(image)
    label_vis = colorize_mask(label, palette=palette)
    pred_vis = colorize_mask(prediction, palette=palette)

    save_image(image_vis, image_dir / f"{case_name}.png")
    save_image(label_vis, gt_dir / f"{case_name}.png")
    save_image(pred_vis, pred_dir / f"{case_name}.png")
    _save_labeled_triplet_panel(torch.cat([image_vis, label_vis, pred_vis], dim=2), panel_dir / f"{case_name}.png")
