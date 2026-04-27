from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from networks.PGD_Unet.gated_unet import PDGUNet
from networks.PGD_Unet.middle_pruned_resnet_unet import build_middle_pruned_resnet_unet
from networks.PGD_Unet.pruning import load_blueprint_artifact
from utils.checkpoints import load_checkpoint, load_checkpoint_into_model
from utils.model_output import extract_logits


DEFAULT_DATASET = "kvasir_seg"
DEFAULT_TEACHER = "unet_resnet152_teacher"
DEFAULT_LOSS_TAG = "loss_seg_kd_sparsity"
DEFAULT_OUTPUT_DIR = "output_kneedle_auto_no"
DEFAULT_PATCH_SIZE = (256, 256)


def _default_run_dir(dataset: str = DEFAULT_DATASET, output_dir: str = DEFAULT_OUTPUT_DIR) -> Path:
    return PROJECT_ROOT / "outputs" / "pgd_unet" / dataset / DEFAULT_TEACHER / DEFAULT_LOSS_TAG / output_dir


def _default_checkpoint_path(dataset: str = DEFAULT_DATASET, output_dir: str = DEFAULT_OUTPUT_DIR) -> Path:
    app_checkpoint = PROJECT_ROOT / "app" / "checkpoints" / "best.pth"
    if app_checkpoint.is_file():
        return app_checkpoint
    run_dir = _default_run_dir(dataset, output_dir)
    best = run_dir / "3_student" / "checkpoints" / "best.pth"
    return best if best.is_file() else run_dir / "3_student" / "checkpoints" / "last.pth"


def _default_metadata_path(checkpoint_path: Path) -> Optional[Path]:
    candidates = [
        checkpoint_path.parent / "metadata" / f"{checkpoint_path.stem}.json",
        PROJECT_ROOT / "app" / "checkpoints" / "metadata" / f"{checkpoint_path.stem}.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _default_blueprint_path(dataset: str = DEFAULT_DATASET, output_dir: str = DEFAULT_OUTPUT_DIR) -> Path:
    return _default_run_dir(dataset, output_dir) / "2_pruning" / "artifacts" / "blueprint.json"


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_blueprint(
    *,
    blueprint_path: Optional[Path],
    metadata: Dict[str, Any],
    checkpoint_payload: Dict[str, Any],
) -> Dict[str, Any]:
    if blueprint_path is not None and blueprint_path.is_file():
        return load_blueprint_artifact(blueprint_path)

    for source in (
        metadata.get("extra_state", {}),
        checkpoint_payload.get("extra_state", {}),
        metadata,
        checkpoint_payload,
    ):
        blueprint = source.get("blueprint") if isinstance(source, dict) else None
        if isinstance(blueprint, dict):
            if "channel_config" in blueprint:
                blueprint = dict(blueprint)
                blueprint["channel_config"] = tuple(int(value) for value in blueprint["channel_config"])
            return blueprint
    return {}


def _resolve_model_config(metadata: Dict[str, Any], checkpoint_payload: Dict[str, Any]) -> Dict[str, Any]:
    config = {}
    for source in (metadata.get("config"), checkpoint_payload.get("config")):
        if isinstance(source, dict):
            config.update(source)
    model_info = {}
    for source in (metadata.get("model_info"), checkpoint_payload.get("model_info")):
        if isinstance(source, dict):
            model_info.update(source)
    architecture_config = model_info.get("architecture_config", {}) if isinstance(model_info.get("architecture_config"), dict) else {}
    return {
        "config": config,
        "model_info": model_info,
        "architecture_config": architecture_config,
    }


def _build_student_model(
    *,
    blueprint: Dict[str, Any],
    metadata: Dict[str, Any],
    checkpoint_payload: Dict[str, Any],
):
    resolved = _resolve_model_config(metadata, checkpoint_payload)
    config = resolved["config"]
    architecture_config = resolved["architecture_config"]
    in_channels = int(config.get("in_channels", architecture_config.get("in_channels", 3)))
    num_classes = int(config.get("num_classes", architecture_config.get("num_classes", 2)))

    student_architecture = str(blueprint.get("student_architecture", "")).lower()
    if student_architecture == "middle_pruned_resnet_unet":
        return build_middle_pruned_resnet_unet(
            in_channels=in_channels,
            num_classes=num_classes,
            blueprint=blueprint,
        )

    channel_config = blueprint.get("channel_config") or architecture_config.get("channel_config")
    if not channel_config:
        raise RuntimeError(
            "Cannot build PGD student: missing channel_config. "
            "Provide blueprint.json or a checkpoint metadata file containing the blueprint."
        )
    return PDGUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        channel_config=tuple(int(value) for value in channel_config),
        normalization=str(architecture_config.get("normalization", "batchnorm")),
    )


@st.cache_resource(show_spinner="Loading PGD-UNet student...")
def load_model_cached(
    checkpoint_path_text: str,
    blueprint_path_text: str,
    device_name: str,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path_text).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    blueprint_path: Optional[Path] = None
    if blueprint_path_text.strip():
        candidate = Path(blueprint_path_text).expanduser()
        blueprint_path = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()

    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    metadata = _read_json(_default_metadata_path(checkpoint_path))
    checkpoint_payload = load_checkpoint(checkpoint_path, device=device)
    blueprint = _load_blueprint(
        blueprint_path=blueprint_path,
        metadata=metadata,
        checkpoint_payload=checkpoint_payload,
    )
    model = _build_student_model(
        blueprint=blueprint,
        metadata=metadata,
        checkpoint_payload=checkpoint_payload,
    ).to(device)

    try:
        load_checkpoint_into_model(checkpoint_path, model, device=device, strict=True)
        strict_loaded = True
    except RuntimeError:
        load_checkpoint_into_model(checkpoint_path, model, device=device, strict=False)
        strict_loaded = False
    model.eval()

    info = {
        "checkpoint_path": str(checkpoint_path),
        "blueprint_path": str(blueprint_path) if blueprint_path else "metadata/extra_state",
        "device": str(device),
        "strict_loaded": strict_loaded,
        "channel_config": list(blueprint.get("channel_config", [])),
        "student_architecture": blueprint.get("student_architecture", "pdg_unet"),
    }
    return model, info


def preprocess_image(image: Image.Image, patch_size: Tuple[int, int], in_channels: int = 3) -> torch.Tensor:
    image = image.convert("RGB" if in_channels == 3 else "L")
    resized = image.resize((int(patch_size[1]), int(patch_size[0])), Image.BILINEAR)
    array = np.asarray(resized).astype(np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = tensor.mean(dim=(1, 2), keepdim=True)
    std = tensor.std(dim=(1, 2), keepdim=True)
    tensor = (tensor - mean) / (std + 1e-6)
    return tensor.unsqueeze(0)


def predict_mask(
    model: torch.nn.Module,
    image: Image.Image,
    *,
    patch_size: Tuple[int, int],
    threshold: float,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    original_size = image.size
    model_info = getattr(model, "get_model_info", lambda: {})()
    architecture_config = model_info.get("architecture_config", {}) if isinstance(model_info, dict) else {}
    in_channels = int(architecture_config.get("in_channels", 3))

    input_tensor = preprocess_image(image, patch_size, in_channels=in_channels).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model(input_tensor)
        logits = extract_logits(output)
        if logits.shape[1] == 1:
            prob = torch.sigmoid(logits)[:, 0:1]
            pred = (prob >= threshold).float()
        else:
            probs = torch.softmax(logits, dim=1)
            prob = probs[:, 1:2] if probs.shape[1] > 1 else probs[:, 0:1]
            pred = torch.argmax(probs, dim=1, keepdim=True).float()
        prob = F.interpolate(prob, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)
        pred = F.interpolate(pred, size=(original_size[1], original_size[0]), mode="nearest")
    elapsed = time.perf_counter() - start
    prob_np = prob.squeeze().detach().cpu().numpy()
    mask_np = (pred.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    return mask_np, prob_np, elapsed


def make_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGBA")
    mask_bool = mask > 0
    color = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    color[..., 0] = 255
    color[..., 1] = 64
    color[..., 3] = (mask_bool.astype(np.float32) * 255 * alpha).astype(np.uint8)
    overlay = Image.fromarray(color, mode="RGBA")
    return Image.alpha_composite(base, overlay)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="PGD-UNet Inference", layout="wide")
    st.title("PGD-UNet Inference")

    with st.sidebar:
        st.subheader("Model")
        dataset = st.text_input("Dataset", value=DEFAULT_DATASET)
        output_dir = st.text_input("Output folder", value=DEFAULT_OUTPUT_DIR)
        checkpoint_default = _default_checkpoint_path(dataset, output_dir)
        blueprint_default = _default_blueprint_path(dataset, output_dir)
        checkpoint_path = st.text_input("Checkpoint path", value=str(checkpoint_default.relative_to(PROJECT_ROOT) if checkpoint_default.is_absolute() and PROJECT_ROOT in checkpoint_default.parents else checkpoint_default))
        blueprint_path = st.text_input("Blueprint path", value=str(blueprint_default.relative_to(PROJECT_ROOT) if blueprint_default.is_absolute() and PROJECT_ROOT in blueprint_default.parents else blueprint_default))
        device_name = st.selectbox("Device", options=["cuda:0", "cpu"], index=0 if torch.cuda.is_available() else 1)
        patch_h = st.number_input("Patch height", min_value=64, max_value=1024, value=DEFAULT_PATCH_SIZE[0], step=32)
        patch_w = st.number_input("Patch width", min_value=64, max_value=1024, value=DEFAULT_PATCH_SIZE[1], step=32)
        threshold = st.slider("Mask threshold", min_value=0.05, max_value=0.95, value=0.5, step=0.05)

    try:
        model, info = load_model_cached(checkpoint_path, blueprint_path, device_name)
    except Exception as error:
        st.error(f"Cannot load model: {error}")
        st.stop()

    with st.expander("Loaded model", expanded=False):
        st.json(info)

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    if uploaded is None:
        st.info("Upload an endoscopy image to run inference.")
        return

    image = Image.open(uploaded)
    mask, prob, elapsed = predict_mask(
        model,
        image,
        patch_size=(int(patch_h), int(patch_w)),
        threshold=float(threshold),
        device=torch.device(info["device"]),
    )
    overlay = make_overlay(image, mask)
    mask_image = Image.fromarray(mask, mode="L")
    prob_image = Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8), mode="L")

    st.caption(f"Inference time: {elapsed:.4f} s | Mask area: {(mask > 0).mean() * 100:.2f}%")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(image, caption="Input", use_container_width=True)
    with col2:
        st.image(mask_image, caption="Mask", use_container_width=True)
    with col3:
        st.image(prob_image, caption="Probability", use_container_width=True)
    with col4:
        st.image(overlay, caption="Overlay", use_container_width=True)

    down1, down2 = st.columns(2)
    with down1:
        st.download_button("Download mask PNG", image_to_png_bytes(mask_image), file_name="pgd_unet_mask.png", mime="image/png")
    with down2:
        st.download_button("Download overlay PNG", image_to_png_bytes(overlay), file_name="pgd_unet_overlay.png", mime="image/png")


if __name__ == "__main__":
    main()
