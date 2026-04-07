# networks/net_factory.py
from typing import Dict, List, Type

# Import tất cả các model có sẵn
from networks.unet import UNet2D
from networks.Unet_restnet import UNetResNet152
from networks.residual_unet import ResidualUNet2D
from networks.VNet import VNet2D
from networks.unetr import UNETR2D

# ================== MODEL MỚI (Student) ==================
from networks.gated_unet import GatedUNet   

# ====================== REGISTRY ======================
MODEL_REGISTRY: Dict[str, Type] = {
    "unet": UNet2D,
    "unet_resnet152": UNetResNet152,
    "resunet": ResidualUNet2D,
    "vnet": VNet2D,
    "unetr": UNETR2D,
    "gated_unet": GatedUNet,          # Model Student mới
}

MODEL_ALIASES = {
    "u-net": "unet",
    "unet2d": "unet",
    "unet_restnet": "unet_resnet152",
    "unet_restnet152": "unet_resnet152",
    "unet_resnet": "unet_resnet152",
    "resnet_unet": "unet_resnet152",
    "res_unet": "resunet",
    "resunet2d": "resunet",
    "residualunet": "resunet",
    "residual_unet": "resunet",
    "residual-u-net": "resunet",
    "residual_unet2d": "resunet",
    "v_net": "vnet",
    "vnet2d": "vnet",
    "VNet": "vnet",
    "UNETR": "unetr",
    "unetr2d": "unetr",
    # Alias cho Student
    "gatedunet": "gated_unet",
    "student": "gated_unet",
    "gated_unet": "gated_unet",
}


def _normalize_model_name(model_name: str) -> str:
    normalized_name = MODEL_ALIASES.get(model_name, model_name)
    normalized_name = MODEL_ALIASES.get(normalized_name.lower(), normalized_name.lower())
    if normalized_name not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{model_name}'. Available models: {available_models}.")
    return normalized_name


def list_models() -> List[str]:
    return sorted(MODEL_REGISTRY)


def net_factory(net_type: str = "unet", in_chns: int = 3, class_num: int = 2, mode: str = "train", **kwargs):
    model_name = _normalize_model_name(net_type)
    kwargs.pop("tsne", None)

    model_kwargs = {
        "in_channels": in_chns,
        "num_classes": class_num,
    }

    if model_name == "vnet" and "has_dropout" not in kwargs:
        model_kwargs["has_dropout"] = mode.lower() == "train"
    if model_name == "unetr":
        if "img_size" in kwargs and "image_size" not in kwargs:
            kwargs["image_size"] = kwargs.pop("img_size")
    if model_name == "unet_resnet152" and "encoder_pretrained" not in kwargs:
        model_kwargs["encoder_pretrained"] = False

    model_kwargs.update(kwargs)
    return MODEL_REGISTRY[model_name](**model_kwargs)