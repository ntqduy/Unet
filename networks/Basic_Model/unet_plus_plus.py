from __future__ import annotations

from typing import Any

import torch

from utils.model_output import BaseSegmentationModel
from utils.pretrained_cache import ensure_pretrain_cache


class UNetPlusPlus2D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        encoder_name: str = "resnet152",
        encoder_weights: str | None = "imagenet",
        encoder_pretrained: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if encoder_pretrained is not None:
            encoder_weights = "imagenet" if bool(encoder_pretrained) else None
        if encoder_weights:
            ensure_pretrain_cache()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise ImportError(
                "UNetPlusPlus2D requires segmentation-models-pytorch. "
                "Install it with `pip install segmentation-models-pytorch` or `pip install -r requirements.txt`."
            ) from error
        self.model_name = "unet_plus_plus"
        self.backbone_name = encoder_name
        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            **kwargs,
        )
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits = self.model(x)
        output = self.build_output(
            logits,
            features={"smp_model": "UnetPlusPlus", "encoder_name": self.backbone_name},
        )
        if return_features:
            return output
        return output


UNetPlusPlus = UNetPlusPlus2D
