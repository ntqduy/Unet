from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from networks.PGD_Unet.gated_unet import PlainDoubleConv2d, PlainDownBlock
from utils.model_output import BaseSegmentationModel


def _resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape[-2:] == target.shape[-2:]:
        return source
    return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)


class BlueprintUNetPlusPlus(BaseSegmentationModel):
    """Small UNet++ student built from a 5-stage pruning blueprint."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        channel_config: Sequence[int] = (32, 64, 128, 256, 512),
        normalization: str = "batchnorm",
    ) -> None:
        super().__init__()
        channels = tuple(int(channel) for channel in channel_config)
        if len(channels) != 5:
            raise ValueError("BlueprintUNetPlusPlus expects exactly 5 stages in channel_config.")

        c0, c1, c2, c3, c4 = channels
        self.model_name = "blueprint_unet_plus_plus"
        self.backbone_name = "unet_plus_plus_blueprint_encoder"
        self.student_name = "blueprint_unet_plus_plus_student"
        self.channel_config = channels

        self.stem = PlainDoubleConv2d(in_channels, c0, normalization=normalization)
        self.down1 = PlainDownBlock(c0, c1, normalization=normalization)
        self.down2 = PlainDownBlock(c1, c2, normalization=normalization)
        self.down3 = PlainDownBlock(c2, c3, normalization=normalization)
        self.down4 = PlainDownBlock(c3, c4, normalization=normalization)

        self.decoder = nn.ModuleDict(
            {
                "x0_1": PlainDoubleConv2d(c0 + c1, c0, normalization=normalization),
                "x1_1": PlainDoubleConv2d(c1 + c2, c1, normalization=normalization),
                "x2_1": PlainDoubleConv2d(c2 + c3, c2, normalization=normalization),
                "x3_1": PlainDoubleConv2d(c3 + c4, c3, normalization=normalization),
                "x0_2": PlainDoubleConv2d(c0 * 2 + c1, c0, normalization=normalization),
                "x1_2": PlainDoubleConv2d(c1 * 2 + c2, c1, normalization=normalization),
                "x2_2": PlainDoubleConv2d(c2 * 2 + c3, c2, normalization=normalization),
                "x0_3": PlainDoubleConv2d(c0 * 3 + c1, c0, normalization=normalization),
                "x1_3": PlainDoubleConv2d(c1 * 3 + c2, c1, normalization=normalization),
                "x0_4": PlainDoubleConv2d(c0 * 4 + c1, c0, normalization=normalization),
            }
        )
        self.head = nn.Conv2d(c0, num_classes, kernel_size=1)

        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            channel_config=list(channels),
            normalization=normalization,
            student_variant="blueprint_unet_plus_plus",
            decoder_architecture="unet_plus_plus",
            uses_gates=False,
        )

    def forward_features(self, x: torch.Tensor):
        x0_0 = self.stem(x)
        x1_0 = self.down1(x0_0)
        x2_0 = self.down2(x1_0)
        x3_0 = self.down3(x2_0)
        x4_0 = self.down4(x3_0)

        x0_1 = self.decoder["x0_1"](torch.cat([x0_0, _resize_like(x1_0, x0_0)], dim=1))
        x1_1 = self.decoder["x1_1"](torch.cat([x1_0, _resize_like(x2_0, x1_0)], dim=1))
        x2_1 = self.decoder["x2_1"](torch.cat([x2_0, _resize_like(x3_0, x2_0)], dim=1))
        x3_1 = self.decoder["x3_1"](torch.cat([x3_0, _resize_like(x4_0, x3_0)], dim=1))

        x0_2 = self.decoder["x0_2"](torch.cat([x0_0, x0_1, _resize_like(x1_1, x0_0)], dim=1))
        x1_2 = self.decoder["x1_2"](torch.cat([x1_0, x1_1, _resize_like(x2_1, x1_0)], dim=1))
        x2_2 = self.decoder["x2_2"](torch.cat([x2_0, x2_1, _resize_like(x3_1, x2_0)], dim=1))

        x0_3 = self.decoder["x0_3"](torch.cat([x0_0, x0_1, x0_2, _resize_like(x1_2, x0_0)], dim=1))
        x1_3 = self.decoder["x1_3"](torch.cat([x1_0, x1_1, x1_2, _resize_like(x2_2, x1_0)], dim=1))

        x0_4 = self.decoder["x0_4"](torch.cat([x0_0, x0_1, x0_2, x0_3, _resize_like(x1_3, x0_0)], dim=1))
        logits = self.head(x0_4)
        features = {
            "bottleneck": x4_0,
            "encoder": {"stem": x0_0, "down1": x1_0, "down2": x2_0, "down3": x3_0, "down4": x4_0},
            "decoder": {
                "x0_1": x0_1,
                "x1_1": x1_1,
                "x2_1": x2_1,
                "x3_1": x3_1,
                "x0_2": x0_2,
                "x1_2": x1_2,
                "x2_2": x2_2,
                "x0_3": x0_3,
                "x1_3": x1_3,
                "x0_4": x0_4,
                "final": x0_4,
            },
        }
        return logits, features

    def forward(self, x: torch.Tensor):
        logits, features = self.forward_features(x)
        aux_logits = {
            name: feature.mean(dim=1, keepdim=True).repeat(1, logits.shape[1], 1, 1)
            for name, feature in features["decoder"].items()
            if name in {"x0_1", "x0_2", "x0_3", "x0_4"}
        }
        return self.build_output(
            logits,
            features=features,
            aux={
                "channel_config": list(self.channel_config),
                "aux_logits": aux_logits,
            },
        )

    def get_gate_tensors(self) -> List[torch.Tensor]:
        return []

    def get_gate_modules(self) -> List[nn.Module]:
        return []

    def set_gate_trainable(self, trainable: bool) -> None:
        return None

    def force_gates_open(self, open_probability: float = 0.999) -> None:
        return None

    def get_gate_statistics(self) -> Dict[str, List[float]]:
        return {"gate_means": [], "gate_sparsity_proxy": []}


def build_blueprint_unet_plus_plus(
    *,
    in_channels: int = 3,
    num_classes: int = 2,
    channel_config: Sequence[int] = (32, 64, 128, 256, 512),
    normalization: str = "batchnorm",
) -> BlueprintUNetPlusPlus:
    return BlueprintUNetPlusPlus(
        in_channels=in_channels,
        num_classes=num_classes,
        channel_config=channel_config,
        normalization=normalization,
    )

