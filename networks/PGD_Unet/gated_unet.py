from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from networks.Basic_Model.common import DoubleConv2d
from utils.model_output import BaseSegmentationModel


class PlainDoubleConv2d(nn.Module):
    """Compatibility wrapper around DoubleConv2d without learnable gates."""

    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.conv = DoubleConv2d(in_channels, out_channels, normalization=normalization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def gate_values(self) -> torch.Tensor:
        return torch.empty(0)


class PlainDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = PlainDoubleConv2d(in_channels, out_channels, normalization=normalization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class PlainUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = PlainDoubleConv2d(out_channels + skip_channels, out_channels, normalization=normalization)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PDGUNet(BaseSegmentationModel):
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
            raise ValueError("PDGUNet expects exactly 5 stages in channel_config.")

        self.model_name = "pdg_unet"
        self.backbone_name = "pruned_teacher_blueprint"
        self.student_name = "pruned_unet_student"
        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            channel_config=list(channels),
            normalization=normalization,
            student_variant="plain_pruned_unet",
            uses_gates=False,
        )

        self.channel_config = channels
        self.stem = PlainDoubleConv2d(in_channels, channels[0], normalization=normalization)
        self.down1 = PlainDownBlock(channels[0], channels[1], normalization=normalization)
        self.down2 = PlainDownBlock(channels[1], channels[2], normalization=normalization)
        self.down3 = PlainDownBlock(channels[2], channels[3], normalization=normalization)
        self.down4 = PlainDownBlock(channels[3], channels[4], normalization=normalization)

        self.up1 = PlainUpBlock(channels[4], channels[3], channels[3], normalization=normalization)
        self.up2 = PlainUpBlock(channels[3], channels[2], channels[2], normalization=normalization)
        self.up3 = PlainUpBlock(channels[2], channels[1], channels[1], normalization=normalization)
        self.up4 = PlainUpBlock(channels[1], channels[0], channels[0], normalization=normalization)

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward_features(self, x: torch.Tensor):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        up1 = self.up1(x4, x3)
        up2 = self.up2(up1, x2)
        up3 = self.up3(up2, x1)
        decoder_features = self.up4(up3, x0)
        logits = self.head(decoder_features)
        features = {
            "bottleneck": x4,
            "encoder": {"stem": x0, "down1": x1, "down2": x2, "down3": x3, "down4": x4},
            "decoder": {"up1": up1, "up2": up2, "up3": up3, "up4": decoder_features, "final": decoder_features},
        }
        return logits, features

    def forward(self, x: torch.Tensor):
        logits, features = self.forward_features(x)
        aux_logits = {
            name: feature.mean(dim=1, keepdim=True).repeat(1, logits.shape[1], 1, 1)
            for name, feature in features["decoder"].items()
            if name in {"up1", "up2", "up3", "up4"}
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


GatedDoubleConv2d = PlainDoubleConv2d
GatedDownBlock = PlainDownBlock
GatedUpBlock = PlainUpBlock
GatedUNet = PDGUNet
