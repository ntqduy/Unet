from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from networks.Basic_Model.common import DoubleConv2d
from utils.model_output import BaseSegmentationModel


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm", dropout: float = 0.0) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv2d(in_channels, out_channels, normalization=normalization, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        normalization: str = "batchnorm",
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv2d(out_channels + skip_channels, out_channels, normalization=normalization)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(BaseSegmentationModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        feature_channels: Sequence[int] = (32, 64, 128, 256, 512),
        normalization: str = "batchnorm",
        dropout: float = 0.0,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        channels = tuple(int(channel) for channel in feature_channels)
        if len(channels) != 5:
            raise ValueError("UNet2D expects exactly 5 encoder stages.")

        self.model_name = "unet"
        self.backbone_name = "unet_encoder"
        self.stem = DoubleConv2d(in_channels, channels[0], normalization=normalization, dropout=dropout)
        self.down1 = DownBlock(channels[0], channels[1], normalization=normalization, dropout=dropout)
        self.down2 = DownBlock(channels[1], channels[2], normalization=normalization, dropout=dropout)
        self.down3 = DownBlock(channels[2], channels[3], normalization=normalization, dropout=dropout)
        self.down4 = DownBlock(channels[3], channels[4], normalization=normalization, dropout=dropout)

        self.up1 = UpBlock(channels[4], channels[3], channels[3], normalization=normalization, bilinear=bilinear)
        self.up2 = UpBlock(channels[3], channels[2], channels[2], normalization=normalization, bilinear=bilinear)
        self.up3 = UpBlock(channels[2], channels[1], channels[1], normalization=normalization, bilinear=bilinear)
        self.up4 = UpBlock(channels[1], channels[0], channels[0], normalization=normalization, bilinear=bilinear)

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        features = self.up4(x, x0)
        logits = self.head(features)
        return logits, features

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits, features = self.forward_features(x)
        output = self.build_output(
            logits,
            features={"decoder": features},
            aux={"feature_channels": list(self.head.weight.shape[1:2])},
        )
        if return_features:
            return output
        return output


UNet = UNet2D
