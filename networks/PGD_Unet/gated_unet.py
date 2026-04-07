from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from networks.Basic_Model.common import DoubleConv2d
from utils.model_output import BaseSegmentationModel


class LearnableGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        return x * gate

    def values(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha)

    def set_trainable(self, trainable: bool) -> None:
        self.alpha.requires_grad_(trainable)

    def force_probability(self, value: float) -> None:
        clamped = min(max(float(value), 1e-4), 1.0 - 1e-4)
        logit = torch.logit(torch.tensor(clamped, dtype=self.alpha.dtype, device=self.alpha.device))
        with torch.no_grad():
            self.alpha.fill_(float(logit.item()))


class GatedDoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.conv = DoubleConv2d(in_channels, out_channels, normalization=normalization)
        self.gate = LearnableGate(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.conv(x))

    def gate_values(self) -> torch.Tensor:
        return self.gate.values()


class GatedDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = GatedDoubleConv2d(in_channels, out_channels, normalization=normalization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class GatedUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = GatedDoubleConv2d(out_channels + skip_channels, out_channels, normalization=normalization)

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
        self.student_name = "gated_student"
        self.set_architecture_config(
            in_channels=in_channels,
            num_classes=num_classes,
            channel_config=list(channels),
            normalization=normalization,
            student_variant="gated_unet",
        )

        self.channel_config = channels
        self.stem = GatedDoubleConv2d(in_channels, channels[0], normalization=normalization)
        self.down1 = GatedDownBlock(channels[0], channels[1], normalization=normalization)
        self.down2 = GatedDownBlock(channels[1], channels[2], normalization=normalization)
        self.down3 = GatedDownBlock(channels[2], channels[3], normalization=normalization)
        self.down4 = GatedDownBlock(channels[3], channels[4], normalization=normalization)

        self.up1 = GatedUpBlock(channels[4], channels[3], channels[3], normalization=normalization)
        self.up2 = GatedUpBlock(channels[3], channels[2], channels[2], normalization=normalization)
        self.up3 = GatedUpBlock(channels[2], channels[1], channels[1], normalization=normalization)
        self.up4 = GatedUpBlock(channels[1], channels[0], channels[0], normalization=normalization)

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward_features(self, x: torch.Tensor):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        decoder_features = self.up4(x, x0)
        logits = self.head(decoder_features)
        features = {
            "encoder": {"stem": x0, "down1": x1, "down2": x2, "down3": x3, "down4": x4},
            "decoder": {"final": decoder_features},
        }
        return logits, features

    def forward(self, x: torch.Tensor):
        logits, features = self.forward_features(x)
        return self.build_output(
            logits,
            features=features,
            aux={
                "channel_config": list(self.channel_config),
                "gate_statistics": self.get_gate_statistics(),
            },
        )

    def get_gate_tensors(self) -> List[torch.Tensor]:
        gate_values: List[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, GatedDoubleConv2d):
                gate_values.append(module.gate_values())
        return gate_values

    def get_gate_modules(self) -> List[LearnableGate]:
        gate_modules: List[LearnableGate] = []
        for module in self.modules():
            if isinstance(module, GatedDoubleConv2d):
                gate_modules.append(module.gate)
        return gate_modules

    def set_gate_trainable(self, trainable: bool) -> None:
        for gate in self.get_gate_modules():
            gate.set_trainable(trainable)

    def force_gates_open(self, open_probability: float = 0.999) -> None:
        for gate in self.get_gate_modules():
            gate.force_probability(open_probability)

    def get_gate_statistics(self) -> Dict[str, List[float]]:
        return {
            "gate_means": [float(gate.mean().detach().cpu()) for gate in self.get_gate_tensors()],
            "gate_sparsity_proxy": [float((1.0 - gate).mean().detach().cpu()) for gate in self.get_gate_tensors()],
        }


GatedUNet = PDGUNet
