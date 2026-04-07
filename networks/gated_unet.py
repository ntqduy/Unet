# networks/gated_unet.py
import torch
import torch.nn as nn
from networks.common import DoubleConv2d

class LearnableGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))   # learnable per-channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        return x * gate

    def get_gate_values(self):
        return torch.sigmoid(self.alpha).detach().cpu()


class GatedDoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalization: str = "batchnorm"):
        super().__init__()
        self.conv = DoubleConv2d(in_channels, out_channels, normalization=normalization)
        self.gate = LearnableGate(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.gate(x)
        return x

    def get_gates(self):
        return self.gate.get_gate_values()


class GatedUNet(nn.Module):
    """Student model với Learnable Channel Gating theo blueprint"""
    def __init__(self, 
                 in_channels: int = 3,
                 num_classes: int = 2,
                 channel_config: tuple = (32, 64, 128, 256, 512),   # blueprint từ pruning
                 normalization: str = "batchnorm"):
        super().__init__()
        
        c = channel_config
        self.channel_config = c

        self.stem   = GatedDoubleConv2d(in_channels, c[0], normalization)
        self.down1  = nn.Sequential(nn.MaxPool2d(2), GatedDoubleConv2d(c[0],  c[1], normalization))
        self.down2  = nn.Sequential(nn.MaxPool2d(2), GatedDoubleConv2d(c[1],  c[2], normalization))
        self.down3  = nn.Sequential(nn.MaxPool2d(2), GatedDoubleConv2d(c[2],  c[3], normalization))
        self.down4  = nn.Sequential(nn.MaxPool2d(2), GatedDoubleConv2d(c[3],  c[4], normalization))

        self.up4    = nn.Sequential(nn.ConvTranspose2d(c[4], c[3], 2, 2), 
                                    GatedDoubleConv2d(c[3]*2, c[3], normalization))
        self.up3    = nn.Sequential(nn.ConvTranspose2d(c[3], c[2], 2, 2),
                                    GatedDoubleConv2d(c[2]*2, c[2], normalization))
        self.up2    = nn.Sequential(nn.ConvTranspose2d(c[2], c[1], 2, 2),
                                    GatedDoubleConv2d(c[1]*2, c[1], normalization))
        self.up1    = nn.Sequential(nn.ConvTranspose2d(c[1], c[0], 2, 2),
                                    GatedDoubleConv2d(c[0]*2, c[0], normalization))

        self.head   = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up4(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.up1(x)
        x = torch.cat([x0, x], dim=1)

        return self.head(x)

    def get_all_gates(self):
        """Trả về list các gate values để tính L_sparsity"""
        gates = []
        for m in self.modules():
            if isinstance(m, GatedDoubleConv2d):
                gates.append(m.get_gates())
        return gates