"""Feedforward network for one PVT target (Pb or Bo)."""
import torch
from torch import nn


class PVTNet(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: tuple = (64, 64), dropout: float = 0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)