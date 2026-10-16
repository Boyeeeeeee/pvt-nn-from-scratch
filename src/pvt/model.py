"""Feedforward network. out_dim=1 for a single target (Pb or Bo); out_dim=2 for the joint ablation."""
import torch
from torch import nn


class PVTNet(nn.Module):
    def __init__(self, in_dim: int = 4, hidden: tuple = (64, 64), dropout: float = 0.0, out_dim: int = 1):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
