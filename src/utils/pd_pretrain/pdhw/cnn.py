"""Shared CNN encoder / conditional decoder used by the DAE and the CVAE."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .blocks import Downsample, FiLMResBlock, Upsample, group_norm


class ConvEncoder(nn.Module):
    """Image (B,1,H,W) -> feature map (B, out_ch, H/2^n, W/2^n), n = len(channels)-1.

    The stem and the first stage run at full resolution (no early stride), so
    one-pixel-wide strokes and small oscillations are seen before any pooling.
    """

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128),
        blocks_per_stage: int = 2,
        out_channels: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.stem = nn.Conv2d(1, channels[0], 3, padding=1)
        layers: list[nn.Module] = []
        ch = channels[0]
        for i, out in enumerate(channels):
            for _ in range(blocks_per_stage):
                layers.append(FiLMResBlock(ch, out, dropout=dropout))
                ch = out
            if i < len(channels) - 1:
                layers.append(Downsample(ch))
        self.body = nn.ModuleList(layers)
        self.out_norm = group_norm(ch)
        self.head = nn.Conv2d(ch, out_channels or ch, 1)
        self.out_channels = out_channels or ch
        self.downsample_factor = 2 ** (len(channels) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for layer in self.body:
            h = layer(h)
        return self.head(torch.nn.functional.silu(self.out_norm(h)))


class ConvDecoder(nn.Module):
    """Feature map (B, in_ch, h, w) -> image (B,1,h*2^n,w*2^n), FiLM-conditioned on `cond`.

    `channels` is given from high to low resolution (same order as the encoder)
    and is traversed in reverse. No skip connections: everything must go
    through the bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int] = (32, 64, 128),
        blocks_per_stage: int = 2,
        cond_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        rev = list(reversed(channels))
        self.inp = nn.Conv2d(in_channels, rev[0], 3, padding=1)
        layers: list[nn.Module] = []
        ch = rev[0]
        for i, out in enumerate(rev):
            for _ in range(blocks_per_stage):
                layers.append(FiLMResBlock(ch, out, cond_dim=cond_dim, dropout=dropout))
                ch = out
            if i < len(rev) - 1:
                layers.append(Upsample(ch))
        self.body = nn.ModuleList(layers)
        self.out_norm = group_norm(ch)
        self.out = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.inp(z)
        for layer in self.body:
            h = layer(h, cond) if isinstance(layer, FiLMResBlock) else layer(h)
        # tanh keeps outputs in [-1, 1], matching the input range
        return torch.tanh(self.out(torch.nn.functional.silu(self.out_norm(h))))
