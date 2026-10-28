"""Shared building blocks: FiLM/AdaGN residual blocks, up/down sampling, ink-mask helpers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with the largest group count <= max_groups that divides `channels`.

    GroupNorm (not BatchNorm) everywhere: it behaves identically in train/eval,
    which matters for EMA weights and for deterministic feature extraction.
    """
    groups = max_groups
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class FiLMResBlock(nn.Module):
    """Residual conv block whose normalisation is modulated by a conditioning vector.

    h = norm(h) * (1 + scale(cond)) + shift(cond)   (AdaGN / FiLM)
    If cond_dim is 0 the block is an ordinary pre-activation residual block.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.film = nn.Linear(cond_dim, 2 * out_ch) if cond_dim > 0 else None
        if self.film is not None:  # start as identity modulation
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        if self.film is not None:
            if cond is None:
                raise ValueError("FiLMResBlock built with cond_dim > 0 needs a cond vector")
            scale, shift = self.film(cond)[:, :, None, None].chunk(2, dim=1)
            h = h * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest upsample + conv (avoids transposed-conv checkerboard artefacts,
    which would look like spurious high-frequency 'tremor')."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


def ink_mask(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Binary ink mask from an image in [-1, 1] with ink = +1. Shape preserved."""
    return (x > threshold).to(x.dtype)


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def masked_spatial_mean(feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average a (B, C, h, w) feature map over pixels where the (B, 1, H, W) mask is on.

    The mask is max-pooled to the feature resolution, so a feature cell counts as
    'ink' if any ink pixel falls inside it. Falls back to the plain mean if a sample
    has no ink at all.
    """
    m = F.adaptive_max_pool2d(mask, feat.shape[-2:])
    denom = m.sum(dim=(2, 3))
    num = (feat * m).sum(dim=(2, 3))
    plain = feat.mean(dim=(2, 3))
    return torch.where(denom > 0, num / denom.clamp_min(1.0), plain)
