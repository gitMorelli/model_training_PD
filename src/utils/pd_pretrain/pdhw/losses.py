"""Reconstruction loss tuned to keep the fine stroke detail that carries PD signal."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import dilate, ink_mask


def image_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


class ReconLoss(nn.Module):
    """Foreground-weighted L1 + image-gradient L1 + (optional) FFT-magnitude L1.

    * Foreground weighting: pixels within `fg_radius` of ink (in the target) get
      weight `fg_weight`, background gets 1. Without it, a model can score well by
      drawing blank paper with vague strokes.
    * Gradient term: penalises differences in local edges/oscillations, the
      scale at which tremor lives. Plain L1/MSE lets the decoder smooth these out.
    * FFT term: matches the amplitude spectrum, a global check on high frequencies.

    Returns (total, parts) where every term is a per-pixel mean.
    """

    def __init__(
        self,
        fg_weight: float = 5.0,
        fg_radius: int = 1,
        grad_weight: float = 1.0,
        fft_weight: float = 0.0,
    ):
        super().__init__()
        self.fg_weight = fg_weight
        self.fg_radius = fg_radius
        self.grad_weight = grad_weight
        self.fft_weight = fft_weight

    def weight_map(self, target: torch.Tensor) -> torch.Tensor:
        fg = dilate(ink_mask(target), self.fg_radius)
        return 1.0 + (self.fg_weight - 1.0) * fg

    def forward(self, pred: torch.Tensor, target: torch.Tensor, reduce: bool = True):
        w = self.weight_map(target)
        dims = (1, 2, 3)
        l1 = ((pred - target).abs() * w).sum(dims) / w.sum(dims)
        total = l1
        parts = {"l1": l1}

        if self.grad_weight > 0:
            pdx, pdy = image_gradients(pred)
            tdx, tdy = image_gradients(target)
            wdx = torch.maximum(w[..., :, 1:], w[..., :, :-1])
            wdy = torch.maximum(w[..., 1:, :], w[..., :-1, :])
            g = ((pdx - tdx).abs() * wdx).sum(dims) / wdx.sum(dims) + (
                (pdy - tdy).abs() * wdy
            ).sum(dims) / wdy.sum(dims)
            total = total + self.grad_weight * g
            parts["grad"] = g

        if self.fft_weight > 0:
            pf = torch.fft.rfft2(pred.float(), norm="ortho").abs()
            tf = torch.fft.rfft2(target.float(), norm="ortho").abs()
            f = (torch.log1p(pf) - torch.log1p(tf)).abs().mean(dims)
            total = total + self.fft_weight * f
            parts["fft"] = f

        if reduce:
            return total.mean(), {k: v.mean() for k, v in parts.items()}
        return total, parts
