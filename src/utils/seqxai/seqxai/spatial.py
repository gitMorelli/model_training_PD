"""Pixel-level attributions, always w.r.t. the SEQUENCE prediction.

Trick that keeps this cheap and memory-safe: in eval mode the CNN maps each crop
independently, so by the chain rule

    d score / d pixels(crop) = (d score / d feats(crop)) . (d feats(crop) / d pixels(crop))

`runner.feature_grad` gives the first factor for all crops in one head backward pass;
the CNN is then re-run (with grad) on chunks of crops only. Results are exactly the
gradients of the full model, without holding the whole sequence's CNN graph in memory.

- gradcam_sequence   Grad-CAM and HiResCAM maps for every (or chosen) crop of a sequence
- pixel_ig           Integrated Gradients on pixels of one crop (others held fixed)
- patch_occlusion    sliding-patch occlusion on one crop
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .runner import HeadInputs, ModelRunner, grad_context


# --------------------------------------------------------------------------- layers
def get_layer(module: nn.Module, name: Optional[str] = None) -> nn.Module:
    """Dotted name inside vision_model (e.g. 'layer4' or 'features.7'); default: the
    last Conv2d."""
    if name:
        m = module
        for p in name.split('.'):
            m = m[int(p)] if p.isdigit() else getattr(m, p)
        return m
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError('no Conv2d in vision_model; pass target_layer explicitly')
    return convs[-1]


class _ActGrad:
    """Capture a layer's output and its gradient. The hook hands a clone downstream so
    in-place ops after the layer (ReLU(inplace=True)) cannot corrupt the stored tensor."""

    def __init__(self, layer):
        self.act, self.grad = None, None
        self.h = layer.register_forward_hook(self._hook)

    def _hook(self, m, i, out):
        self.act = out
        if out.requires_grad:
            out.register_hook(self._save)
        return out.clone()

    def _save(self, g):
        self.grad = g

    def remove(self):
        self.h.remove()


# --------------------------------------------------------------------------- baselines
def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    C = x.shape[-3]
    x4 = x.reshape(-1, C, *x.shape[-2:])
    x4 = F.pad(x4, (r, r, r, r), mode='reflect' if min(x.shape[-2:]) > r else 'replicate')
    x4 = F.conv2d(x4, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x4 = F.conv2d(x4, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x4.reshape(x.shape)


def make_baseline(x: torch.Tensor, kind: Union[str, float, Callable, torch.Tensor] = 'blur') -> torch.Tensor:
    """x: (C, H, W) model-input tensor (already normalized).
    'blur'      heavy gaussian blur (sigma = min(H, W) / 10): removes shapes, keeps colour
    'zeros'     zeros in normalized space (= the normalization mean colour)
    'mean'      per-channel mean of this image
    float       constant value in normalized space (e.g. the normalized value of blank paper)
    tensor      a fixed (C, H, W) baseline
    callable    x -> baseline
    """
    if callable(kind) and not torch.is_tensor(kind):
        return kind(x)
    if torch.is_tensor(kind):
        return kind.to(x).expand_as(x)
    if isinstance(kind, (int, float)):
        return torch.full_like(x, float(kind))
    if kind == 'blur':
        return gaussian_blur(x, min(x.shape[-2:]) / 10)
    if kind == 'zeros':
        return torch.zeros_like(x)
    if kind == 'mean':
        return x.mean((-2, -1), keepdim=True).expand_as(x).clone()
    raise ValueError(kind)


# --------------------------------------------------------------------------- Grad-CAM
def gradcam_sequence(runner: ModelRunner, item: HeadInputs, frames: torch.Tensor, target: int,
                     layer: nn.Module, crops: Optional[Sequence[tuple]] = None,
                     chunk: int = 32) -> dict:
    """Grad-CAM (channel-mean gradient weights) and HiResCAM (element-wise grad*act,
    more faithful for nets ending in global pooling) for crops of one sequence.

    frames: (T, k, C, H, W) of that sequence (from the adapter).
    Returns maps at the layer's resolution (unnormalized, so scales are comparable
    across crops of the sequence), plus per-channel contributions for each crop.
    """
    g_feats, score = runner.feature_grad(item, target)
    if crops is None:
        crops = [tuple(x) for x in item.base_mask().nonzero().tolist()]
    cap = _ActGrad(layer)
    gc, hr, ch = {}, {}, {}
    try:
        for a in range(0, len(crops), chunk):
            cc = crops[a:a + chunk]
            imgs = torch.stack([frames[t, j] for t, j in cc]).to(runner.device)
            gf = torch.stack([g_feats[t, j] for t, j in cc])
            with torch.enable_grad(), grad_context(runner.device):
                f = runner.vision(imgs)
                torch.autograd.backward(f, gf.to(f.dtype))
            A, G = cap.act.detach(), cap.grad.detach()
            w = G.mean((2, 3), keepdim=True)
            cam = F.relu((w * A).sum(1))
            hires = F.relu((G * A).sum(1))
            contrib = (G * A).sum((2, 3))                  # per-channel contribution
            for n, (t, j) in enumerate(cc):
                gc[(t, j)] = cam[n].cpu().numpy()
                hr[(t, j)] = hires[n].cpu().numpy()
                ch[(t, j)] = contrib[n].cpu().numpy()
            cap.act = cap.grad = None
    finally:
        cap.remove()
    return dict(gradcam=gc, hirescam=hr, channel_contrib=ch, score=float(score[0]))


def upsample(m: np.ndarray, size) -> np.ndarray:
    t = torch.as_tensor(m, dtype=torch.float32)[None, None]
    return F.interpolate(t, size=size, mode='bilinear', align_corners=False)[0, 0].numpy()


# --------------------------------------------------------------------------- pixel IG
def _score_with_crop_feats(runner, item, t, j, f_new, target):
    """Head score for n copies of the sequence where crop (t, j) has features f_new (n, F)."""
    n = f_new.shape[0]
    Fall = item.feats.detach().unsqueeze(0).repeat(n, 1, 1, 1)
    Fall[:, t, j] = f_new
    batch = runner.make_batch([item] * n, feats=list(Fall))
    return runner.score(runner.head_forward(batch), target)


def pixel_ig(runner: ModelRunner, item: HeadInputs, frames: torch.Tensor, t: int, j: int,
             target: int, baseline='blur', steps: int = 32, batch: int = 16,
             noise_samples: int = 0, noise_std: float = 0.1) -> dict:
    """Integrated Gradients on the pixels of crop (t, j); all other crops fixed.
    noise_samples > 0 averages IG over noisy copies of the input (SmoothGrad-style
    'NoiseTunnel'), which reduces speckle.
    Returns attr (C, H, W), map (H, W) = sum over channels, completeness gap."""
    x = frames[t, j].to(runner.device).float()
    base = make_baseline(x, baseline)
    alphas = (torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5) / steps
    inputs = [x] if noise_samples == 0 else \
        [x + noise_std * x.std() * torch.randn_like(x) for _ in range(noise_samples)]
    total = torch.zeros_like(x)
    for xi in inputs:
        gsum = torch.zeros_like(x)
        for a in range(0, steps, batch):
            al = alphas[a:a + batch]
            with torch.enable_grad(), grad_context(runner.device):
                xa = (base[None] + al[:, None, None, None] * (xi - base)[None]).requires_grad_(True)
                s = _score_with_crop_feats(runner, item, t, j, runner.vision(xa), target)
                g, = torch.autograd.grad(s.sum(), xa)
            gsum += g.sum(0)
        total += (xi - base) * gsum / steps
    attr = total / len(inputs)
    with torch.no_grad():
        s1 = _score_with_crop_feats(runner, item, t, j, runner.vision(x[None]), target)[0]
        s0 = _score_with_crop_feats(runner, item, t, j, runner.vision(base[None]), target)[0]
    return dict(attr=attr.cpu().numpy(), map=attr.sum(0).cpu().numpy(),
                completeness_gap=float(attr.sum() - (s1 - s0)), score=float(s1),
                baseline_score=float(s0))


@torch.no_grad()
def patch_occlusion(runner: ModelRunner, item: HeadInputs, frames: torch.Tensor, t: int, j: int,
                    target: int, patch: Optional[int] = None, stride: Optional[int] = None,
                    fill='blur', batch: int = 32) -> np.ndarray:
    """(H, W) map: average score drop when a patch covering the pixel is replaced by
    the baseline. Positive = region supports the target."""
    x = frames[t, j].to(runner.device).float()
    C, H, W = x.shape
    patch = patch or max(4, min(H, W) // 8)
    stride = stride or max(1, patch // 2)
    base = make_baseline(x, fill)
    full = _score_with_crop_feats(runner, item, t, j, runner.vision(x[None]), target)[0]
    ys = list(range(0, max(1, H - patch) + 1, stride))
    xs = list(range(0, max(1, W - patch) + 1, stride))
    if ys[-1] + patch < H:
        ys.append(H - patch)
    if xs[-1] + patch < W:
        xs.append(W - patch)
    boxes = [(y, xx) for y in ys for xx in xs]
    heat = torch.zeros(H, W, device=x.device)
    cnt = torch.zeros(H, W, device=x.device)
    for a in range(0, len(boxes), batch):
        bb = boxes[a:a + batch]
        imgs = x[None].repeat(len(bb), 1, 1, 1)
        for n, (y, xx) in enumerate(bb):
            imgs[n, :, y:y + patch, xx:xx + patch] = base[:, y:y + patch, xx:xx + patch]
        s = _score_with_crop_feats(runner, item, t, j, runner.vision(imgs), target)
        for n, (y, xx) in enumerate(bb):
            heat[y:y + patch, xx:xx + patch] += full - s[n]
            cnt[y:y + patch, xx:xx + patch] += 1
    return (heat / cnt.clamp(min=1)).cpu().numpy()
