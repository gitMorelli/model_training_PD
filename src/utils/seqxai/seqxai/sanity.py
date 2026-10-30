"""Checks that the explanations mean something.

- randomization_test   Adebayo et al. 2018 cascading parameter randomization: maps that
                       barely change when weights are destroyed don't explain the model
- temporal_deletion    remove tokens in attribution order; faithful rankings make the
                       target probability collapse faster than random order
- pixel_deletion       same for pixels of one crop (deletion and insertion curves)
- border_mass          fraction of attribution near the image border (shortcut hint)
"""
from __future__ import annotations

import copy
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from .runner import HeadInputs, ModelRunner
from .spatial import gradcam_sequence, get_layer, make_baseline

_trapz = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------- randomization
def _reinit(m: nn.Module, gen: torch.Generator):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
    else:
        for p in m.parameters(recurse=False):
            with torch.no_grad():
                p.normal_(0, 0.02, generator=gen) if p.device.type == 'cpu' else p.normal_(0, 0.02)
    if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.running_mean is not None:
        m.reset_running_stats()


def _param_leaves(module: nn.Module):
    return [(n, m) for n, m in module.named_modules()
            if any(True for _ in m.parameters(recurse=False))]


def randomization_test(runner: ModelRunner, items: Sequence[HeadInputs],
                       frames: Sequence[torch.Tensor], targets: Sequence[int],
                       layer_name: Optional[str] = None, max_crops: int = 8,
                       max_steps: int = 12, seed: int = 0) -> pd.DataFrame:
    """Cascading randomization: first the whole classifier head, then vision layers from
    the top down. After each step, recompute Grad-CAM/HiResCAM on the same crops and
    report the Spearman correlation of |map| with the original. Healthy result: the
    correlation drops quickly. For sanity, also reports the correlation of maps with
    the input image intensity (high = edge-detector-like map)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = copy.deepcopy(runner.model).eval()
    r2 = ModelRunner(model, runner.device, max_tokens_per_call=runner.max_tokens)

    def crops_of(it):
        c = [tuple(x) for x in it.base_mask().nonzero().tolist()]
        return c[:max_crops]

    def maps(rr):
        layer = get_layer(rr.model.vision_model, layer_name)
        out = []
        for it, fr, tg in zip(items, frames, targets):
            # features must be recomputed after randomizing the CNN
            f = torch.stack([rr.vision(fr[t:t + 1].flatten(0, 1).to(rr.device)).detach()
                             for t in range(it.T)])
            it2 = HeadInputs(it.seq_id, f.view(it.T, it.k, -1), it.slot_ids, it.view_mask,
                             it.local_properties, it.global_properties, it.label, it.meta)
            res = gradcam_sequence(rr, it2, fr, tg, layer, crops_of(it))
            out.append(res)
        return out

    ref = maps(r2)
    vis_leaves = _param_leaves(model.vision_model)[::-1]
    stages = [('classifier(all)', [model.classifier])]
    step = max(1, len(vis_leaves) // max_steps)
    acc = []
    for i, (n, m) in enumerate(vis_leaves):
        acc.append(m)
        if (i + 1) % step == 0 or i == len(vis_leaves) - 1:
            stages.append((f'vision<-{n}', list(acc)))
            acc = []
    rows = []
    for name, mods in stages:
        for m in mods:
            for sub in m.modules():
                _reinit(sub, gen)
        cur = maps(r2)
        for kind in ('gradcam', 'hirescam'):
            cs = []
            for a, b in zip(ref, cur):
                for key in a[kind]:
                    x, y = np.abs(a[kind][key]).ravel(), np.abs(b[kind][key]).ravel()
                    if x.std() > 0 and y.std() > 0:
                        cs.append(spearmanr(x, y)[0])
            rows.append(dict(stage=name, method=kind, spearman_mean=np.mean(cs) if cs else np.nan,
                             spearman_std=np.std(cs) if cs else np.nan, n=len(cs)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- deletion
def _auc(y):
    y = np.asarray(y, float)
    return float(_trapz(y, dx=1 / max(1, len(y) - 1))) if len(y) > 1 else float(y[0])


@torch.no_grad()
def temporal_deletion(runner: ModelRunner, item: HeadInputs, target: int,
                      rankings: dict, n_random: int = 10, seed: int = 0) -> pd.DataFrame:
    """rankings: {method: per-token importance (token order, higher = more important)}.
    Deletion: remove the top-r tokens, r = 0..T-1. Insertion: keep only the top-r,
    r = 1..T. Curves are target probabilities; lower deletion AUC and higher insertion
    AUC = more faithful. 'random' averages n_random random orders."""
    T = item.T
    rng = np.random.default_rng(seed)
    orders = {m: np.argsort(-np.nan_to_num(np.asarray(v, float), nan=-np.inf), kind='stable')
              for m, v in rankings.items()}
    for r in range(n_random):
        orders[f'random#{r}'] = rng.permutation(T)
    rows = []
    for m, o in orders.items():
        dele = [item.select_tokens(sorted(o[r:].tolist())) for r in range(T)]
        ins = [item.select_tokens(sorted(o[:r].tolist())) for r in range(1, T + 1)]
        pd_ = runner.probs(runner.logits(dele))[:, target].cpu().numpy()
        pi_ = runner.probs(runner.logits(ins))[:, target].cpu().numpy()
        rows.append(dict(seq_id=item.seq_id, method=m.split('#')[0], T=T,
                         deletion_auc=_auc(pd_), insertion_auc=_auc(pi_),
                         deletion_curve=pd_.round(4).tolist(), insertion_curve=pi_.round(4).tolist()))
    df = pd.DataFrame(rows)
    agg = df.groupby(['seq_id', 'method', 'T'], as_index=False)[['deletion_auc', 'insertion_auc']].mean()
    return agg


@torch.no_grad()
def pixel_deletion(runner: ModelRunner, item: HeadInputs, frames: torch.Tensor, t: int, j: int,
                   target: int, maps: dict, fill='blur', n_points: int = 11,
                   n_random: int = 3, seed: int = 0) -> pd.DataFrame:
    """maps: {method: (H, W) attribution (higher = more important)}. Pixels are replaced
    by the baseline (deletion) or revealed from it (insertion) in attribution order."""
    x = frames[t, j].to(runner.device).float()
    base = make_baseline(x, fill)
    H, W = x.shape[-2:]
    fr = np.linspace(0, 1, n_points)
    rng = np.random.default_rng(seed)
    maps = dict(maps)
    for r in range(n_random):
        maps[f'random#{r}'] = rng.random((H, W))
    rows = []
    for m, a in maps.items():
        order = torch.as_tensor(np.argsort(-np.asarray(a).ravel(), kind='stable').copy(), device=x.device)
        dels, inss = [], []
        for f in fr:
            n = int(round(f * H * W))
            mask = torch.zeros(H * W, device=x.device, dtype=torch.bool)
            mask[order[:n]] = True
            mask = mask.view(1, H, W)
            dels.append(torch.where(mask, base, x))
            inss.append(torch.where(mask, x, base))
        imgs = torch.stack(dels + inss)
        n_ = imgs.shape[0]
        Fall = item.feats.unsqueeze(0).repeat(n_, 1, 1, 1)
        Fall[:, t, j] = runner.vision(imgs)
        P = runner.probs(runner.head_forward(runner.make_batch([item] * n_, feats=list(Fall))))[:, target]
        P = P.cpu().numpy()
        rows.append(dict(seq_id=item.seq_id, token=t, crop=j, method=m.split('#')[0],
                         deletion_auc=_auc(P[:n_points]), insertion_auc=_auc(P[n_points:])))
    return pd.DataFrame(rows).groupby(['seq_id', 'token', 'crop', 'method'], as_index=False).mean()


# --------------------------------------------------------------------------- shortcuts
def border_mass(m: np.ndarray, band: float = 0.1) -> dict:
    """Share of positive attribution within `band` (fraction of size) of the border,
    vs the share of area. Ratio >> 1 on many crops suggests border artefacts."""
    a = np.clip(np.asarray(m, float), 0, None)
    H, W = a.shape
    bh, bw = max(1, int(H * band)), max(1, int(W * band))
    border = np.ones_like(a, dtype=bool)
    border[bh:H - bh, bw:W - bw] = False
    tot = a.sum()
    share = float(a[border].sum() / tot) if tot > 0 else np.nan
    area = float(border.mean())
    return dict(border_share=share, border_area=area, border_ratio=share / area if area else np.nan)
