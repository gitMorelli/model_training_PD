"""Modality / crop level analyses (head only, on cached features).

Masking uses the model's own mechanism (view_mask), i.e. the same path as
ModularModel.forward(..., modalities=subset): masked crops are dropped by the view
aggregator (and their CNN features zeroed when skip_masked_views=True).

- modality_shapley     exact Shapley values of each modality (all 2^M subsets)
- subset_metrics       dataset-level AUC/accuracy for every modality subset
- crop_occlusion       mask one crop at a time -> delta score, (T, k)
- crop_ig              Integrated Gradients on the CNN features of each crop, (T, k)
- view_attention       the view aggregator's attention, if reducible to (T, k)
- property_ablation    replace local/global properties with dataset means
"""
from __future__ import annotations

import itertools
import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .runner import HeadInputs, ModelRunner, grad_context


def _subset_mask(runner, item, subset):
    m = runner.modality_crop_mask(subset, item.k)
    return m[None].expand(item.T, item.k).clone()


def all_subsets(names):
    return [tuple(c) for r in range(len(names) + 1) for c in itertools.combinations(names, r)]


@torch.no_grad()
def subset_scores(runner: ModelRunner, item: HeadInputs, target: int, subsets) -> dict:
    masks = [_subset_mask(runner, item, s) for s in subsets]
    lg = runner.logits([item] * len(subsets), masks=masks)
    s = runner.score(lg, target).cpu().numpy()
    p = runner.probs(lg)[:, target].cpu().numpy()
    present = [bool((m & item.base_mask()).any()) for m in masks]
    return {sub: (s[i], p[i], present[i]) for i, sub in enumerate(subsets)}


def modality_shapley(runner: ModelRunner, item: HeadInputs, target: int) -> dict:
    """Exact Shapley value of each modality for the target score. The empty coalition is
    the properties-only prediction (all crops masked). Values sum to
    score(all) - score(none)."""
    names = runner.modality_names
    M = len(names)
    if M > 10:
        raise ValueError('more than 10 modalities: use a sampling approximation')
    subs = all_subsets(names)
    sc = subset_scores(runner, item, target, subs)
    v = {frozenset(s): sc[s][0] for s in subs}
    phi = {}
    for m in names:
        others = [n for n in names if n != m]
        tot = 0.0
        for r in range(M):
            w = math.factorial(r) * math.factorial(M - r - 1) / math.factorial(M)
            for c in itertools.combinations(others, r):
                tot += w * (v[frozenset(c) | {m}] - v[frozenset(c)])
        phi[m] = tot
    return dict(shapley=phi, v_all=v[frozenset(names)], v_none=v[frozenset()],
                subsets={'+'.join(s) or 'none': sc[s] for s in subs})


def subset_metrics(runner: ModelRunner, items: Sequence[HeadInputs], subsets=None) -> pd.DataFrame:
    """Per-sequence probabilities under each subset (default: all, each single modality,
    each leave-one-out, none) + whether the subject has any real crop in the subset."""
    names = runner.modality_names
    if subsets is None:
        subsets = [tuple(names)] + [(m,) for m in names]
        if len(names) > 2:
            subsets += [tuple(n for n in names if n != m) for m in names]
        subsets += [()]
        subsets = list(dict.fromkeys(subsets))
    rows = []
    for it in items:
        masks = [_subset_mask(runner, it, s) for s in subsets]
        with torch.no_grad():
            P = runner.probs(runner.logits([it] * len(subsets), masks=masks)).cpu().numpy()
        for s, m, p in zip(subsets, masks, P):
            rows.append(dict(seq_id=it.seq_id, label=it.label, subset='+'.join(s) or 'none',
                             n_mod=len(s), present=bool((m & it.base_mask()).any()),
                             **{f'prob_{c}': p[c] for c in range(len(p))}))
    return pd.DataFrame(rows)


def summarize_subsets(df: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    pc = [f'prob_{c}' for c in range(n_classes)]
    out = []
    for (sub, nm), g in df.groupby(['subset', 'n_mod'], sort=False):
        for only_present in (False, True):
            h = g[g.present] if only_present else g
            if len(h) == 0:
                continue
            y, P = h['label'].values.astype(int), h[pc].values
            r = dict(subset=sub, n_mod=nm, subjects='with_crops' if only_present else 'all',
                     n=len(h), accuracy=float((P.argmax(1) == y).mean()))
            try:
                r['auc'] = float(roc_auc_score(y, P[:, 1]) if n_classes == 2
                                 else roc_auc_score(y, P, multi_class='ovr'))
            except ValueError:
                r['auc'] = np.nan
            out.append(r)
    return pd.DataFrame(out)


@torch.no_grad()
def crop_occlusion(runner: ModelRunner, item: HeadInputs, target: int) -> np.ndarray:
    """(T, k) delta score when masking each single crop; NaN where the crop is absent."""
    base = item.base_mask()
    full = runner.score(runner.logits([item]), target)[0].item()
    idx = base.nonzero().tolist()
    out = np.full((item.T, item.k), np.nan)
    if not idx:
        return out
    masks = []
    for t, j in idx:
        m = torch.ones_like(base)
        m[t, j] = False
        masks.append(m)
    s = runner.score(runner.logits([item] * len(masks), masks=masks), target).cpu().numpy()
    for (t, j), v in zip(idx, s):
        out[t, j] = full - v
    return out


def crop_ig(runner: ModelRunner, item: HeadInputs, target: int, baseline=None,
            steps: int = 32) -> tuple[np.ndarray, float]:
    """IG on the CNN features of all crops jointly, summed per crop -> (T, k).
    baseline: None -> zeros; or (F,) / (k, F) mean features (recommended: the
    dataset-mean feature of each crop position). Returns (attr, completeness_gap)."""
    f = item.feats
    base = torch.zeros_like(f) if baseline is None else \
        torch.as_tensor(baseline, dtype=f.dtype, device=f.device).expand_as(f)
    m = item.base_mask()
    base = torch.where(m[..., None], base, f)                  # absent crops: no path
    alphas = (torch.arange(steps, device=f.device, dtype=f.dtype) + 0.5) / steps
    with torch.enable_grad(), grad_context(runner.device):
        path = base[None] + alphas[:, None, None, None] * (f - base)[None]
        path.requires_grad_(True)
        batch = runner.make_batch([item] * steps, feats=list(path))
        s = runner.score(runner.head_forward(batch), target)
        g, = torch.autograd.grad(s.sum(), path)
    attr = ((f - base) * g.mean(0)).sum(-1)
    with torch.no_grad():
        s1 = runner.score(runner.head_forward(runner.make_batch([item])), target)[0]
        s0 = runner.score(runner.head_forward(runner.make_batch([item], feats=[base])), target)[0]
    a = attr.cpu().numpy()
    a[~m.cpu().numpy()] = np.nan
    return a, float(attr.sum() - (s1 - s0))


def view_attention(runner: ModelRunner, item: HeadInputs) -> Optional[np.ndarray]:
    """Best effort: reduce the aggregator's attention to (T, k). None if not possible
    (e.g. hierarchical aggregators returning nested structures)."""
    va, _ = runner.attentions(item)
    if not torch.is_tensor(va):
        return None
    a = va.detach().float()
    T, k = item.T, item.k
    while a.dim() > 2 and a.shape[-1] == k and a.shape[0] == T:
        a = a.mean(1)
    if tuple(a.shape) != (T, k):
        return None
    return a.cpu().numpy()


@torch.no_grad()
def property_ablation(runner: ModelRunner, item: HeadInputs, target: int,
                      local_mean=None, global_mean=None, per_feature: bool = True) -> dict:
    """Delta score when replacing properties by their dataset mean (all at once and,
    optionally, one global feature at a time). Big deltas = the prediction leans on
    tabular properties rather than on the images."""
    full = runner.score(runner.logits([item]), target)[0].item()
    out = {}

    def rep(local=None, glob=None):
        return HeadInputs(item.seq_id, item.feats, item.slot_ids, item.view_mask,
                          item.local_properties if local is None else local,
                          item.global_properties if glob is None else glob, item.label, item.meta)

    variants, names = [], []
    if item.local_properties is not None and local_mean is not None:
        variants.append(rep(local=torch.as_tensor(local_mean, device=item.feats.device,
                                                  dtype=item.local_properties.dtype).expand_as(item.local_properties)))
        names.append('local_all')
    if item.global_properties is not None and global_mean is not None:
        gm = torch.as_tensor(global_mean, device=item.feats.device, dtype=item.global_properties.dtype)
        variants.append(rep(glob=gm.clone()))
        names.append('global_all')
        if per_feature:
            for c in range(gm.numel()):
                g = item.global_properties.clone()
                g[c] = gm[c]
                variants.append(rep(glob=g))
                names.append(f'global_{c}')
    if variants:
        s = runner.score(runner.logits(variants), target).cpu().numpy()
        out = {n: full - v for n, v in zip(names, s)}
    return out
