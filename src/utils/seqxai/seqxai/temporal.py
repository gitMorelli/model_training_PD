"""Temporal analyses: which tokens (time slots) matter, and does order matter at all.

All functions work on cached CNN features: only the head is re-run.

- slot_occlusion      remove (or fully mask) one token at a time -> delta score
- token_ig            Integrated Gradients on the mixer-input token embeddings q
- slot_attention      the mixer's own attention, mapped back to tokens (if it has any)
- order_tests         shuffle / reverse slot positions: is the head bag-of-frames?
- truncation_curves   prediction from the first j tokens (causal prefixes) and last j
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .runner import HeadInputs, ModelRunner, grad_context


# --------------------------------------------------------------------------- occlusion
@torch.no_grad()
def slot_occlusion(runner: ModelRunner, item: HeadInputs, target: int,
                   mode: str = 'remove') -> np.ndarray:
    """delta[t] = score(full) - score(without token t). Positive = token supports target.

    mode='remove': token deleted from the sequence (like a missing visit)
    mode='mask'  : token kept at its slot but all its crops masked (learned 'missing'
                   handling; only meaningful if the model was trained with crop/modality
                   dropout or imputed crops)
    """
    T = item.T
    full = runner.score(runner.logits([item]), target)[0]
    if mode == 'remove':
        if T == 1:
            return np.full(1, np.nan)
        variants = [item.select_tokens([j for j in range(T) if j != t]) for t in range(T)]
        s = runner.score(runner.logits(variants), target)
    elif mode == 'mask':
        masks = []
        for t in range(T):
            m = torch.ones(T, item.k, dtype=torch.bool, device=item.feats.device)
            m[t] = False
            masks.append(m)
        s = runner.score(runner.logits([item] * T, masks=masks), target)
    else:
        raise ValueError(mode)
    return (full - s).cpu().numpy()


# --------------------------------------------------------------------------- IG on tokens
def token_ig(runner: ModelRunner, item: HeadInputs, target: int, baseline=None,
             steps: int = 32) -> dict:
    """Integrated Gradients w.r.t. the token embeddings q (mixer input, after view pooling,
    property encoders and seq_in_norm). Slot position embeddings are added inside the
    mixer, so they stay intact along the path: we attribute token *content*.

    baseline: None/'zero' -> zeros; a (D,) or (T, D) tensor, e.g. the dataset-mean token.
    Returns signed per-token attribution, its L2 norm over dims, and the completeness gap
    (sum of attributions vs score(q) - score(baseline); should be ~0).
    """
    with torch.no_grad():
        b1 = runner.make_batch([item])
        q = runner.encode(b1)                                    # (T, D)
    if baseline is None or (isinstance(baseline, str) and baseline == 'zero'):
        base = torch.zeros_like(q)
    else:
        base = torch.as_tensor(baseline, dtype=q.dtype, device=q.device).expand_as(q)
    alphas = (torch.arange(steps, device=q.device, dtype=q.dtype) + 0.5) / steps   # midpoint rule
    bS = runner.make_batch([item] * steps)
    with torch.enable_grad(), grad_context(runner.device):
        path = (base[None] + alphas[:, None, None] * (q - base)[None]).reshape(-1, q.shape[-1])
        path.requires_grad_(True)
        s = runner.score(runner.mix(path, bS), target)
        g, = torch.autograd.grad(s.sum(), path)
    avg_g = g.view(steps, *q.shape).mean(0)
    attr = (q - base) * avg_g                                     # (T, D)
    with torch.no_grad():
        s_q = runner.score(runner.mix(q, b1), target)[0]
        s_b = runner.score(runner.mix(base, b1), target)[0]
    return dict(signed=attr.sum(-1).cpu().numpy(), norm=attr.norm(dim=-1).cpu().numpy(),
                completeness_gap=float(attr.sum() - (s_q - s_b)), score=float(s_q),
                baseline_score=float(s_b))


# --------------------------------------------------------------------------- attention
def reduce_slot_attention(attn, T: int, cols: torch.Tensor) -> Optional[np.ndarray]:
    """Best-effort reduction of the mixer's attention to one weight per token.
    Handles (B, L), (B, h, L), (B, L, L) and (B, h, L, L) (mean over heads and queries).
    Returns None when the shape is not recognised (e.g. GRU mixers return None)."""
    if attn is None or not torch.is_tensor(attn):
        return None
    a = attn.detach().float()
    if a.dim() >= 2 and a.shape[0] == 1:
        a = a[0]
    while a.dim() > 1:
        a = a.mean(0)
    if a.dim() != 1 or a.numel() <= int(cols.max()):
        return None
    w = a[cols].cpu().numpy()
    return w / (w.sum() + 1e-12)


def slot_attention(runner: ModelRunner, item: HeadInputs) -> Optional[np.ndarray]:
    _, sa = runner.attentions(item)
    return reduce_slot_attention(sa, item.T, runner.token_layout(item))


# --------------------------------------------------------------------------- order tests
def _time_order(item):
    return torch.argsort(item.slot_ids)


@torch.no_grad()
def order_tests(runner: ModelRunner, items: Sequence[HeadInputs], targets: Sequence[int],
                n_perm: int = 10, seed: int = 0) -> pd.DataFrame:
    """For each sequence: score/prob with the original slots, with the slot positions
    randomly permuted among tokens (content stays, timing scrambled), and reversed.
    Small changes => the head effectively pools a bag of frames."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    base_l = runner.logits(items)
    for i, (it, tgt) in enumerate(zip(items, targets)):
        variants, kinds = [], []
        if it.T > 1:
            for _ in range(n_perm):
                perm = torch.randperm(it.T, generator=g).to(it.slot_ids.device)
                variants.append(it.with_slots(it.slot_ids[perm]))
                kinds.append('shuffle')
            o = _time_order(it)
            rev = it.slot_ids.clone()
            rev[o] = it.slot_ids[o].flip(0)
            variants.append(it.with_slots(rev))
            kinds.append('reverse')
        p0 = runner.probs(base_l[i:i + 1])[0, tgt].item()
        s0 = runner.score(base_l[i:i + 1], tgt).item()
        pred0 = int(runner.probs(base_l[i:i + 1]).argmax(1))
        if variants:
            lg = runner.logits(variants)
            ss = runner.score(lg, tgt).cpu().numpy()
            pp = runner.probs(lg)[:, tgt].cpu().numpy()
            pr = runner.probs(lg).argmax(1).cpu().numpy()
        for j, kind in enumerate(kinds):
            rows.append(dict(seq_id=it.seq_id, T=it.T, kind=kind, target=tgt, score0=s0, prob0=p0,
                             score=ss[j], prob=pp[j], d_prob=pp[j] - p0, flipped=pr[j] != pred0))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- truncation
@torch.no_grad()
def truncation_curves(runner: ModelRunner, items: Sequence[HeadInputs],
                      targets: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Rows (seq_id, direction, n_seen, frac_seen, prob_<c>...) for:
       'first': causal prefixes (model.forward_prefixes; exact for any mixer)
       'last' : suffixes of the last n tokens."""
    rows = []
    prefixes = runner.prefix_logits(items)
    for it, pl in zip(items, prefixes):
        o = _time_order(it).cpu()
        P = runner.probs(pl).cpu().numpy()                       # in token order
        for rank, t in enumerate(o.tolist()):
            rows.append(dict(seq_id=it.seq_id, label=it.label, direction='first', n_seen=rank + 1,
                             frac_seen=(rank + 1) / it.T, T=it.T,
                             **{f'prob_{c}': P[t, c] for c in range(P.shape[1])}))
        variants = [it.select_tokens(o[-n:].tolist()) for n in range(1, it.T + 1)]
        P2 = runner.probs(runner.logits(variants)).cpu().numpy()
        for n in range(1, it.T + 1):
            rows.append(dict(seq_id=it.seq_id, label=it.label, direction='last', n_seen=n,
                             frac_seen=n / it.T, T=it.T,
                             **{f'prob_{c}': P2[n - 1, c] for c in range(P2.shape[1])}))
    return pd.DataFrame(rows)


def truncation_metrics(df: pd.DataFrame, n_classes: int, bins: int = 5) -> pd.DataFrame:
    """Accuracy / AUC as a function of the fraction of the sequence seen."""
    from sklearn.metrics import roc_auc_score
    d = df.copy()
    d['frac_bin'] = np.minimum((d['frac_seen'] * bins - 1e-9).astype(int), bins - 1)
    pcols = [f'prob_{c}' for c in range(n_classes)]
    out = []
    for (direction, fb), g in d.groupby(['direction', 'frac_bin']):
        # one row per sequence: the largest n_seen within the bin
        g = g.sort_values('n_seen').groupby('seq_id').tail(1)
        y = g['label'].values.astype(int)
        P = g[pcols].values
        r = dict(direction=direction, frac_upto=(fb + 1) / bins, n=len(g),
                 accuracy=float((P.argmax(1) == y).mean()))
        try:
            r['auc'] = float(roc_auc_score(y, P[:, 1]) if n_classes == 2
                             else roc_auc_score(y, P, multi_class='ovr'))
        except ValueError:
            r['auc'] = np.nan
        out.append(r)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- per sequence
def explain_tokens(runner: ModelRunner, item: HeadInputs, target: int, ig_baseline=None,
                   ig_steps: int = 32, occlusion_modes=('remove', 'mask')) -> pd.DataFrame:
    """All per-token importances for one sequence, in time order."""
    o = _time_order(item).cpu().numpy()
    df = pd.DataFrame(dict(seq_id=item.seq_id, token=o,
                           slot=item.slot_ids.cpu().numpy()[o], target=target))
    for mode in occlusion_modes:
        df[f'occl_{mode}'] = slot_occlusion(runner, item, target, mode)[o]
    ig = token_ig(runner, item, target, ig_baseline, ig_steps)
    df['ig_signed'], df['ig_norm'] = ig['signed'][o], ig['norm'][o]
    df['ig_completeness_gap'] = ig['completeness_gap']
    att = slot_attention(runner, item)
    df['attention'] = att[o] if att is not None else np.nan
    return df
