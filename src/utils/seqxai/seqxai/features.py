"""What the learned features represent.

- top_activating        stream the dataset; keep, for each channel of a CNN layer, the
                        K crops/locations that activate it most (real examples > synthetic)
- probe_concepts        linear probes for human concepts (token- or crop-level), with
                        subject-grouped CV and a shuffled-label control
- probe_metadata        can nuisance variables (site, device, ...) be decoded from the
                        sequence embedding? (shortcut risk)
- tcav                  concept activation vectors in CNN-feature space + directional
                        derivatives of the sequence score, with random-CAV significance
- SparseAutoencoder     TopK SAE on CNN crop features -> monosemantic-ish directions,
                        ranked by their attribution to the prediction
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .runner import HeadInputs, ModelRunner


# --------------------------------------------------------------------------- top activating
@torch.no_grad()
def top_activating(runner: ModelRunner, adapter, ids: Sequence[str], layer: nn.Module,
                   K: int = 16, channels: Optional[Sequence[int]] = None,
                   chunk: int = 64, verbose: bool = True) -> dict:
    """Returns dict(values (C, K), refs (C, K, 5) = [seq_index, token, crop, y, x],
    act_hw (h, w), mean_act (C,), channels). Only real (unmasked) crops are scanned."""
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__('a', o.detach()))
    vals = refs = None
    mean_sum, n_seen = None, 0
    try:
        for si, sid in enumerate(ids):
            s = adapter.load(sid)
            m = s.view_mask.bool() if s.view_mask is not None else torch.ones(s.T, s.k, dtype=torch.bool)
            allowed = runner.modality_crop_mask(runner.modality_names if runner.modality_aware else None, s.k).cpu()
            m = m & allowed[None]
            crops = m.nonzero().tolist()
            for a in range(0, len(crops), chunk):
                cc = crops[a:a + chunk]
                imgs = torch.stack([s.frames[t, j] for t, j in cc]).to(runner.device)
                runner.vision(imgs)
                A = store['a'].float()                       # (n, C, h, w)
                if channels is not None:
                    A = A[:, list(channels)]
                n, C, hh, ww = A.shape
                flat = A.flatten(2)
                v, pos = flat.max(2)                         # (n, C)
                mean_sum = flat.mean(2).sum(0) if mean_sum is None else mean_sum + flat.mean(2).sum(0)
                n_seen += n
                r = torch.stack([torch.full_like(pos, si), torch.tensor([t for t, _ in cc], device=pos.device)[:, None].expand_as(pos),
                                 torch.tensor([j for _, j in cc], device=pos.device)[:, None].expand_as(pos),
                                 pos // ww, pos % ww], -1)   # (n, C, 5)
                v, r = v.T, r.transpose(0, 1)                # (C, n), (C, n, 5)
                if vals is None:
                    vals, refs = v, r
                else:
                    vals, refs = torch.cat([vals, v], 1), torch.cat([refs, r], 1)
                if vals.shape[1] > K:
                    top = vals.topk(K, 1).indices
                    vals = vals.gather(1, top)
                    refs = refs.gather(1, top[..., None].expand(-1, -1, 5))
            if verbose and (si + 1) % 100 == 0:
                print(f'[top_activating] {si + 1}/{len(ids)}')
    finally:
        h.remove()
    order = vals.argsort(1, descending=True)
    vals, refs = vals.gather(1, order), refs.gather(1, order[..., None].expand(-1, -1, 5))
    return dict(values=vals.cpu().numpy(), refs=refs.cpu().numpy(), act_hw=(hh, ww),
                mean_act=(mean_sum / n_seen).cpu().numpy(),
                channels=np.array(channels if channels is not None else np.arange(vals.shape[0])),
                ids=list(ids))


def patch_box(y, x, act_hw, img_hw, patch: Optional[int] = None):
    """Approximate image window for an activation location (centre of the cell,
    side `patch`, default 2 cells or 1/4 of the image, whichever is larger)."""
    H, W = img_hw
    h, w = act_hw
    cy, cx = (y + 0.5) * H / h, (x + 0.5) * W / w
    p = patch or int(max(2 * H / h, H / 4))
    y0 = int(np.clip(cy - p / 2, 0, max(0, H - p)))
    x0 = int(np.clip(cx - p / 2, 0, max(0, W - p)))
    return y0, x0, p


# --------------------------------------------------------------------------- probes
def _probe(X, y, groups, continuous, n_splits=5, seed=0):
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ng = len(np.unique(groups))
    if ng < 2 or (not continuous and len(np.unique(y)) < 2):
        return np.nan
    cv = GroupKFold(n_splits=min(n_splits, ng))
    pred = np.zeros(len(y), float)
    for tr, te in cv.split(X, y, groups):
        if not continuous and len(np.unique(y[tr])) < 2:
            return np.nan
        clf = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)) if continuous
                            else LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te]) if continuous else clf.predict_proba(X[te])[:, 1]
    try:
        return float(r2_score(y, pred) if continuous else roc_auc_score(y, pred))
    except ValueError:
        return np.nan


def probe_concepts(concepts: pd.DataFrame, cache, emb: dict, crop_names: Sequence[str],
                   n_shuffle: int = 5, seed: int = 0) -> pd.DataFrame:
    """concepts: columns seq_id, slot, concept, value [, crop]
       - without 'crop' (or NaN): token-level concept, probed on token embeddings q
       - with 'crop' (index or name): crop-level concept, probed on CNN features
    Metric: AUC (binary) or R^2 (continuous), subject-grouped 5-fold CV. The shuffled
    control should be ~0.5 / ~0."""
    rng = np.random.default_rng(seed)
    ids = [str(i) for i in emb['ids']]
    pos = {(ids[int(a)], int(s)): n for n, (a, s) in enumerate(emb['q_index'])}
    name_to_j = {n: j for j, n in enumerate(crop_names)}
    rows = []
    for concept, g in concepts.groupby('concept'):
        crop_level = 'crop' in g and g['crop'].notna().any()
        X, y, grp = [], [], []
        feats_cache = {}
        for r in g.itertuples():
            sid, slot = str(r.seq_id), int(r.slot)
            if crop_level:
                if sid not in feats_cache:
                    feats_cache[sid] = cache.load(sid)
                it = feats_cache[sid]
                j = name_to_j.get(r.crop, None) if isinstance(r.crop, str) else int(r.crop)
                tok = (it.slot_ids == slot).nonzero()
                if j is None or len(tok) == 0:
                    continue
                X.append(it.feats[int(tok[0]), j].numpy())
            else:
                n = pos.get((sid, slot))
                if n is None:
                    continue
                X.append(emb['q'][n])
            y.append(r.value)
            grp.append(sid)
        if len(X) < 10:
            rows.append(dict(concept=concept, level='crop' if crop_level else 'token', n=len(X)))
            continue
        X, y, grp = np.stack(X), np.asarray(y, float), np.asarray(grp)
        continuous = len(np.unique(y)) > 2
        if not continuous:
            y = (y > 0).astype(int)
        score = _probe(X, y, grp, continuous)
        shuf = [_probe(X, rng.permutation(y), grp, continuous) for _ in range(n_shuffle)]
        rows.append(dict(concept=concept, level='crop' if crop_level else 'token', n=len(X),
                         metric='r2' if continuous else 'auc', score=score,
                         shuffled_mean=float(np.nanmean(shuf)), shuffled_max=float(np.nanmax(shuf))))
    return pd.DataFrame(rows)


def probe_metadata(emb: dict, meta: pd.DataFrame, columns: Optional[Sequence[str]] = None,
                   n_shuffle: int = 5, seed: int = 0) -> pd.DataFrame:
    """Decode sequence-level metadata from the pooled embedding z. A nuisance variable
    (acquisition site, device, operator) that is easy to decode is a shortcut risk."""
    rng = np.random.default_rng(seed)
    ids = [str(i) for i in emb['ids']]
    Z = emb['z']
    meta = meta.reindex(ids)
    rows = []
    for c in columns or meta.columns:
        v = meta[c]
        ok = v.notna().values
        if ok.sum() < 10:
            continue
        y = v.values[ok]
        if v.dtype == object or v.nunique() <= 10:
            classes = pd.Series(y).astype(str)
            top = classes.value_counts().index
            if len(top) < 2:
                continue
            res = []
            for cl in top[:min(len(top), 5)] if len(top) > 2 else top[:1]:   # one-vs-rest
                yy = (classes == cl).astype(int).values
                res.append((_probe(Z[ok], yy, np.arange(ok.sum()), False),
                            np.nanmean([_probe(Z[ok], rng.permutation(yy), np.arange(ok.sum()), False)
                                        for _ in range(n_shuffle)])))
            rows.append(dict(column=c, kind='categorical', metric='auc (mean one-vs-rest)',
                             score=float(np.nanmean([r[0] for r in res])),
                             shuffled_mean=float(np.nanmean([r[1] for r in res])), n=int(ok.sum())))
        else:
            yy = y.astype(float)
            rows.append(dict(column=c, kind='continuous', metric='r2',
                             score=_probe(Z[ok], yy, np.arange(ok.sum()), True),
                             shuffled_mean=float(np.nanmean([_probe(Z[ok], rng.permutation(yy), np.arange(ok.sum()), True)
                                                             for _ in range(n_shuffle)])), n=int(ok.sum())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- TCAV
def _cav(pos, neg, seed, method='lr'):
    """Concept direction in RAW feature space. 'lr': logistic regression on raw features
    (as in the original TCAV); 'mean_diff': difference of class means (most stable).
    Do not standardize first: mapping back divides by per-feature std, and near-dead
    ReLU channels (std ~ 0) then dominate the direction with noise."""
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([pos, neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    clf = LogisticRegression(C=1.0, max_iter=5000, random_state=seed).fit(X, y)
    w = clf.coef_[0] if method == 'lr' else pos.mean(0) - neg.mean(0)
    return w / (np.linalg.norm(w) + 1e-12), clf.score(X, y)


def tcav(runner: ModelRunner, cache, concept_feats: dict, pool_feats: np.ndarray,
         eval_items: Sequence[HeadInputs], target: int, n_random: int = 10,
         crop_filter: Optional[np.ndarray] = None, seed: int = 0,
         method: str = 'mean_diff') -> pd.DataFrame:
    """concept_feats: {concept: (n, F) positives} or {concept: ((n, F) pos, (m, F) neg)}
                   Annotated negatives are strongly preferred: random negatives that
                   contain the concept make logistic-regression CAVs unstable.
    pool_feats:    (M, F) CNN features of random crops (random concepts / fallback negatives)
    eval_items:    sequences whose sensitivity we measure (e.g. all with label==target)
    crop_filter:   optional (k,) bool, only crops at these positions (e.g. one modality)

    TCAV score = fraction of real crops whose directional derivative of the target
    score along the CAV is > 0. It is a LOCAL measure: it depends on which sequences
    you evaluate on (by default those whose label is the target). Concept CAVs are trained against n_random different
    random negative sets; the null distribution uses random-vs-random CAVs.
    p-value: two-sided t-test concept vs null."""
    from scipy.stats import ttest_ind
    rng = np.random.default_rng(seed)
    grads = []
    for it in eval_items:
        g, _ = runner.feature_grad(it, target)
        m = it.base_mask()
        if crop_filter is not None:
            m = m & torch.as_tensor(crop_filter, device=m.device)[None]
        grads.append(g[m].cpu().numpy())
    G = np.concatenate(grads)                        # (n_crops, F)

    def score(w):
        return float((G @ w > 0).mean())

    def sample(n):
        return pool_feats[rng.choice(len(pool_feats), min(n, len(pool_feats)), replace=False)]

    null = []
    for r in range(n_random):
        a, b = sample(64), sample(64)
        null.append(score(_cav(a, b, seed + r, method)[0]))
    rows = []
    for name, val in concept_feats.items():
        pos, negs = val if isinstance(val, tuple) else (val, None)
        sc, accs = [], []
        for r in range(n_random):
            if negs is not None:
                neg = negs[rng.choice(len(negs), min(len(negs), max(len(pos), 32)), replace=False)]
            else:
                neg = sample(max(len(pos), 32))
            w, acc = _cav(pos, neg, seed + r, method)
            sc.append(score(w))
            accs.append(acc)
        p = ttest_ind(sc, null).pvalue if np.std(sc) + np.std(null) > 0 else np.nan
        rows.append(dict(concept=name, target=target, method=method,
                         negatives='annotated' if negs is not None else 'random',
                         tcav_mean=np.mean(sc), tcav_std=np.std(sc),
                         null_mean=np.mean(null), null_std=np.std(null), p_value=p,
                         cav_train_acc=np.mean(accs), n_concept=len(pos), n_crops_eval=len(G)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- SAE
class SparseAutoencoder(nn.Module):
    """TopK SAE (Gao et al. 2024): x -> ReLU top-k latents -> linear decoder.
    Inputs are centred and scaled to unit average norm before training."""

    def __init__(self, d_in: int, n_latents: int, k: int = 16):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d_in, n_latents)
        self.dec = nn.Linear(n_latents, d_in, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.enc.weight)
        with torch.no_grad():
            self.dec.weight.copy_(self.enc.weight.T)
            self._norm_dec()
        self.register_buffer('mu', torch.zeros(d_in))
        self.register_buffer('scale', torch.ones(()))

    def _norm_dec(self):
        self.dec.weight.data /= self.dec.weight.data.norm(dim=0, keepdim=True) + 1e-8

    def prep(self, x):
        return (x - self.mu) / self.scale

    def encode(self, x):
        pre = F.relu(self.enc(self.prep(x) - self.b_pre))
        v, i = pre.topk(self.k, -1)
        return torch.zeros_like(pre).scatter_(-1, i, v)

    def decode(self, z):
        return self.dec(z) + self.b_pre

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def fit(self, X: torch.Tensor, steps: int = 3000, batch: int = 256, lr: float = 1e-3,
            seed: int = 0, verbose: bool = True):
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.mu.copy_(X.mean(0))
            self.scale.copy_((X - self.mu).norm(dim=1).mean().clamp(min=1e-8))
            self.b_pre.copy_(self.prep(X).median(0).values)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for s in range(steps):
            xb = X[torch.randint(len(X), (batch,), generator=g)].to(self.enc.weight.device)
            rec, _ = self(xb)
            loss = F.mse_loss(rec, self.prep(xb))
            opt.zero_grad()
            loss.backward()
            self.dec.weight.grad -= (self.dec.weight.grad * self.dec.weight).sum(0, keepdim=True) * self.dec.weight
            opt.step()
            self._norm_dec()
            if verbose and (s + 1) % max(1, steps // 5) == 0:
                print(f'[sae] step {s + 1}/{steps} mse {loss.item():.4f}')
        return self

    @torch.no_grad()
    def stats(self, X: torch.Tensor) -> dict:
        Z, R = [], []
        for a in range(0, len(X), 4096):
            xb = X[a:a + 4096].to(self.enc.weight.device)
            rec, z = self(xb)
            Z.append(z.cpu())
            R.append(((rec - self.prep(xb)) ** 2).sum(1).cpu())
        Z = torch.cat(Z)
        Xp = self.prep(X.to(self.enc.weight.device)).cpu()
        fvu = torch.cat(R).sum() / ((Xp - Xp.mean(0)) ** 2).sum()
        return dict(Z=Z, explained_variance=float(1 - fvu),
                    dead_frac=float(((Z > 0).sum(0) == 0).float().mean()))

    def decoder_dirs_raw(self) -> torch.Tensor:
        """(n_latents, d_in) decoder directions mapped back to raw feature units."""
        return self.dec.weight.T.detach() * self.scale


def sae_attribution(sae: SparseAutoencoder, Z: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Per-crop contribution of each latent to the score, first order:
    z_l * (d_l . grad_feats). Z: (n, L), G: (n, F) -> (n, L)."""
    D = sae.decoder_dirs_raw().to(G.device)            # (L, F)
    return Z.to(G.device) * (G @ D.T)
