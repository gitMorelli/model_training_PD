"""Embedding geometry over time: where do sequences go, and how do they move?"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def project(X: np.ndarray, method: str = 'pca', seed: int = 0, fit_on: Optional[np.ndarray] = None):
    """2-D projection. 'umap' needs umap-learn; falls back to PCA if missing."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    if method == 'umap':
        try:
            import umap
            return umap.UMAP(random_state=seed).fit_transform(Xs), 'umap'
        except ImportError:
            method = 'pca'
    from sklearn.decomposition import PCA
    p = PCA(2, random_state=seed).fit(Xs if fit_on is None else fit_on)
    return p.transform(Xs), f'pca ({p.explained_variance_ratio_.sum():.0%} var)'


def token_table(emb: dict, preds: Optional[pd.DataFrame] = None, method: str = 'pca') -> pd.DataFrame:
    """One row per token: seq_id, slot, 2-D coordinates, label/pred/correct."""
    ids = np.array([str(i) for i in emb['ids']])
    Y, name = project(emb['q'], method)
    df = pd.DataFrame(dict(seq_id=ids[emb['q_index'][:, 0].astype(int)],
                           slot=emb['q_index'][:, 1].astype(int), x=Y[:, 0], y=Y[:, 1]))
    df.attrs['projection'] = name
    if preds is not None:
        df = df.join(preds[['label', 'pred', 'correct']], on='seq_id')
    return df


def trajectory_stats(tok: pd.DataFrame) -> pd.DataFrame:
    """Per sequence: start/end position, path length, net displacement. Comparing these
    between classes tells you whether classes differ by *state* (where) or *dynamics*
    (how much / in which direction they move)."""
    rows = []
    for sid, g in tok.sort_values('slot').groupby('seq_id'):
        P = g[['x', 'y']].values
        step = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P) > 1 else np.zeros(1)
        rows.append(dict(seq_id=sid, label=g['label'].iloc[0] if 'label' in g else np.nan,
                         T=len(g), cx=P[:, 0].mean(), cy=P[:, 1].mean(),
                         path_len=step.sum(), mean_step=step.mean(),
                         net_disp=np.linalg.norm(P[-1] - P[0]),
                         dx=P[-1, 0] - P[0, 0], dy=P[-1, 1] - P[0, 1]))
    return pd.DataFrame(rows)


def class_separation(stats: pd.DataFrame, cols=('cx', 'cy', 'mean_step', 'net_disp', 'dx', 'dy')) -> pd.DataFrame:
    """Effect size (Cohen's d, or eta^2 for >2 classes) of each trajectory statistic."""
    rows = []
    labels = sorted(stats['label'].dropna().unique())
    for c in cols:
        if len(labels) == 2:
            a = stats.loc[stats.label == labels[0], c]
            b = stats.loc[stats.label == labels[1], c]
            sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / max(1, len(a) + len(b) - 2))
            rows.append(dict(stat=c, cohen_d=(b.mean() - a.mean()) / (sp + 1e-12)))
        else:
            grand = stats[c].mean()
            ssb = sum(len(g) * (g[c].mean() - grand) ** 2 for _, g in stats.groupby('label'))
            sst = ((stats[c] - grand) ** 2).sum()
            rows.append(dict(stat=c, eta2=ssb / (sst + 1e-12)))
    return pd.DataFrame(rows)
