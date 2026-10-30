"""Stage 2: choose which sequences get the (expensive, qualitative) explanations.

The rule is fixed and written to disk BEFORE any explanation is computed, so the
examples you end up showing are not cherry-picked.

Cells (per true class):
    confident_correct   correct and margin >= hi
    confident_wrong     wrong and margin >= hi        (the most informative cell)
    borderline          margin <= lo                  (correct or not; may be empty)
    lowconf_correct     correct and lo < margin < hi
    nearest_boundary    the n smallest margins, whatever their value (never empty)
plus
    cluster_medoid      medoid of each k-means cluster of pooled sequence embeddings
    outlier             farthest sequences from their cluster centre
    random              uniform random control set

margin is the distance from the decision boundary in [0, 1] (see predictions.py); for a
binary model with threshold 0.5, margin 0.8 <=> p >= 0.9 and margin 0.2 <=> 0.4 <= p <= 0.6.
Thresholds are fixed rather than quantiles: on a confident model, quantile-based
'borderline' cases can have p = 0.95.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SelectionRule:
    n_per_cell: int = 10
    margin_hi: Optional[float] = 0.8       # None -> 75th percentile of margins
    margin_lo: Optional[float] = 0.2       # None -> 15th percentile of margins
    pick: str = 'extreme'                  # 'extreme' (most confident / most borderline) or 'random'
    n_clusters: int = 8
    n_outliers: int = 5
    n_random: int = 20
    stratify_by: list = field(default_factory=list)   # metadata columns to balance within cells
    seed: int = 0


def _take(df, n, order_col, ascending, pick, rng, strat):
    if len(df) <= n:
        return df.index.tolist()
    if strat:
        groups = list(df.groupby(strat))
        per = max(1, n // len(groups))
        out = []
        for _, g in groups:
            out += _take(g, per, order_col, ascending, pick, rng, None)
        return out[:n]
    if pick == 'extreme':
        return df.sort_values(order_col, ascending=ascending).index[:n].tolist()
    return rng.choice(df.index.values, n, replace=False).tolist()


def select_sequences(preds: pd.DataFrame, rule: SelectionRule = SelectionRule(),
                     embeddings: Optional[dict] = None, meta: Optional[pd.DataFrame] = None,
                     out_dir: Optional[str] = None) -> pd.DataFrame:
    """Return a manifest (seq_id, cell, true class, pred, margin, reason). A sequence can
    appear in more than one cell; `cells` lists them all and `cell` is the first."""
    rng = np.random.default_rng(rule.seed)
    df = preds.copy()
    if meta is not None:
        df = df.join(meta, how='left')
    hi = rule.margin_hi if rule.margin_hi is not None else float(df['margin'].quantile(0.75))
    lo = rule.margin_lo if rule.margin_lo is not None else float(df['margin'].quantile(0.15))
    strat = rule.stratify_by or None
    picked: dict[str, list] = {}

    def add(ids, cell, reason):
        for i in ids:
            picked.setdefault(i, []).append((cell, reason))

    for c in sorted(df['label'].unique()):
        d = df[df['label'] == c]
        cells = {
            'confident_correct': (d[d.correct & (d.margin >= hi)], 'margin', False),
            'confident_wrong': (d[~d.correct & (d.margin >= hi)], 'margin', False),
            'borderline': (d[d.margin <= lo], 'margin', True),
            'lowconf_correct': (d[d.correct & (d.margin > lo) & (d.margin < hi)], 'margin', True),
            'nearest_boundary': (d, 'margin', True),
        }
        # if no confident errors exist, still look at the worst errors available
        if len(cells['confident_wrong'][0]) == 0 and (~d.correct).any():
            cells['confident_wrong'] = (d[~d.correct], 'margin', False)
        for name, (sub, col, asc) in cells.items():
            pick = 'extreme' if name == 'nearest_boundary' else rule.pick
            ids = _take(sub, rule.n_per_cell, col, asc, pick, rng, strat)
            add(ids, name, f'label={c}, {len(sub)} candidates, margin hi={hi:.3f} lo={lo:.3f}')

    if embeddings is not None and rule.n_clusters > 0:
        from sklearn.cluster import KMeans
        ids = np.array([str(i) for i in embeddings['ids']])
        keep = np.isin(ids, df.index.values)
        Z, ids = embeddings['z'][keep], ids[keep]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
        kc = min(rule.n_clusters, len(Z))
        km = KMeans(kc, n_init=10, random_state=rule.seed).fit(Z)
        dist = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
        for c in range(kc):
            members = np.where(km.labels_ == c)[0]
            add([ids[members[dist[members].argmin()]]], 'cluster_medoid',
                f'cluster {c} ({len(members)} seqs)')
        add(list(ids[np.argsort(-dist)[:rule.n_outliers]]), 'outlier', 'far from cluster centre')
        cl = pd.Series(km.labels_, index=ids, name='cluster')
        df = df.join(cl, how='left')

    add(rng.choice(df.index.values, min(rule.n_random, len(df)), replace=False).tolist(),
        'random', 'uniform control')

    rows = []
    for sid, lst in picked.items():
        r = df.loc[sid]
        rows.append(dict(seq_id=sid, cell=lst[0][0], cells='|'.join(sorted({x[0] for x in lst})),
                         label=int(r['label']), pred=int(r['pred']), p_pred=float(r['p_pred']),
                         margin=float(r['margin']), correct=bool(r['correct']),
                         cluster=r.get('cluster', np.nan), reason=lst[0][1]))
    man = pd.DataFrame(rows).set_index('seq_id').sort_values(['cell', 'label', 'margin'])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        man.to_csv(os.path.join(out_dir, 'selection.csv'))
        with open(os.path.join(out_dir, 'selection_rule.json'), 'w') as fh:
            json.dump(dict(**asdict(rule), margin_hi_used=hi, margin_lo_used=lo,
                           n_selected=len(man),
                           cell_counts=man['cells'].str.split('|').explode().value_counts().to_dict()),
                      fh, indent=2)
    return man
