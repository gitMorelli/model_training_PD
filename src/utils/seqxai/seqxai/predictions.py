"""Load the per-sequence predictions CSV and derive confidence measures."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd


def load_predictions(path_or_df, id_col: str = 'id', label_col: str = 'label',
                     prob_cols: Union[str, Sequence[str]] = 'prob',
                     pred_col: Optional[str] = None, threshold: float = 0.5) -> pd.DataFrame:
    """Return a DataFrame indexed by seq_id (str) with standard columns:

        label, pred, p_pred (prob of predicted class), p_true (prob of true class),
        margin (distance from the decision boundary, in [0, 1]), correct,
        prob_<c> for every class c.

    Binary: pass a single column with P(class 1) as `prob_cols`.
    Multiclass: pass one column per class, in class order.
    If ids appear several times (e.g. one row per image), rows are averaged per id
    and a warning is printed: the pipeline explains whole sequences.
    """
    df = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df.copy()
    cols = [prob_cols] if isinstance(prob_cols, str) else list(prob_cols)
    missing = [c for c in [id_col, label_col, *cols] if c not in df.columns]
    if missing:
        raise KeyError(f'columns not in CSV: {missing}; available: {list(df.columns)}')
    df[id_col] = df[id_col].astype(str)
    if df[id_col].duplicated().any():
        n = df[id_col].duplicated().sum()
        print(f'[predictions] {n} duplicated ids: averaging probabilities per id')
        df = df.groupby(id_col, as_index=False).agg({label_col: 'first', **{c: 'mean' for c in cols}})

    out = pd.DataFrame(index=df[id_col].values)
    out.index.name = 'seq_id'
    out['label'] = df[label_col].astype(int).values
    if len(cols) == 1:
        p1 = df[cols[0]].astype(float).values
        P = np.stack([1 - p1, p1], 1)
        pred = (p1 >= threshold).astype(int)
        margin = np.abs(p1 - threshold) / max(threshold, 1 - threshold)
    else:
        P = df[cols].astype(float).values
        pred = P.argmax(1)
        s = np.sort(P, 1)
        margin = s[:, -1] - s[:, -2]
    if pred_col is not None:
        pred = df[pred_col].astype(int).values
    for c in range(P.shape[1]):
        out[f'prob_{c}'] = P[:, c]
    out['pred'] = pred
    out['p_pred'] = P[np.arange(len(P)), pred]
    out['p_true'] = P[np.arange(len(P)), out['label'].values]
    out['margin'] = margin
    out['correct'] = out['pred'] == out['label']
    out.attrs['n_classes'] = P.shape[1]
    out.attrs['binary_single_col'] = len(cols) == 1
    out.attrs['threshold'] = threshold
    return out
