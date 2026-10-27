import numpy as np
from scipy import stats
from .fit import fit_full_and_null
import pandas as pd

def lrt(full, null):
    stat = 2 * (full.llf - null.llf)
    dof = len(full.fe_params) - len(null.fe_params)
    return stat, dof, stats.chi2.sf(stat, dof)


def permutation_lrt(df, time_term, n_perm=200, seed=0, adj=''):
    """Shuffle class labels between SUBJECTS (never between observations)."""
    rng = np.random.default_rng(seed)
    full, null = fit_full_and_null(df, time_term, adj=adj)
    observed, _, _ = lrt(full, null)
    labels = df.groupby("subject")["group"].first()
    perm_stats = []
    for _ in range(n_perm):
        shuffled = pd.Series(rng.permutation(labels.values), index=labels.index)
        d = df.assign(group=df["subject"].map(shuffled))
        f, n = fit_full_and_null(d, time_term, adj=adj)
        perm_stats.append(2 * (f.llf - n.llf))
    p = (1 + np.sum(np.array(perm_stats) >= observed)) / (n_perm + 1)
    return observed, p

