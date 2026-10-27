from .fit import fit_full_and_null
from .helpers import _design, _group_slot_weights
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------
# 3. Difference curve with simultaneous confidence band
# ----------------------------------------------------------------------------
def difference_curve(result, df, n_grid=200, n_sim=10000, level=0.95, seed=0,
                     model_split=False, ref_split=None, baseline_time=None,
                     slot_mode="marginal", slot_weights="pooled", bw=None):
    """Delta(t) = mean(class 2) - mean(class 1), with pointwise and simultaneous bands.
 
    model_split   the model contains C(split) * C(group)
    slot_mode     "marginal"  (default) gap averaged over the slots observed at each t
                  "reference" gap at one slot (ref_split, default: most frequent) - the old
                              behaviour, i.e. the gap holding the questionnaire fixed
    slot_weights  "pooled" or "group" (see module docstring); only used when marginal
    baseline_time if given, the curve is Delta(t) - Delta(baseline_time): how much the gap
                  has GROWN since the baseline. In marginal mode this growth includes the
                  change in slot mix between baseline and t (it is the total growth).
    """
    groups = sorted(df["group"].unique())
    lo = df.groupby("group")["time"].min().max()   # time range where BOTH classes have data
    hi = df.groupby("group")["time"].max().min()
    grid = np.linspace(lo, hi, n_grid)
 
    if not model_split:
        L = _design(result, grid, groups[1]) - _design(result, grid, groups[0])
    elif slot_mode == "reference":
        ref = df["split"].mode()[0] if ref_split is None else ref_split
        L = _design(result, grid, groups[1], split=ref) - _design(result, grid, groups[0], split=ref)
    elif slot_mode == "marginal":
        slots, Wg = _group_slot_weights(df, grid, groups, slot_weights, bw)
        Lg = {}
        for g in groups:
            Xs = np.stack([_design(result, grid, g, split=k) for k in slots], axis=1)  # (t, slot, p)
            Lg[g] = np.einsum("ts,tsp->tp", Wg[g], Xs)
        L = Lg[groups[1]] - Lg[groups[0]]
    else:
        raise ValueError("slot_mode must be 'marginal' or 'reference'")
 
    if baseline_time is not None:
        i0 = np.argmin(np.abs(grid - baseline_time))
        L = L - L[i0]
        keep = np.arange(len(grid)) != i0          # se = 0 at the baseline point itself
        L, grid = L[keep], grid[keep]
 
    beta = result.fe_params.values
    V = result.cov_params().loc[result.fe_params.index, result.fe_params.index].values
    delta = L @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", L, V, L))
 
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.zeros_like(beta), V, size=n_sim)
    crit_sim = np.quantile(np.max(np.abs(draws @ L.T) / se, axis=1), level)
    crit_pw = stats.norm.ppf(0.5 + level / 2)
 
    out = pd.DataFrame({
        "time": grid, "delta": delta, "se": se,
        "pw_lo": delta - crit_pw * se, "pw_hi": delta + crit_pw * se,
        "sim_lo": delta - crit_sim * se, "sim_hi": delta + crit_sim * se,
    })
    out["significant"] = (out["sim_lo"] > 0) | (out["sim_hi"] < 0)
    return out, crit_sim, groups
