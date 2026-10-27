from .fit import fit_full_and_null
from .helpers import _design, _group_slot_weights, _adj_col, _support
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------
# 3. Difference curve with simultaneous confidence band
# ----------------------------------------------------------------------------
def difference_curve(result, df, n_grid=200, n_sim=10000, level=0.95, seed=0,
                     model_split=False, ref_split=None, baseline_time=None,
                     slot_mode="marginal", slot_weights="pooled", bw=None, min_support=None):
    """Delta(t) = mean(class 2) - mean(class 1), with pointwise and simultaneous bands.

    model_split   adjustment in the model: False/None, True or "slot" (C(split) * C(group)),
                  "n_quest_add" / "n_quest_int" (number of questionnaires), or a column name
    slot_mode     "marginal"  (default) gap averaged over the values of the adjustment column
                              observed at each t (slots, or numbers of questionnaires)
                  "reference" gap at one value (ref_split, default: most frequent), i.e. the gap
                              holding the slot / number of questionnaires fixed
    slot_weights  "pooled" or "group": compare both classes at the same mix, or each at its own;
                  only used when marginal
    min_support   reference mode only: keep the times where P(ref value | t) >= min_support, so
                  a gap at a fixed n_quest is not extrapolated to times where nobody has that n
    baseline_time if given, the curve is Delta(t) - Delta(baseline_time): how much the gap
                  has GROWN since the baseline. In marginal mode this growth includes the
                  change in mix between baseline and t (it is the total growth).

    With "n_quest_add" the n_quest term is identical in both classes and cancels: marginal and
    reference curves coincide (with pooled weights).
    """
    col = _adj_col(model_split)
    groups = sorted(df["group"].unique())
    lo = df.groupby("group")["time"].min().max()   # time range where BOTH classes have data
    hi = df.groupby("group")["time"].max().min()
    grid = np.linspace(lo, hi, n_grid)
    support = np.ones(len(grid))

    if not col:
        L = _design(result, grid, groups[1]) - _design(result, grid, groups[0])
    elif slot_mode == "reference":
        ref = df[col].mode()[0] if ref_split is None else ref_split
        if min_support is not None:
            support = _support(df, grid, col, ref, bw)
            keep = np.flatnonzero(support >= min_support)
            if len(keep) < 2:
                raise ValueError(f"{col} = {ref} is (almost) never observed; lower min_support")
            sl = slice(keep[0], keep[-1] + 1)
            grid, support = grid[sl], support[sl]
        L = (_design(result, grid, groups[1], **{col: ref})
             - _design(result, grid, groups[0], **{col: ref}))
    elif slot_mode == "marginal":
        vals, Wg = _group_slot_weights(df, grid, groups, slot_weights, bw, col=col)
        Lg = {}
        for g in groups:
            Xs = np.stack([_design(result, grid, g, **{col: k}) for k in vals], axis=1)  # (t, value, p)
            Lg[g] = np.einsum("ts,tsp->tp", Wg[g], Xs)
        L = Lg[groups[1]] - Lg[groups[0]]
    else:
        raise ValueError("slot_mode must be 'marginal' or 'reference'")

    if baseline_time is not None:
        i0 = np.argmin(np.abs(grid - baseline_time))
        L = L - L[i0]
        keep = np.arange(len(grid)) != i0          # se = 0 at the baseline point itself
        L, grid, support = L[keep], grid[keep], support[keep]

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
        "support": support,
    })
    out["significant"] = (out["sim_lo"] > 0) | (out["sim_hi"] < 0)
    return out, crit_sim, groups


def curves_by_value(result, df, model_split, values=None, min_support=0.05, **kw):
    """{value: gap at a FIXED value of the adjustment column} (e.g. n_quest = 2, 4, 6, 8), each
    restricted to the times where that value is observed (P(value | t) >= min_support).
    Default values: the 10th, 35th, 65th and 90th percentiles of the column."""
    col = _adj_col(model_split)
    if values is None:
        values = np.unique(np.quantile(df[col], [0.1, 0.35, 0.65, 0.9]).round().astype(int))
    out = {}
    for v in values:
        try:
            out[v], _, _ = difference_curve(result, df, model_split=model_split, slot_mode="reference",
                                            ref_split=v, min_support=min_support, **kw)
        except ValueError as e:
            print(f"  {col} = {v}: skipped ({e})")
    return out
