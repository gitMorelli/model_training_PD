import numpy as np
import pandas as pd
import patsy
from scipy.stats import multivariate_normal

def _design(result, tgrid, g, **cols):
    newdata = pd.DataFrame({"time": tgrid, "group": g, **cols})
    return np.asarray(patsy.build_design_matrices([result.design_info], newdata)[0])

def _population_curve(full, df, tgrid, g, model_split, slot_weights="pooled", bw=None):
    """Fixed-effect prediction for class g; with an adjustment column (slot or n_quest),
    averaged over its values with the SAME weights the difference curve uses, so
    (curve B) - (curve A) equals Delta(t)."""
    beta = full.fe_params.values
    col = _adj_col(model_split)
    if not col:
        return _design(full, tgrid, g) @ beta
    groups = sorted(df["group"].unique())
    vals, Wg = _group_slot_weights(df, tgrid, groups, slot_weights, bw, col=col)
    preds = np.column_stack([_design(full, tgrid, g, **{col: k}) @ beta for k in vals])
    return (preds * Wg[g]).sum(axis=1)

def _information_criteria(res):
    """AIC/BIC of an ML fit. Parameters: fixed effects + random-effect (co)variances + residual variance."""
    q = res.cov_re.shape[0]
    n_vc = res.model.k_vc
    k = len(res.fe_params) + q * (q + 1) // 2 + n_vc + 1
    n = int(res.nobs)
    return k, -2 * res.llf + 2 * k, -2 * res.llf + k * np.log(n)

# ----------------------------------------------------------------------------
# Adjustments: questionnaire slot, or number of questionnaires behind the prediction
# ----------------------------------------------------------------------------
ADJUSTMENTS = (None, "slot", "n_quest_add", "n_quest_int")


def adjustment_spec(adjust, n_form="log"):
    """Formula terms appended to `y ~ {time_term} * C(group)`, the data column they use,
    and whether that column enters as a categorical factor.

    adjust  None          no adjustment
            "slot"        C(split) * C(group): class-specific level per questionnaire slot
            "n_quest_add" f(n_quest): the number of questionnaires shifts BOTH classes equally.
                          The gap is then adjusted for the classes having different numbers of
                          questionnaires at the same time; the n_quest term cancels in Delta(t).
            "n_quest_int" f(n_quest) * C(group): the separation itself may grow with the number
                          of questionnaires. Collinear with time (n_quest rises toward diagnosis),
                          so the gap at a FIXED n_quest can look flat even when the total isn't.
    n_form  "log" (each extra questionnaire adds less), "linear", or "cat" (C(n_quest); sparse
            high counts, and unseen levels cannot be predicted)
    """
    if adjust in (None, False, "", "none"):
        return "", None, False
    if adjust is True or adjust == "slot":
        return " + C(split) * C(group)", "split", True
    if adjust in ("n_quest_add", "n_quest_int"):
        term = {"log": "np.log(n_quest)", "linear": "n_quest", "cat": "C(n_quest)"}[n_form]
        if adjust == "n_quest_int":
            term = f"{term} * C(group)"
        return f" + {term}", "n_quest", n_form == "cat"
    raise ValueError(f"adjust must be one of {ADJUSTMENTS}")


def _adj_col(model_split):
    """Prediction column of an adjustment: True / "slot" -> "split";
    "n_quest_add" / "n_quest_int" -> "n_quest"; a column name passes through; falsy -> None."""
    if model_split is True or model_split == "slot":
        return "split"
    if model_split in ("n_quest_add", "n_quest_int"):
        return "n_quest"
    return model_split or None

# ----------------------------------------------------------------------------
# 0. Example data (replace with your own DataFrame)
# ----------------------------------------------------------------------------
def simulate_data(n_per_group=40, true_onset=4.0, seed=1):
    """Two classes share a trajectory until `true_onset`, then class B rises faster."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in ["A", "B"]:
        for i in range(n_per_group):
            sid = f"{g}{i:03d}"
            k = rng.integers(3, 12)                          # k_s observations
            t = np.sort(rng.uniform(0, 10, size=k))          # irregular times
            b0, b1 = rng.normal(0, 1.0), rng.normal(0, 0.15)  # subject random effects
            mean = 2 + np.sin(t / 2) + b0 + b1 * t
            if g == "B":
                mean += 0.6 * np.maximum(t - true_onset, 0)
            y = mean + rng.normal(0, 0.5, size=k)
            rows += [dict(subject=sid, group=g, time=tt, y=yy) for tt, yy in zip(t, y)]
    return pd.DataFrame(rows)


def _kernel_mass(df, tgrid, bw=None, slots=None, col="split"):
    """Unnormalised kernel mass of each value of `col` at each t: (values, (n_grid, n_values))."""
    if slots is None:
        slots = np.sort(df[col].unique())
    t, s = df["time"].values, df[col].values
    if bw is None:
        bw = 1.06 * np.std(t) * len(t) ** (-0.2)
    K = np.exp(-0.5 * ((np.asarray(tgrid)[:, None] - t[None, :]) / bw) ** 2)
    return slots, np.column_stack([K[:, s == k].sum(axis=1) for k in slots])


def _slot_weights(df, tgrid, bw=None, slots=None, col="split"):
    """P(value of `col` | t) on tgrid via Gaussian kernel smoothing of the observed (time, value)
    pairs. `col` is the adjustment column: "split" (slot) or "n_quest".

    slots: fixes the column order (and lets a value with no rows in `df` get weight 0).
    bw:    kernel bandwidth in time units; default is Silverman's rule on all times, which can
           be too wide if each questionnaire sits in a narrow time window - check the mix
           panel and pass a smaller bw if the weights look smeared.
    """
    slots, W = _kernel_mass(df, tgrid, bw, slots, col)
    return slots, W / np.maximum(W.sum(axis=1, keepdims=True), 1e-300)


def _group_slot_weights(df, tgrid, groups, slot_weights="pooled", bw=None, col="split"):
    """{group: (n_grid, n_values) weight matrix} for the chosen weighting scheme."""
    slots = np.sort(df[col].unique())
    if slot_weights == "pooled":
        _, W = _slot_weights(df, tgrid, bw=bw, slots=slots, col=col)
        return slots, {g: W for g in groups}
    if slot_weights == "group":
        return slots, {g: _slot_weights(df[df["group"] == g], tgrid, bw=bw, slots=slots, col=col)[1]
                       for g in groups}
    raise ValueError("slot_weights must be 'pooled' or 'group'")


def _support(df, tgrid, col, value, bw=None):
    """Share of observations at each t (kernel-smoothed) that have col == value."""
    vals, K = _kernel_mass(df, tgrid, bw, None, col)
    if value not in set(vals):
        return np.zeros(len(tgrid))
    return K[:, list(vals).index(value)] / np.maximum(K.sum(axis=1), 1e-300)


def significant_intervals(curve):
    """Contiguous time intervals where the simultaneous band excludes zero."""
    sig = curve["significant"].values
    t = curve["time"].values
    intervals, start = [], None
    for i, s in enumerate(sig):
        if s and start is None:
            start = t[i]
        if not s and start is not None:
            intervals.append((start, t[i - 1])); start = None
    if start is not None:
        intervals.append((start, t[-1]))
    return intervals

# ----------------------------------------------------------------------------
# Compare training ranges: same model, trained on different time intervals,
# evaluated on the same interval with subject-level cross-validation.
# ----------------------------------------------------------------------------
def _in_interval(d, iv):
    lo, hi = iv
    m = np.ones(len(d), dtype=bool)
    if lo is not None:
        m &= d["time"].values >= lo
    if hi is not None:
        m &= d["time"].values < hi
    return m
 
 
def _interval_label(iv):
    lo, hi = iv
    return f"[{'-inf' if lo is None else f'{lo:g}'}, {'inf' if hi is None else f'{hi:g}'})"
 
 
def _estimable_slots(d, min_obs=5, col="split"):
    """Levels of `col` with at least `min_obs` observations in BOTH classes
    (needed for C(col) * C(group))."""
    counts = d.groupby([col, "group"]).size().unstack(fill_value=0)
    if counts.shape[1] < 2:
        return set()
    return set(counts.index[(counts >= min_obs).all(axis=1)])
 
 
def _heldout_loglik(res, test, matched_sets):
    """Log-likelihood of unseen subjects' observations under a fitted model, with the random
    effects integrated out exactly as in the model. Also returns the sum of squared errors of
    the population (fixed-effect) prediction."""
 
    X = np.asarray(patsy.build_design_matrices([res.design_info], test)[0])
    mu = X @ res.fe_params.values
    structure = getattr(res, "re_structure", "set" if matched_sets else "subject")
    center = getattr(res, "re_center", 0.0)        # time origin of the random slope
    y, t, s2 = test["y"].values, test["time"].values - center, res.scale
    if structure == "set":
        vc = dict(zip(res.model.exog_vc.names, res.vcomp))
        v_set, v_int, v_sl = float(np.asarray(res.cov_re)[0, 0]), vc["subj_int"], vc["subj_slope"]
        subj = test["subject"].to_numpy(dtype=object)
        units = test.groupby("set").indices
    elif structure == "subject_uncorr":
        vc = dict(zip(res.model.exog_vc.names, res.vcomp))
        G = np.diag([float(np.asarray(res.cov_re)[0, 0]), vc["subj_slope"]])
        units = test.groupby("subject").indices
    else:
        G = np.asarray(res.cov_re)
        units = test.groupby("subject").indices
    ll = 0.0
    for idx in units.values():
        ti = t[idx]
        if structure == "set":
            same = subj[idx][:, None] == subj[idx][None, :]
            C = v_set + same * (v_int + v_sl * np.outer(ti, ti))
        else:
            Z = np.column_stack([np.ones(len(idx)), ti])
            C = Z @ G @ Z.T
        ll += multivariate_normal.logpdf(y[idx], mu[idx], C + s2 * np.eye(len(idx)))
    return ll, float(np.sum((y - mu) ** 2))