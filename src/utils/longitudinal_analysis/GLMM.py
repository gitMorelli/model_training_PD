"""
Do two classes of subjects differ in how a quantity evolves over time,
and when does the difference start?

Data format (long): one row per observation, columns
    subject : subject id
    group   : class label (exactly two levels)
    time    : observation time (irregular, k_s per subject is fine)
    y       : measured quantity

Pipeline
    1. Linear mixed model with spline time trends per class
       (random intercept + slope per subject)                     -> statsmodels MixedLM
    2. Global test: does the class change level or shape?        -> likelihood-ratio test (ML fits)
    3. Difference curve Delta(t) with SIMULTANEOUS 95% band       -> posterior simulation of fixed effects
       Onset = first time the simultaneous band excludes 0
    4. (Optional) change-point model: identical curves until tau,
       class 2 diverges linearly after tau                        -> profile likelihood over tau
    5. (Optional) subject-level permutation test of step 2       -> robustness check

Requires: numpy, pandas, scipy, statsmodels, patsy, matplotlib
"""

import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")  # MixedLM is chatty about convergence/boundary warnings

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


# ----------------------------------------------------------------------------
# 1. Model fitting
# ----------------------------------------------------------------------------
def fit_mixed(formula, df, reml=False, re_formula="~time"):
    """Linear mixed model with random intercept + random slope in time per subject.
    Use reml=False (ML) whenever fixed-effect structures are compared by LRT.

    Design matrices are built with patsy directly so that the spline knots are
    stored (result.design_info) and can be reused to predict on new time grids."""
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    Z = patsy.dmatrix(re_formula, df, return_type="dataframe")
    model = sm.MixedLM(y, X, groups=df["subject"], exog_re=Z)
    result = model.fit(reml=reml, method=["lbfgs", "powell"])
    result.design_info = X.design_info
    return result


def fit_full_and_null(df, time_term, adj=''):
    full = fit_mixed(f"y ~ {time_term} * C(group){adj}", df)  # class-specific trajectories
    null = fit_mixed(f"y ~ {time_term}{adj}", df)             # one shared trajectory
    return full, null


# ----------------------------------------------------------------------------
# 2. Global test
# ----------------------------------------------------------------------------
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
# 4. Change-point model: profile likelihood for the onset tau
# ----------------------------------------------------------------------------
def changepoint_profile(df, time_term,tau_grid=None, level=0.95):
    """y = f(t) + delta * 1[class 2] * (t - tau)_+ + random effects.

    Assumes the classes are IDENTICAL before tau and diverge linearly after.
    The likelihood-based CI is approximate (change-point models are non-regular);
    for a more trustworthy interval, use cluster_bootstrap_tau below.
    """
    groups = sorted(df["group"].unique())
    is_g2 = (df["group"] == groups[1]).astype(float).values
    if tau_grid is None:
        tau_grid = np.linspace(*np.quantile(df["time"], [0.05, 0.9]), 40)

    llf = []
    for tau in tau_grid:
        d = df.assign(hinge=is_g2 * np.maximum(df["time"].values - tau, 0))
        llf.append(fit_mixed(f"y ~ {time_term} + hinge", d).llf)
    llf = np.array(llf)

    tau_hat = tau_grid[np.argmax(llf)]
    inside = 2 * (llf.max() - llf) <= stats.chi2.ppf(level, 1)
    ci = (tau_grid[inside].min(), tau_grid[inside].max())
    return tau_hat, ci, pd.DataFrame({"tau": tau_grid, "llf": llf})


def cluster_bootstrap_tau(df, time_term,n_boot=200, tau_grid=None, seed=0):
    """Resample whole SUBJECTS with replacement, re-estimate tau each time."""
    rng = np.random.default_rng(seed)
    subjects = df.groupby("group")["subject"].unique()
    taus = []
    for b in range(n_boot):
        parts = []
        for g, subs in subjects.items():  # stratified by class
            for j, s in enumerate(rng.choice(subs, size=len(subs), replace=True)):
                parts.append(df[df["subject"] == s].assign(subject=f"{s}_b{j}"))
        tau_b, _, _ = changepoint_profile(pd.concat(parts), time_term=time_term, tau_grid=tau_grid)
        taus.append(tau_b)
    return np.percentile(taus, [2.5, 97.5]), np.array(taus)


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def _slot_weights(df, tgrid, bw=None, slots=None):
    """P(slot | t) on tgrid via Gaussian kernel smoothing of the observed (time, slot) pairs.
 
    slots: fixes the column order (and lets a slot with no rows in `df` get weight 0).
    bw:    kernel bandwidth in time units; default is Silverman's rule on all times, which can
           be too wide if each questionnaire sits in a narrow time window - check the slot-mix
           panel and pass a smaller bw if the weights look smeared.
    """
    if slots is None:
        slots = np.sort(df["split"].unique())
    t, s = df["time"].values, df["split"].values
    if bw is None:
        bw = 1.06 * np.std(t) * len(t) ** (-0.2)
    K = np.exp(-0.5 * ((np.asarray(tgrid)[:, None] - t[None, :]) / bw) ** 2)
    W = np.column_stack([K[:, s == k].sum(axis=1) for k in slots])
    return slots, W / np.maximum(W.sum(axis=1, keepdims=True), 1e-300)

def _group_slot_weights(df, tgrid, groups, slot_weights="pooled", bw=None):
    """{group: (n_grid, n_slots) weight matrix} for the chosen weighting scheme."""
    slots = np.sort(df["split"].unique())
    if slot_weights == "pooled":
        _, W = _slot_weights(df, tgrid, bw=bw, slots=slots)
        return slots, {g: W for g in groups}
    if slot_weights == "group":
        return slots, {g: _slot_weights(df[df["group"] == g], tgrid, bw=bw, slots=slots)[1]
                       for g in groups}
    raise ValueError("slot_weights must be 'pooled' or 'group'")

def _design(result, tgrid, g, **cols):
    newdata = pd.DataFrame({"time": tgrid, "group": g, **cols})
    return np.asarray(patsy.build_design_matrices([result.design_info], newdata)[0])

def _population_curve(full, df, tgrid, g, model_split, slot_weights="pooled", bw=None):
    """Fixed-effect prediction for class g; with slots, averaged over slots with the SAME
    weights the difference curve uses, so (curve B) - (curve A) equals Delta(t)."""
    beta = full.fe_params.values
    if not model_split:
        return _design(full, tgrid, g) @ beta
    groups = sorted(df["group"].unique())
    slots, Wg = _group_slot_weights(df, tgrid, groups, slot_weights, bw)
    preds = np.column_stack([_design(full, tgrid, g, split=k) @ beta for k in slots])
    return (preds * Wg[g]).sum(axis=1)
 
def _fit(formula, df, reml=False, matched_sets=False):
    """Mixed model with subject intercept + slope.
    matched_sets=False: subject random intercept and slope (correlated), as in fit_mixed.
    matched_sets=True:  adds a random intercept per matched set; subjects nested in sets.
                        (statsmodels cannot keep the subject intercept-slope correlation here.)"""
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    if matched_sets:
        vc = {"subj_int": "0 + C(subject)", "subj_slope": "0 + C(subject):time"}
        model = sm.MixedLM.from_formula(formula, df, groups="set", re_formula="1", vc_formula=vc)
    else:
        Z = patsy.dmatrix("~time", df, return_type="dataframe")
        model = sm.MixedLM(y, X, groups=df["subject"], exog_re=Z)
    result = model.fit(reml=reml, method=["lbfgs", "powell"])
    result.design_info = X.design_info
    return result
def plot_results(df, full, curve, groups, profile=None, path="results.png",
                 model_split=False, show_binned=True, n_bins=15):
    import matplotlib.pyplot as plt

    ncols = 2 + (profile is not None) + model_split
    fig, ax = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5))
    colors = dict(zip(groups, ["tab:blue", "tab:orange"]))
    tgrid = curve["time"].values
    bins = np.quantile(df["time"], np.linspace(0, 1, n_bins + 1))

    for g in groups:
        sub = df[df["group"] == g]
        for _, s in sub.groupby("subject"):
            ax[0].plot(s["time"], s["y"], color=colors[g], alpha=0.12, lw=0.8)
        ax[0].plot(tgrid, _population_curve(full, df, tgrid, g, model_split),
                   color=colors[g], lw=3, label=g)
        if show_binned:                              # raw binned means: a visual fit check
            b = sub.groupby(pd.cut(sub["time"], bins, include_lowest=True), observed=True)
            ax[0].plot(b["time"].mean(), b["y"].mean(), "o", color=colors[g],
                       mec="k", ms=5, zorder=5)

    subtitle = "\n(averaged over slots, weighted by P(slot | t))" if model_split else ""
    ax[0].set(title="Population trajectories" + subtitle, xlabel="time", ylabel="y")
    ax[0].legend(title="class")

    t = curve["time"]
    ax[1].fill_between(t, curve["sim_lo"], curve["sim_hi"], alpha=0.25, color="grey",
                       label="simultaneous 95%")
    ax[1].fill_between(t, curve["pw_lo"], curve["pw_hi"], alpha=0.35, color="tab:green",
                       label="pointwise 95%")
    ax[1].plot(t, curve["delta"], color="k", lw=2)
    ax[1].axhline(0, color="k", ls=":")
    for a, b in significant_intervals(curve):
        ax[1].axvspan(a, b, color="tab:red", alpha=0.08)
    ax[1].set(title=f"Difference {groups[1]} − {groups[0]}", xlabel="time", ylabel="Δ(t)")
    ax[1].legend(loc="upper left")

    if profile is not None:
        ax[2].plot(profile["tau"], profile["llf"], "o-", ms=3)
        ax[2].set(title="Change-point profile likelihood", xlabel="τ", ylabel="log-likelihood")

    if model_split:
        plot_slot_mix(ax[-1], df, tgrid, by_group=True)   # dashed lines = cases' mix

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path

def plot_slot_mix(ax, df, tgrid, bw=None, by_group=False):
    """Stacked area of P(slot | t): shows which questionnaires the marginal curve averages over.
    by_group=True draws the case mix as lines on top of the pooled areas."""
    slots, W = _slot_weights(df, tgrid, bw=bw)
    ax.stackplot(tgrid, W.T, labels=[str(k) for k in slots], alpha=0.8)
    if by_group:
        groups = sorted(df["group"].unique())
        _, Wc = _slot_weights(df[df["group"] == groups[1]], tgrid, bw=bw, slots=slots)
        for c in np.cumsum(Wc, axis=1).T[:-1]:
            ax.plot(tgrid, c, color="k", lw=0.8, ls="--")
    ax.set(xlim=(tgrid.min(), tgrid.max()), ylim=(0, 1), xlabel="time", ylabel="P(slot | t)",
           title="Slot mix used for averaging" + ("\n(dashed: cases only)" if by_group else ""))
    ax.legend(title="slot", fontsize=7, ncol=2, frameon=False, loc="center left",
              bbox_to_anchor=(1.0, 0.5))

preset_columns = {             # your column name -> name used internally
        "subject":"unique_id",
        "group": "true_label",
        "time": "case_dt",
        "y": "probability_1",
        "split": "slot",
        "set": "case_control_group",
    }
def run_analysis(raw, columns=preset_columns, logit=True, eps=1e-4, path=None, spline_df=4,
                 run_permutation_test=False, n_perm=200, run_bootstrap=False, n_boot=200, time_model="spline", model_split=True,
                 use_baseline=True):
    '''logit = True            # model logit(probability_1) instead of the raw probability
    eps = 1e-4                # clip probabilities away from 0/1 before the logit
    '''
    SPLINE_DF = spline_df  # flexibility of the time trend; 3-5 is typical, check sensitivity
    if time_model == "spline":
        TIME_TERM = f"cr(time, df={spline_df}, constraints='center')"
    elif time_model == "quadratic":
        TIME_TERM = "(center(time) + I(center(time)**2))"
    else:
        raise ValueError("time_model must be 'spline' or 'quadratic'")
    if model_split:
        ADJ = " + C(split) * C(group)"
    else:
        ADJ = ""

    def prepare(raw):
        COLUMNS = {v: k for k, v in columns.items()}
        df = raw.rename(columns=COLUMNS)[list(COLUMNS.values())].dropna()
        df["time"] = df["time"].astype(float)
        if logit:
            p = df["y"].clip(eps, 1 - eps)
            df["y"] = np.log(p / (1 - p))
        assert df["group"].nunique() == 2, "expected exactly two classes in true_label"
        return df
    
    df = prepare(raw)
    # df = simulate_data()              # <- uncomment to run on simulated data instead
    print(f"{df['subject'].nunique()} subjects, {len(df)} observations\n")

    # Steps 1-2: models and global test
    full, null = fit_full_and_null(df, TIME_TERM, adj=ADJ)
    stat, dof, p = lrt(full, null)
    what = "does the gap change over time, within slot?" if model_split else "class affects level or shape"
    print(f"Global LRT ({what}): chi2={stat:.2f}, df={dof}, p={p:.3g}")

    # Step 3: difference curve (refit with REML for less biased variance components)
    full_reml = fit_mixed(f"y ~ {TIME_TERM} * C(group){ADJ}", df, reml=True)
    baseline_time = df["time"].quantile(0.05) if use_baseline else None
    curve, crit, groups = difference_curve(full_reml, df, model_split=model_split, baseline_time=baseline_time)
    print(f"Simultaneous critical value: {crit:.2f} (vs 1.96 pointwise)")
    intervals = significant_intervals(curve)
    if intervals:
        print("Simultaneous band excludes 0 on: "
              + ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in intervals))
        print(f"-> difference DETECTABLE from t ≈ {intervals[0][0]:.2f}")
    else:
        print("No time interval where the difference is significant.")

    # Step 4: change-point estimate of the onset
    tau_hat, tau_ci, profile = changepoint_profile(df, time_term=TIME_TERM+ADJ)
    print(f"\nChange-point onset: tau = {tau_hat:.2f}, "
          f"approx. 95% CI [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]")

    #step 5: interaction between split and group (optional; only if model_split=True)
    if model_split:
        '''# fit the additive model
        add = fit_mixed(f"y ~ {TIME_TERM} * C(group) + C(split)", df)
        het = fit_mixed(f"y ~ {TIME_TERM} * C(group) + C(split) * C(group)", df)
        stat, dof, p = lrt(het, add)   # does the class gap vary by slot beyond what time explains?
        print(f"\nInteraction test (does the class gap vary by slot beyond what time explains?): chi2={stat:.2f}, df={dof}, p={p:.3g}")'''
        # one extra gap slope per slot (reference slot excluded, so nothing is redundant)
        ref = df["split"].mode()[0]
        is_case = (df["group"] == sorted(df["group"].unique())[1]).astype(float)
        d, slope_cols = df.copy(), []
        for i, k in enumerate(sorted(df["split"].unique())):
            if k != ref:
                d[f"gslope_{i}"] = (df["split"] == k) * is_case * df["time"]
                slope_cols.append(f"gslope_{i}")
        het_slope = fit_mixed(f"y ~ {TIME_TERM} * C(group){ADJ} + " + " + ".join(slope_cols), d)
        stat, dof, p = lrt(het_slope, full)
        print(f"\nDoes the gap's time trend differ by slot? chi2={stat:.2f}, df={dof}, p={p:.3g}")

    if path:
        print("\nSaved plot:", plot_results(df, full_reml, curve, groups, profile, path=path, model_split=model_split))

    # Step 5 (slow; uncomment for robustness checks)
    if run_permutation_test:
        obs, p_perm = permutation_lrt(df, TIME_TERM,n_perm=n_perm, adj=ADJ); print(f"Permutation p = {p_perm:.3g}")
    if run_bootstrap:
        ci_boot, _ = cluster_bootstrap_tau(df, TIME_TERM+ADJ,n_boot=n_boot); print(f"Bootstrap tau CI: {ci_boot}")


def _information_criteria(res):
    """AIC/BIC of an ML fit. Parameters: fixed effects + random-effect (co)variances + residual variance."""
    q = res.cov_re.shape[0]
    n_vc = res.model.k_vc
    k = len(res.fe_params) + q * (q + 1) // 2 + n_vc + 1
    n = int(res.nobs)
    return k, -2 * res.llf + 2 * k, -2 * res.llf + k * np.log(n)

def compare_models(raw, columns=preset_columns, logit=True, eps=1e-4, path="model_comparison.png",
                    baseline_quantile=0.05, max_time=None, n_sim=10000, matched_sets=(False, True), nested_comparison=False):
    """Fit several models on the SAME data and compare them.
 
    logit              model logit(probability) instead of the raw probability
    eps                clip probabilities away from 0/1 before the logit
    baseline_quantile  baseline time for the "growth of the gap" curves (quantile of time)
    max_time           keep only observations with time < max_time (0 = pre-diagnosis only);
                       None keeps the full time axis
 
    matched_sets       which random-effect structures to run: False = subject only,
                       True = subject nested in matched set (needs a "set" column in `columns`)
 
    Each model gets two difference curves: the gap itself, and its growth since the baseline.
    """
    SPLINE_4 = f"cr(time, df=4, constraints='center')"
    SPLINE_8 = f"cr(time, df=8, constraints='center')"
    QUADRATIC = "(center(time) + I(center(time)**2))"
    SLOT = " + C(split) * C(group)"
    specs = {                                   # name -> (time term, adjust for slot?, matched sets)
        "spline_4":           (SPLINE_4, False, False),
        "spline_4_w_slot":    (SPLINE_4, True, False),
        "spline_4_w_match": (SPLINE_4, False, True),
        "spline_4_w_slot_match":    (SPLINE_4, True, True),
        "spline_8":         (SPLINE_8, False, False),
        "spline_8_w_match":  (SPLINE_8, False, True),
        #"quadratic_w_slot": (QUADRATIC, True),
        #"quadratic":        (QUADRATIC, False),
    }
    nested_pairs = [("spline_w_slot", "spline"), ("quadratic_w_slot", "quadratic")]
 
    # --- data -----------------------------------------------------------------
    COLUMNS = {v: k for k, v in columns.items()}
    df = raw.rename(columns=COLUMNS)[list(COLUMNS.values())].dropna()
    df["time"] = df["time"].astype(float)
    if max_time is not None:
        df = df[df["time"] < max_time].copy()
    if logit:
        p = df["y"].clip(eps, 1 - eps)
        df["y"] = np.log(p / (1 - p))
    assert df["group"].nunique() == 2, "expected exactly two classes"
    if True in matched_sets:
        assert "set" in df.columns, 'add "set": "case_control_group" to `columns`'
        assert df.groupby("subject")["set"].nunique().max() == 1, "each subject must be in one set"
    print(f"{df['subject'].nunique()} subjects, {len(df)} observations"
          + (f" (time < {max_time})" if max_time is not None else "") + "\n")
 
    baseline_time = df["time"].quantile(baseline_quantile)
    ref_split = df["split"].mode()[0]
 
    # --- fits -----------------------------------------------------------------
    rows, fits, curves = [], {}, {}
    for base in specs:
        tt, slot, ms = specs[base]
        name = base + (" + set" if ms else "")
        adj = SLOT if slot else ""
        print(f"Fitting {name} ...")
        full = _fit(f"y ~ {tt} * C(group){adj}", df, matched_sets=ms)          # ML: AIC/BIC/LRT
        no_trend = _fit(f"y ~ {tt} + C(group){adj}", df, matched_sets=ms)      # gap constant in time
        full_reml = _fit(f"y ~ {tt} * C(group){adj}", df, reml=True, matched_sets=ms)
 
        k, aic, bic = _information_criteria(full)
        stat, dof, p = lrt(full, no_trend)
        gap, crit_gap, groups = difference_curve(full_reml, df, n_sim=n_sim, model_split=slot,
                                                 ref_split=ref_split)
        growth, crit_growth, _ = difference_curve(full_reml, df, n_sim=n_sim, model_split=slot,
                                                  ref_split=ref_split, baseline_time=baseline_time)
        iv_gap, iv_growth = significant_intervals(gap), significant_intervals(growth)
        fits[name] = (full_reml, slot)
        curves[name] = {"gap": gap, "growth": growth}
        rows.append(dict(model=name, set_var=float(full.cov_re.iloc[0, 0]) if ms else np.nan,
                         n_params=k, loglik=full.llf, AIC=aic, BIC=bic,
                         trend_chi2=stat, trend_df=dof, trend_p=p,
                         onset_gap=iv_gap[0][0] if iv_gap else np.nan,
                         onset_growth=iv_growth[0][0] if iv_growth else np.nan))
 
    table = pd.DataFrame(rows).set_index("model")
    table["dAIC"] = table["AIC"] - table["AIC"].min()
    table["dBIC"] = table["BIC"] - table["BIC"].min()

    if nested_comparison:
        # --- nested comparisons (same random effects, more fixed effects) ---------
        nested = []
        for ms in matched_sets:
            sfx = " + set" if ms else ""
            for big, small in nested_pairs:
                f_big = _fit(f"y ~ {specs[big][0]} * C(group){SLOT}", df, matched_sets=ms)
                f_small = _fit(f"y ~ {specs[small][0]} * C(group)", df, matched_sets=ms)
                stat, dof, p = lrt(f_big, f_small)
                nested.append(dict(comparison=f"{big}{sfx} vs {small}{sfx}", chi2=stat, df=dof, p=p))
        nested = pd.DataFrame(nested).set_index("comparison")
 
    # --- report -----------------------------------------------------------------
    pd.set_option("display.width", 160)
    print("\nInformation criteria (ML fits, lower = better; dAIC < 2 ~ tie)")
    print(table[["n_params", "loglik", "AIC", "dAIC", "BIC", "dBIC"]].round(1).to_string())
    if True in matched_sets:
        print("\nMatched-set variance (ML; near 0 = the sets add nothing)")
        print(table["set_var"].dropna().round(4).to_string())
    print("\nDoes the class gap change over time?  (LRT: full model vs gap constant in time)")
    print(table[["trend_chi2", "trend_df", "trend_p"]].round(4).to_string())
    if nested_comparison:
        print("\nNested comparisons (do the slot terms C(split) * C(group) improve the fit?)")
        print(nested.round(4).to_string())
        print("\nSpline vs quadratic, and with vs without set, are NOT nested: compare them with AIC/BIC only.")
    print(f"\nOnset (first time the simultaneous band excludes 0)\n"
          f"  onset_gap:    the gap itself (slot models: at reference slot {ref_split})\n"
          f"  onset_growth: growth of the gap since baseline t = {baseline_time:.2f}")
    print(table[["onset_gap", "onset_growth"]].round(2).to_string())
 
    # --- figure -----------------------------------------------------------------
    if path:
        _plot_comparison(df, fits, curves, groups, table, baseline_time, ref_split, path)
        print("\nSaved plot:", path)
    return table, curves
 
 
def _plot_comparison(df, fits, curves, groups, table, baseline_time, ref_split, path, n_bins=15):
    """One row per model; columns: trajectories | gap | growth of the gap since baseline."""
    import matplotlib.pyplot as plt
 
    class_colors = dict(zip(groups, ["#2a78d6", "#eb6834"]))
    muted, band_sim, band_pw, sig = "#52514e", "#b9b8b2", "#1baf7a", "#e34948"
    n = len(fits)
    fig, ax = plt.subplots(n, 3, figsize=(16, 3.0 * n + 0.8), sharex=True, sharey="col",
                           squeeze=False)
    bins = np.quantile(df["time"], np.linspace(0, 1, n_bins + 1))
 
    for r, (name, (res, slot)) in enumerate(fits.items()):
        row = table.loc[name]
        # column 0: population trajectories + raw binned means
        tgrid = curves[name]["gap"]["time"].values
        for g in groups:
            col = class_colors[g]
            ax[r, 0].plot(tgrid, _population_curve(res, df, tgrid, g, slot), color=col, lw=2,
                          label=f"class {g}")
            sub = df[df["group"] == g]
            b = sub.groupby(pd.cut(sub["time"], bins, include_lowest=True), observed=True)
            ax[r, 0].plot(b["time"].mean(), b["y"].mean(), "o", color=col, mfc="white",
                          ms=6, mew=1.5, ls="none", zorder=5)
        ax[r, 0].set_ylabel(f"{name}\n\ny (logit)", fontsize=10)
        ax[r, 0].text(0.02, 0.97, f"dAIC {row['dAIC']:.1f}   dBIC {row['dBIC']:.1f}\n"
                                  f"gap changes in time: p = {row['trend_p']:.2g}",
                      transform=ax[r, 0].transAxes, va="top", fontsize=8, color=muted)
 
        # columns 1-2: difference curves
        for c, (kind, onset_col) in enumerate([("gap", "onset_gap"), ("growth", "onset_growth")], 1):
            cv, a = curves[name][kind], ax[r, c]
            a.fill_between(cv["time"], cv["sim_lo"], cv["sim_hi"], color=band_sim, alpha=0.5, lw=0,
                           label="simultaneous 95%")
            a.fill_between(cv["time"], cv["pw_lo"], cv["pw_hi"], color=band_pw, alpha=0.25, lw=0,
                           label="pointwise 95%")
            a.plot(cv["time"], cv["delta"], color="#0b0b0b", lw=2)
            for lo_, hi_ in significant_intervals(cv):
                a.axvspan(lo_, hi_, color=sig, alpha=0.08, lw=0)
            onset = row[onset_col]
            if np.isfinite(onset):
                a.plot(onset, np.interp(onset, cv["time"], cv["delta"]), "o", color=sig, ms=8,
                       mec="white", mew=2, zorder=6)
            a.text(0.02, 0.97, f"onset: {onset:.1f}" if np.isfinite(onset) else "no onset",
                   transform=a.transAxes, va="top", fontsize=8, color=muted)
            a.axhline(0, color=muted, ls=":", lw=1)
            if kind == "growth":
                a.axvline(baseline_time, color=muted, lw=1, ls="--")
 
    ax[0, 0].set_title("Population trajectories\n(dots: raw binned means; slot models "
                       "averaged over slots by P(slot | t))", fontsize=10)
    ax[0, 1].set_title(f"Gap {groups[1]} − {groups[0]}\n(slot models: averaged over slots by P(slot | t) "
                       f"{ref_split})", fontsize=10)
    ax[0, 2].set_title(f"Growth of the gap since baseline t = {baseline_time:.1f}\n"
                       "(dashed line = baseline)", fontsize=10)
    for c in range(3):
        ax[-1, c].set_xlabel("time to diagnosis")
    ax[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    ax[0, 1].legend(fontsize=8, frameon=False, loc="lower right")
 
    for a in ax.flat:
        a.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(labelbottom=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

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
 
 
def _estimable_slots(d, min_obs=5):
    """Slots with at least `min_obs` observations in BOTH classes (needed for C(split) * C(group))."""
    counts = d.groupby(["split", "group"]).size().unstack(fill_value=0)
    if counts.shape[1] < 2:
        return set()
    return set(counts.index[(counts >= min_obs).all(axis=1)])
 
 
def _heldout_loglik(res, test, matched_sets):
    """Log-likelihood of unseen subjects' observations under a fitted model, with the random
    effects integrated out exactly as in the model. Also returns the sum of squared errors of
    the population (fixed-effect) prediction."""
    from scipy.stats import multivariate_normal
 
    X = np.asarray(patsy.build_design_matrices([res.design_info], test)[0])
    mu = X @ res.fe_params.values
    y, t, s2 = test["y"].values, test["time"].values, res.scale
    if matched_sets:
        vc = dict(zip(res.model.exog_vc.names, res.vcomp))
        v_set, v_int, v_sl = float(np.asarray(res.cov_re)[0, 0]), vc["subj_int"], vc["subj_slope"]
        subj = test["subject"].to_numpy(dtype=object)
        units = test.groupby("set").indices
    else:
        G = np.asarray(res.cov_re)
        units = test.groupby("subject").indices
    ll = 0.0
    for idx in units.values():
        ti = t[idx]
        if matched_sets:
            same = subj[idx][:, None] == subj[idx][None, :]
            C = v_set + same * (v_int + v_sl * np.outer(ti, ti))
        else:
            Z = np.column_stack([np.ones(len(idx)), ti])
            C = Z @ G @ Z.T
        ll += multivariate_normal.logpdf(y[idx], mu[idx], C + s2 * np.eye(len(idx)))
    return ll, float(np.sum((y - mu) ** 2))

def compare_range(raw, train_intervals, eval_interval, columns=preset_columns, time_model="spline",
                  spline_df=4, model_split=True, matched_sets=False, logit=True, eps=1e-4,
                  n_folds=5, seed=0, baseline_quantile=0.05, n_sim=10000, min_slot_obs=5,
                  path="range_comparison.png"):
    """Train ONE model on several time intervals and compare them on the same evaluation interval.
 
    train_intervals  list of (lo, hi): train on observations with lo <= time < hi (None = open end)
    eval_interval    (t_min, t_max): every version is scored on observations in this interval
    time_model       "spline" or "quadratic";  model_split: add C(split) * C(group)
    matched_sets     True = subjects nested in matched sets (needs a "set" column)
 
    Scoring is cross-validated: participants (or whole matched sets) are split into `n_folds`
    folds; each version is fitted on the other folds (restricted to its training interval) and
    scored on the held-out participants' observations inside eval_interval. Higher held-out
    log-likelihood = better description of new people's data in that interval.
    The figure shows each version (fitted on all participants) over the evaluation interval.
    """
    # --- data -----------------------------------------------------------------
    COLUMNS = {v: k for k, v in columns.items()}
    df = raw.rename(columns=COLUMNS)[list(COLUMNS.values())].dropna()
    df["time"] = df["time"].astype(float)
    if logit:
        p = df["y"].clip(eps, 1 - eps)
        df["y"] = np.log(p / (1 - p))
    assert df["group"].nunique() == 2, "expected exactly two classes"
    if matched_sets:
        assert "set" in df.columns, 'add "set": "case_control_group" to `columns`'
        assert df.groupby("subject")["set"].nunique().max() == 1, "each subject must be in one set"
    if not model_split:
        df = df.drop(columns="split", errors="ignore")
 
    # fixed spline boundaries over the whole time axis, so a model trained on one interval
    # can be evaluated on another (inner knots still come from the training data)
    lo_b, hi_b = df["time"].min() - 0.01, df["time"].max() + 0.01
    if time_model == "spline":
        TT = f"cr(time, df={spline_df}, constraints='center', lower_bound={lo_b:.6f}, upper_bound={hi_b:.6f})"
    elif time_model == "quadratic":
        TT = "(center(time) + I(center(time)**2))"
    else:
        raise ValueError("time_model must be 'spline' or 'quadratic'")
    formula = f"y ~ {TT} * C(group)" + (" + C(split) * C(group)" if model_split else "")
    labels = [_interval_label(iv) for iv in train_intervals]
    eval_label = _interval_label(eval_interval)
 
    def restrict(d, iv):
        d = d[_in_interval(d, iv)]
        if model_split:
            ok = _estimable_slots(d, min_slot_obs)
            return d[d["split"].isin(ok)], ok
        return d, None
 
    # --- folds: whole participants (or whole matched sets), stratified by class --------------
    unit = "set" if matched_sets else "subject"
    unit_class = df.groupby(unit)["group"].max()
    rng = np.random.default_rng(seed)
    fold_of = {}
    for _, units in unit_class.groupby(unit_class):
        for i, u in enumerate(rng.permutation(units.index.values)):
            fold_of[u] = i % n_folds
    fold = df[unit].map(fold_of).values
 
    # --- cross-validated scores ----------------------------------------------------------------
    scores, n_dropped = [], 0
    for k in range(n_folds):
        train_all = df[fold != k]
        test = df[(fold == k) & _in_interval(df, eval_interval)]
        parts = [restrict(train_all, iv) for iv in train_intervals]
        if model_split:            # score every version on the SAME observations
            common = set.intersection(*[ok for _, ok in parts])
            n_dropped += int((~test["split"].isin(common)).sum())
            test = test[test["split"].isin(common)]
        for lab, (tr, _) in zip(labels, parts):
            res = _fit(formula, tr, matched_sets=matched_sets)
            ll, sse = _heldout_loglik(res, test, matched_sets)
            scores.append(dict(train=lab, fold=k, n_eval=len(test), ll=ll, sse=sse))
        print(f"fold {k + 1}/{n_folds} done")
    scores = pd.DataFrame(scores)
 
    per_fold = scores.pivot(index="fold", columns="train", values="ll")[labels]
    summary = scores.groupby("train", sort=False).agg(ll=("ll", "sum"), sse=("sse", "sum"),
                                                      n_eval=("n_eval", "sum"))
    summary["ll_per_obs"] = summary["ll"] / summary["n_eval"]
    summary["d_ll"] = summary["ll"] - summary["ll"].max()
    summary["rmse"] = np.sqrt(summary["sse"] / summary["n_eval"])
    summary["folds_best"] = per_fold.idxmax(axis=1).value_counts().reindex(labels, fill_value=0)
    best = summary["ll"].idxmax()
    summary["se_d_ll"] = [np.nan if lab == best else
                          np.std(per_fold[lab] - per_fold[best], ddof=1) * np.sqrt(n_folds)
                          for lab in summary.index]
 
    # --- fits on all participants, for the figure and the onsets -------------------------------
    fits = {}
    for lab, iv in zip(labels, train_intervals):
        tr, ok = restrict(df, iv)
        fits[lab] = (_fit(formula, tr, reml=True, matched_sets=matched_sets), ok, len(tr))
    ev = df[_in_interval(df, eval_interval)]
    if model_split:
        ev = ev[ev["split"].isin(set.intersection(*[ok for _, ok, _ in fits.values()]))]
    ref_split = ev["split"].mode()[0] if model_split else None
    baseline_time = ev["time"].quantile(baseline_quantile)
    curves = {}
    for lab, (res, _, n_tr) in fits.items():
        gap, _, groups = difference_curve(res, ev, n_sim=n_sim, model_split=model_split, ref_split=ref_split)
        growth, _, _ = difference_curve(res, ev, n_sim=n_sim, model_split=model_split,
                                        ref_split=ref_split, baseline_time=baseline_time)
        curves[lab] = {"gap": gap, "growth": growth}
        iv_gap, iv_growth = significant_intervals(gap), significant_intervals(growth)
        summary.loc[lab, "n_train"] = n_tr
        summary.loc[lab, "onset_gap"] = iv_gap[0][0] if iv_gap else np.nan
        summary.loc[lab, "onset_growth"] = iv_growth[0][0] if iv_growth else np.nan
 
    # --- report --------------------------------------------------------------------------------
    pd.set_option("display.width", 160)
    re_txt = "subject nested in matched set" if matched_sets else "subject"
    print(f"\nModel: {time_model}{' + slot' if model_split else ''}, random effects: {re_txt}")
    print(f"Evaluation interval {eval_label}: {summary['n_eval'].iloc[0]} held-out observations"
          f" over {n_folds} folds" + (f" ({n_dropped} dropped: slots not estimable in every "
                                       f"training interval)" if n_dropped else ""))
    print("\nHeld-out fit on the evaluation interval (higher ll = better; d_ll = difference to the best,"
          "\nse_d_ll = its cross-validation standard error; rmse of the population prediction)")
    cols_out = ["n_train", "ll", "ll_per_obs", "d_ll", "se_d_ll", "folds_best", "rmse"]
    print(summary[cols_out].round(3).to_string())
    print(f"\nOnsets within the evaluation interval (fits on all participants; growth baseline "
          f"t = {baseline_time:.2f})")
    print(summary[["onset_gap", "onset_growth"]].round(2).to_string())
    for lab, iv in zip(labels, train_intervals):
        lo, hi = iv
        if (lo is not None and (eval_interval[0] is None or eval_interval[0] < lo)) or \
           (hi is not None and (eval_interval[1] is None or eval_interval[1] > hi)):
            print(f"Note: {lab} does not cover the evaluation interval, so part of its "
                  f"evaluation is extrapolation.")
 
    if path:
        _plot_range(ev, fits, curves, groups, summary, train_intervals, labels, eval_interval,
                    baseline_time, ref_split, model_split, path)
        print("\nSaved plot:", path)
    return summary, scores, curves
 
 
def _plot_range(ev, fits, curves, groups, summary, train_intervals, labels, eval_interval,
                baseline_time, ref_split, model_split, path, n_bins=12):
    """One row per training interval; columns: trajectories | gap | growth, on the evaluation interval."""
    import matplotlib.pyplot as plt
 
    class_colors = dict(zip(groups, ["#2a78d6", "#eb6834"]))
    muted, band_sim, band_pw, sig, train_c = "#52514e", "#b9b8b2", "#1baf7a", "#e34948", "#6250d6"
    n = len(fits)
    fig, ax = plt.subplots(n, 3, figsize=(16, 3.0 * n + 0.8), sharex=True, sharey="col", squeeze=False)
    bins = np.quantile(ev["time"], np.linspace(0, 1, n_bins + 1))
    x_lo, x_hi = ev["time"].min(), ev["time"].max()
 
    for r, (lab, iv) in enumerate(zip(labels, train_intervals)):
        res, _, _ = fits[lab]
        row = summary.loc[lab]
        tgrid = curves[lab]["gap"]["time"].values
        for g in groups:
            col = class_colors[g]
            ax[r, 0].plot(tgrid, _population_curve(res, ev, tgrid, g, model_split), color=col, lw=2,
                          label=f"class {g}")
            sub = ev[ev["group"] == g]
            b = sub.groupby(pd.cut(sub["time"], bins, include_lowest=True), observed=True)
            ax[r, 0].plot(b["time"].mean(), b["y"].mean(), "o", color=col, mfc="white",
                          ms=6, mew=1.5, ls="none", zorder=5)
        ax[r, 0].set_ylabel(f"trained on {lab}\n\ny (logit)", fontsize=10)
        se = "" if np.isnan(row["se_d_ll"]) else f" ± {row['se_d_ll']:.1f}"
        ax[r, 0].text(0.02, 0.97, f"held-out d_ll {row['d_ll']:.1f}{se}   "
                                  f"best in {int(row['folds_best'])} folds\nrmse {row['rmse']:.3f}",
                      transform=ax[r, 0].transAxes, va="top", fontsize=8, color=muted)
 
        for c, (kind, onset_col) in enumerate([("gap", "onset_gap"), ("growth", "onset_growth")], 1):
            cv, a = curves[lab][kind], ax[r, c]
            a.fill_between(cv["time"], cv["sim_lo"], cv["sim_hi"], color=band_sim, alpha=0.5, lw=0,
                           label="simultaneous 95%")
            a.fill_between(cv["time"], cv["pw_lo"], cv["pw_hi"], color=band_pw, alpha=0.25, lw=0,
                           label="pointwise 95%")
            a.plot(cv["time"], cv["delta"], color="#0b0b0b", lw=2)
            for lo_, hi_ in significant_intervals(cv):
                a.axvspan(lo_, hi_, color=sig, alpha=0.08, lw=0)
            onset = row[onset_col]
            if np.isfinite(onset):
                a.plot(onset, np.interp(onset, cv["time"], cv["delta"]), "o", color=sig, ms=8,
                       mec="white", mew=2, zorder=6)
            a.text(0.98, 0.97, f"onset: {onset:.1f}" if np.isfinite(onset) else "no onset",
                   transform=a.transAxes, va="top", ha="right", fontsize=8, color=muted)
            a.axhline(0, color=muted, ls=":", lw=1)
            if kind == "growth":
                a.axvline(baseline_time, color=muted, lw=1, ls="--")
        # training interval, drawn as a bar under each panel
        lo, hi = iv
        for a in ax[r]:
            a.axvspan(x_lo if lo is None else max(lo, x_lo), x_hi if hi is None else min(hi, x_hi),
                      ymin=0, ymax=0.025, color=train_c, alpha=0.8, lw=0)
 
    ref_txt = f"\n(at reference slot {ref_split})" if model_split else ""
    ax[0, 0].set_title("Population trajectories\n(dots: raw binned means in the evaluation interval)",
                       fontsize=10)
    ax[0, 1].set_title(f"Gap {groups[1]} − {groups[0]}{ref_txt}", fontsize=10)
    ax[0, 2].set_title(f"Growth of the gap since t = {baseline_time:.1f}\n(dashed line = baseline)",
                       fontsize=10)
    for c in range(3):
        ax[-1, c].set_xlabel(f"time to diagnosis (evaluation interval {_interval_label(eval_interval)};"
                             f" purple bar = training interval)", fontsize=9)
    ax[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    ax[0, 1].legend(fontsize=8, frameon=False, loc="lower right")
    for a in ax.flat:
        a.set_xlim(x_lo, x_hi)
        a.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(labelbottom=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path