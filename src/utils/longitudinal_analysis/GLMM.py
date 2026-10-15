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


def fit_full_and_null(df, time_term):
    full = fit_mixed(f"y ~ {time_term} * C(group)", df)  # class-specific trajectories
    null = fit_mixed(f"y ~ {time_term}", df)             # one shared trajectory
    return full, null


# ----------------------------------------------------------------------------
# 2. Global test
# ----------------------------------------------------------------------------
def lrt(full, null):
    stat = 2 * (full.llf - null.llf)
    dof = len(full.fe_params) - len(null.fe_params)
    return stat, dof, stats.chi2.sf(stat, dof)


def permutation_lrt(df, time_term, n_perm=200, seed=0):
    """Shuffle class labels between SUBJECTS (never between observations)."""
    rng = np.random.default_rng(seed)
    full, null = fit_full_and_null(df, time_term)
    observed, _, _ = lrt(full, null)
    labels = df.groupby("subject")["group"].first()
    perm_stats = []
    for _ in range(n_perm):
        shuffled = pd.Series(rng.permutation(labels.values), index=labels.index)
        d = df.assign(group=df["subject"].map(shuffled))
        f, n = fit_full_and_null(d, time_term)
        perm_stats.append(2 * (f.llf - n.llf))
    p = (1 + np.sum(np.array(perm_stats) >= observed)) / (n_perm + 1)
    return observed, p


# ----------------------------------------------------------------------------
# 3. Difference curve with simultaneous confidence band
# ----------------------------------------------------------------------------
def difference_curve(result, df, n_grid=200, n_sim=10000, level=0.95, seed=0):
    """Delta(t) = mean(class 2) - mean(class 1), with pointwise and simultaneous bands.

    Simultaneous band: draw beta* ~ N(beta_hat, V), compute the max over t of
    |Delta*(t) - Delta_hat(t)| / se(t); its `level` quantile replaces 1.96.
    """
    groups = sorted(df["group"].unique())
    # restrict to the time range where BOTH classes have data
    lo = df.groupby("group")["time"].min().max()
    hi = df.groupby("group")["time"].max().min()
    grid = np.linspace(lo, hi, n_grid)

    design_info = result.design_info  # remembers spline knots from the fit
    X = {g: np.asarray(patsy.build_design_matrices(
            [design_info], pd.DataFrame({"time": grid, "group": g}))[0])
         for g in groups}
    L = X[groups[1]] - X[groups[0]]               # contrast matrix, one row per grid time

    beta = result.fe_params.values
    V = result.cov_params().loc[result.fe_params.index, result.fe_params.index].values

    delta = L @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", L, V, L))

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.zeros_like(beta), V, size=n_sim)
    max_abs_t = np.max(np.abs(draws @ L.T) / se, axis=1)
    crit_sim = np.quantile(max_abs_t, level)
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
def plot_results(df, full, curve, groups, profile=None, path="results.png"):
    import matplotlib.pyplot as plt

    ncols = 3 if profile is not None else 2
    fig, ax = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5))
    colors = dict(zip(groups, ["tab:blue", "tab:orange"]))

    design_info = full.design_info
    for g in groups:
        sub = df[df["group"] == g]
        for _, s in sub.groupby("subject"):
            ax[0].plot(s["time"], s["y"], color=colors[g], alpha=0.12, lw=0.8)
        Xg = np.asarray(patsy.build_design_matrices(
            [design_info], pd.DataFrame({"time": curve["time"], "group": g}))[0])
        ax[0].plot(curve["time"], Xg @ full.fe_params.values, color=colors[g], lw=3, label=g)
    ax[0].set(title="Population trajectories", xlabel="time", ylabel="y")
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

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path



preset_columns = {             # your column name -> name used internally
        "subject":"unique_id",
        "group": "true_label",
        "time": "case_dt",
        "y": "probability_1",
    }
def run_analysis(raw, columns=preset_columns, logit=True, eps=1e-4, path=None, spline_df=4,
                 run_permutation_test=False, n_perm=200, run_bootstrap=False, n_boot=200):
    '''logit = True            # model logit(probability_1) instead of the raw probability
    logit = 1e-4              # clip probabilities away from 0/1 before the logit
    '''
    SPLINE_DF = spline_df  # flexibility of the time trend; 3-5 is typical, check sensitivity
    TIME_TERM = f"cr(time, df={SPLINE_DF}, constraints='center')"  # natural cubic regression spline

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
    full, null = fit_full_and_null(df, TIME_TERM)
    stat, dof, p = lrt(full, null)
    print(f"Global LRT (class affects level or shape): chi2={stat:.2f}, df={dof}, p={p:.3g}")

    # Step 3: difference curve (refit with REML for less biased variance components)
    full_reml = fit_mixed(f"y ~ {TIME_TERM} * C(group)", df, reml=True)
    curve, crit, groups = difference_curve(full_reml, df)
    print(f"Simultaneous critical value: {crit:.2f} (vs 1.96 pointwise)")
    intervals = significant_intervals(curve)
    if intervals:
        print("Simultaneous band excludes 0 on: "
              + ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in intervals))
        print(f"-> difference DETECTABLE from t ≈ {intervals[0][0]:.2f}")
    else:
        print("No time interval where the difference is significant.")

    # Step 4: change-point estimate of the onset
    tau_hat, tau_ci, profile = changepoint_profile(df, time_term=TIME_TERM)
    print(f"\nChange-point onset: tau = {tau_hat:.2f}, "
          f"approx. 95% CI [{tau_ci[0]:.2f}, {tau_ci[1]:.2f}]")

    if path:
        print("\nSaved plot:", plot_results(df, full_reml, curve, groups, profile, path=path))

    # Step 5 (slow; uncomment for robustness checks)
    if run_permutation_test:
        obs, p_perm = permutation_lrt(df, TIME_TERM,n_perm=n_perm); print(f"Permutation p = {p_perm:.3g}")
    if run_bootstrap:
        ci_boot, _ = cluster_bootstrap_tau(df, TIME_TERM,n_boot=n_boot); print(f"Bootstrap tau CI: {ci_boot}")
