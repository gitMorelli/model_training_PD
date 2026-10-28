import numpy as np
import pandas as pd
from scipy import stats
from .fit import fit_mixed
from .helpers import adjustment_spec

BASELINES = {"linear": "time", "quadratic": "time + I(time**2)"}
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
 
def class_adjustment(adjust, n_form="log"):
    """Adjustment terms without the class interaction (there is only one class in the fit):
    "slot" -> " + C(split)", "n_quest_*" -> " + np.log(n_quest)", None -> ""."""
    terms, _, _ = adjustment_spec(adjust, n_form)
    return terms.replace(" * C(group)", "")
 
 
def changepoint_profile_class(df, cls, baseline="linear", hinge_power=1, adj="", tau_grid=None,
                              level=0.95, min_side_obs=10):
    """y = baseline(t) + delta * (t - tau)_+^p [+ adj] + subject random intercept and slope,
    fitted on the subjects of ONE class.
 
    The reference is the class's own trend before tau, extended forward ("terminal decline"
    style), so no other class and no pseudo-index date is needed for the cases. Run it on the
    controls too: a bend there at a similar tau is a time effect shared by everyone.
 
    baseline      "linear" (default), "quadratic", or any patsy term in `time`. Keep it rigid:
                  with one class, a flexible baseline such as a spline can absorb the bend
                  itself, and tau is no longer identifiable.
    hinge_power   1 = slope changes at tau (linear divergence); 2 = accelerating after tau
    adj           extra terms without C(group), e.g. class_adjustment("slot"). n_quest rises
                  with time inside every subject, so adjusting for it can absorb the bend too
    tau_grid      default: 40 points between the 5th and 90th percentile of the class's times
    min_side_obs  skip taus with fewer observations than this on either side
 
    Returns dict: tau, ci (approximate profile-likelihood interval), delta (bend at tau_hat),
    d_ll (best hinge fit minus no-bend fit: descriptive, NOT chi2-distributed because tau is
    searched over), n_subjects, profile DataFrame (tau, llf, delta).
    """
    d = df[df["group"] == cls]
    base = BASELINES.get(baseline, baseline)
    t = d["time"].values
    if tau_grid is None:
        tau_grid = np.linspace(*np.quantile(t, [0.05, 0.9]), 40)
    null = fit_mixed(f"y ~ {base}{adj}", d)
    llf = np.full(len(tau_grid), np.nan)
    delta = np.full(len(tau_grid), np.nan)
    for i, tau in enumerate(tau_grid):
        after = t > tau
        if after.sum() < min_side_obs or (~after).sum() < min_side_obs:
            continue
        h = np.maximum(t - tau, 0) ** hinge_power
        r = fit_mixed(f"y ~ {base}{adj} + hinge", d.assign(hinge=h))
        llf[i], delta[i] = r.llf, r.fe_params["hinge"]
    if not np.isfinite(llf).any():
        raise ValueError(f"class {cls}: no tau in the grid has enough observations on both sides")
    j = int(np.nanargmax(llf))
    inside = 2 * (np.nanmax(llf) - llf) <= stats.chi2.ppf(level, 1)   # NaN -> False
    return dict(cls=cls, tau=tau_grid[j], ci=(tau_grid[inside].min(), tau_grid[inside].max()),
                delta=delta[j], d_ll=np.nanmax(llf) - null.llf, n_subjects=d["subject"].nunique(),
                profile=pd.DataFrame({"tau": tau_grid, "llf": llf, "delta": delta}))
 
 
def changepoint_by_class(df, tau_grid=None, **kw):
    """Run changepoint_profile_class on every class with the SAME tau grid
    (default: 5th-90th percentile of all times), so the profiles can be compared."""
    if tau_grid is None:
        tau_grid = np.linspace(*np.quantile(df["time"], [0.05, 0.9]), 40)
    return {g: changepoint_profile_class(df, g, tau_grid=tau_grid, **kw)
            for g in sorted(df["group"].unique())}
 
 
def cluster_bootstrap_tau_class(df, cls, n_boot=200, seed=0, tau_grid=None, **kw):
    """Percentile CI for one class's tau: resample that class's SUBJECTS with replacement
    (duplicates get new ids so they stay separate subjects) and re-estimate tau."""
    rng = np.random.default_rng(seed)
    d = df[df["group"] == cls]
    if tau_grid is None:
        tau_grid = np.linspace(*np.quantile(d["time"], [0.05, 0.9]), 40)
    by_subj = {s: g for s, g in d.groupby("subject")}
    subjects = np.array(list(by_subj))
    taus = []
    for b in range(n_boot):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        boot = pd.concat([by_subj[s].assign(subject=f"{s}_b{j}") for j, s in enumerate(pick)])
        try:
            taus.append(changepoint_profile_class(boot, cls, tau_grid=tau_grid, **kw)["tau"])
        except ValueError:
            continue
    taus = np.array(taus)
    return np.percentile(taus, [2.5, 97.5]), taus
 
 
def plot_changepoint_by_class(results, path="changepoint_by_class.png", level=0.95):
    """Left: profile log-likelihood of tau per class, relative to its own maximum (dashed line
    = approximate 95% cut-off). Right: size of the bend delta at each tau.
    Peaked case curve + flat control curve = case-specific bend; both peaked at a similar
    tau = time effect shared by everyone."""
    import matplotlib.pyplot as plt
    colors = ["#2a78d6", "#eb6834"]
    muted = "#52514e"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for (g, r), c in zip(results.items(), colors):
        pr = r["profile"]
        ax[0].plot(pr["tau"], pr["llf"] - pr["llf"].max(), "o-", ms=3, color=c,
                   label=f"class {g}: τ = {r['tau']:.1f} [{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]")
        ax[0].axvline(r["tau"], color=c, lw=1, ls=":")
        ax[1].plot(pr["tau"], pr["delta"], "o-", ms=3, color=c, label=f"class {g}")
    ax[0].axhline(-stats.chi2.ppf(level, 1) / 2, color=muted, lw=1, ls="--")
    ax[0].set(xlabel="τ (time to diagnosis)", ylabel="log-likelihood − max",
              title="Within-class change point: profile likelihood")
    ax[1].axhline(0, color=muted, lw=1, ls=":")
    ax[1].set(xlabel="τ (time to diagnosis)", ylabel="bend δ at τ",
              title="Size of the bend (change in slope after τ)")
    for a in ax:
        a.legend(fontsize=8, frameon=False)
        a.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path