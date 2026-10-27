import numpy as np
import pandas as pd
from scipy import stats
from .fit import fit_mixed

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
