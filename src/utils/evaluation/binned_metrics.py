"""Binned discrimination metrics with pointwise and simultaneous confidence bands.

Metrics per bin of the time axis:
    mean_0, mean_1   mean signal in class 0 / class 1
    difference       mean_1 - mean_0
    roc_auc          unmatched AUC (all case rows vs all control rows in the bin)
    matched_cindex   concordance restricted to case/control pairs of the SAME group
    lr_plus          sens / (1 - spec) at `threshold`
    lr_minus         (1 - sens) / spec at `threshold`

Uncertainty comes from a cluster bootstrap that resamples whole case-control
groups (all their rows, in every bin). This respects the matching and keeps
the between-bin correlation, which is what the simultaneous band needs.

    pointwise     estimate +/- z_{1-alpha/2} * SE_boot          (per bin)
    simultaneous  estimate +/- c * SE_boot, c = (1-alpha) quantile of
                  max over bins |boot - estimate| / SE_boot      (sup-t band)

The simultaneous band covers the whole curve of ONE metric with probability
1-alpha; it is not adjusted across metrics. LR+ / LR- are handled on the log
scale (bands are back-transformed, `se` is the SE of log LR).
"""
import os
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALL_METRICS = ("mean_0", "mean_1", "difference", "roc_auc",
               "matched_cindex", "lr_plus", "lr_minus")
_LOG_METRICS = ("lr_plus", "lr_minus")          # CI built on log scale
_UNIT_METRICS = ("roc_auc", "matched_cindex")   # bands clipped to [0, 1]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def _auc_prep(score, y, g, b, n_bins):
    """Per bin: rows sorted by score plus tie-block boundaries (done once)."""
    prep = []
    for k in range(n_bins):
        idx = np.flatnonzero(b == k)
        if idx.size == 0 or y[idx].min() == y[idx].max():
            prep.append(None)
            continue
        idx = idx[np.argsort(score[idx], kind="mergesort")]
        s = score[idx]
        new = np.r_[True, s[1:] != s[:-1]]
        first = np.maximum.accumulate(np.where(new, np.arange(s.size), 0))
        end = np.r_[np.flatnonzero(new)[1:], s.size]          # block end (exclusive)
        last_excl = end[np.cumsum(new) - 1]
        prep.append((g[idx], y[idx].astype(float), first, last_excl))
    return prep


def _auc_weighted(M, prep, max_cells=2e7):
    """Weighted Mann-Whitney AUC per bin. M: (R, G) group multiplicities."""
    R = M.shape[0]
    out = np.full((R, len(prep)), np.nan)
    for k, p in enumerate(prep):
        if p is None:
            continue
        gk, yk, first, last_excl = p
        step = max(1, int(max_cells // gk.size))
        for r0 in range(0, R, step):
            W = M[r0:r0 + step][:, gk].astype(float)
            wpos, wneg = W * yk, W * (1 - yk)
            c = np.concatenate([np.zeros((W.shape[0], 1)), np.cumsum(wneg, axis=1)], axis=1)
            below, ties = c[:, first], c[:, last_excl] - c[:, first]
            num = (wpos * (below + 0.5 * ties)).sum(axis=1)
            out[r0:r0 + step, k] = _safe_div(num, wpos.sum(axis=1) * wneg.sum(axis=1))
    return out


def _all_metrics(M, A, prep, metrics, lr_continuity):
    """All requested metrics for each row of M (R, G). Returns {metric: (R, K)}.
    LR metrics are returned on the log scale."""
    T = {k: M @ v for k, v in A.items()}                      # (R, K) totals
    out = {}
    m0, m1 = _safe_div(T["sum0"], T["n0"]), _safe_div(T["sum1"], T["n1"])
    if "mean_0" in metrics:
        out["mean_0"] = m0
    if "mean_1" in metrics:
        out["mean_1"] = m1
    if "difference" in metrics:
        out["difference"] = m1 - m0
    if "roc_auc" in metrics:
        out["roc_auc"] = _auc_weighted(M, prep)
    if "matched_cindex" in metrics:
        out["matched_cindex"] = _safe_div(T["conc"], T["pairs"])
    if "lr_plus" in metrics or "lr_minus" in metrics:
        c = lr_continuity
        both = (T["n0"] > 0) & (T["n1"] > 0)
        sens = (T["tp"] + c) / (T["n1"] + 2 * c)
        fpr = (T["fp"] + c) / (T["n0"] + 2 * c)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(both, np.log(sens) - np.log(fpr), np.nan)
            lm = np.where(both, np.log1p(-sens) - np.log1p(-fpr), np.nan)
        lp[~np.isfinite(lp)] = np.nan
        lm[~np.isfinite(lm)] = np.nan
        if "lr_plus" in metrics:
            out["lr_plus"] = lp
        if "lr_minus" in metrics:
            out["lr_minus"] = lm
    return out


def _bands(est, boot, alpha):
    """Pointwise (normal, bootstrap SE) and sup-t simultaneous bands."""
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        se = np.nanstd(boot, axis=0, ddof=1)
    ok = np.isfinite(est) & np.isfinite(se)
    use = ok & (se > 0)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    crit = np.nan
    if use.any():
        t = np.abs(boot[:, use] - est[use]) / se[use]
        t = t[~np.isnan(t).all(axis=1)]
        tmax = np.nanmax(t, axis=1)
        crit = max(float(np.quantile(tmax, 1 - alpha)), z)
    se = np.where(ok, se, np.nan)
    c = crit if np.isfinite(crit) else z
    return se, est - z * se, est + z * se, est - c * se, est + c * se, crit


# --------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------
def binned_metrics(df, n_steps=20, step_is_years=False,
                   signal_column="probability_1", label_column="true_label",
                   time_column="case_dt", group_column="group",
                   metrics=ALL_METRICS, threshold=0.5, min_count=1,
                   alpha=0.05, n_boot=2000, lr_continuity=0.5, seed=0,
                   verbose=True):
    """Per-bin metrics with pointwise and simultaneous (1-alpha) bands.

    Parameters
    ----------
    group_column : case-control set id. Groups are the bootstrap unit and
        define the pairs of `matched_cindex`. Pass None to treat every row as
        independent (matched_cindex is then dropped).
    threshold : signal >= threshold is a positive call (LR+ / LR-).
    min_count : a bin needs at least this many rows of each class involved in
        a metric, otherwise that metric is NaN there and the bin is left out
        of its simultaneous band.
    lr_continuity : added to the 2x2 cells (Haldane correction) so LRs stay
        finite when a cell is empty. Set 0 to disable.

    Returns
    -------
    Long DataFrame, one row per (metric, bin): estimate, se, ci_low/ci_high
    (pointwise), sim_low/sim_high (simultaneous), crit (sup-t multiplier),
    plus bin geometry and counts. `df.attrs` holds alpha, n_boot, threshold.
    """
    metrics = tuple(metrics)
    unknown = set(metrics) - set(ALL_METRICS)
    if unknown:
        raise ValueError(f"unknown metrics: {sorted(unknown)}")
    if group_column is None:
        metrics = tuple(m for m in metrics if m != "matched_cindex")

    cols = [signal_column, time_column, label_column] + ([group_column] if group_column else [])
    d = df.dropna(subset=cols).copy()
    if d.empty:
        raise ValueError(f"no rows with {', '.join(cols)}")
    if not set(np.unique(d[label_column])) <= {0, 1}:
        raise ValueError(f"{label_column} must be 0/1")

    # ---- binning (same rule as plot_probability_trajectories) ----
    lo, hi = d[time_column].min(), d[time_column].max()
    if step_is_years:
        n_steps = max(1, int(np.ceil((hi - lo) / n_steps)))
    edges = np.linspace(lo, hi, n_steps + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    b = np.clip(np.digitize(d[time_column], edges) - 1, 0, n_steps - 1)
    if verbose:
        print(f"{n_steps} bins, {(hi - lo) / n_steps:.2f} wide, from {lo:.2f} to {hi:.2f}")

    y = d[label_column].to_numpy().astype(int)
    s = d[signal_column].to_numpy().astype(float)
    g = pd.factorize(d[group_column])[0] if group_column else np.arange(len(d))
    G, K = int(g.max()) + 1, n_steps

    # ---- per (group, bin) sufficient statistics: everything but AUC is additive ----
    def acc(values):
        out = np.zeros((G, K))
        np.add.at(out, (g, b), values)
        return out

    pred = s >= threshold
    A = {"n0": acc(y == 0), "n1": acc(y == 1),
         "sum0": acc(s * (y == 0)), "sum1": acc(s * (y == 1)),
         "tp": acc(pred & (y == 1)), "fp": acc(pred & (y == 0)),
         "conc": np.zeros((G, K)), "pairs": np.zeros((G, K))}

    if "matched_cindex" in metrics:
        t = pd.DataFrame({"g": g, "b": b, "s": s, "y": y})
        pr = t[t.y == 1].merge(t[t.y == 0], on=["g", "b"], suffixes=("_1", "_0"))
        if len(pr):
            c = (pr.s_1 > pr.s_0) + 0.5 * (pr.s_1 == pr.s_0)
            np.add.at(A["conc"], (pr.g.to_numpy(), pr.b.to_numpy()), c.to_numpy())
            np.add.at(A["pairs"], (pr.g.to_numpy(), pr.b.to_numpy()), 1.0)

    prep = _auc_prep(s, y, g, b, K) if "roc_auc" in metrics else None

    # ---- point estimates ----
    est = {m: v[0] for m, v in
           _all_metrics(np.ones((1, G)), A, prep, metrics, lr_continuity).items()}
    n0, n1 = A["n0"].sum(0), A["n1"].sum(0)
    n_pairs = A["pairs"].sum(0)
    n_sets = (A["pairs"] > 0).sum(0)
    need = {"mean_0": n0 >= min_count, "mean_1": n1 >= min_count}
    both = (n0 >= min_count) & (n1 >= min_count)
    for m in metrics:
        est[m] = np.where(need.get(m, both), est[m], np.nan)

    # ---- cluster bootstrap over groups ----
    rng = np.random.default_rng(seed)
    boot = {m: np.empty((n_boot, K)) for m in metrics}
    step = max(1, int(2e7 // G))
    for r0 in range(0, n_boot, step):
        R = min(step, n_boot - r0)
        M = rng.multinomial(G, np.full(G, 1 / G), size=R).astype(np.float32)
        for m, v in _all_metrics(M, A, prep, metrics, lr_continuity).items():
            boot[m][r0:r0 + R] = v

    # ---- assemble ----
    rows = []
    for m in metrics:
        se, pl, ph, sl, sh, crit = _bands(est[m], boot[m], alpha)
        e = est[m]
        if m in _LOG_METRICS:
            e, pl, ph, sl, sh = (np.exp(a) for a in (e, pl, ph, sl, sh))
        if m in _UNIT_METRICS:
            pl, ph, sl, sh = (np.clip(a, 0, 1) for a in (pl, ph, sl, sh))
        rows.append(pd.DataFrame({
            "metric": m, "_bin": np.arange(K), "center": centers,
            "bin_low": edges[:-1], "bin_high": edges[1:],
            "n_0": n0.astype(int), "n_1": n1.astype(int),
            "n_matched_sets": n_sets.astype(int), "n_matched_pairs": n_pairs.astype(int),
            "estimate": e, "se": se, "ci_low": pl, "ci_high": ph,
            "sim_low": sl, "sim_high": sh, "crit": crit,
        }))
    res = pd.concat(rows, ignore_index=True)
    res = res[res["estimate"].notna()].reset_index(drop=True)
    res.attrs.update(alpha=alpha, n_boot=n_boot, threshold=threshold, n_groups=G,
                     n_steps=n_steps, bin_width=float((hi - lo) / n_steps),
                     signal_column=signal_column, time_column=time_column)
    return res


# --------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------
_PANELS = {  # panel -> (metrics drawn, y label, reference line, log y)
    "means":          (("mean_0", "mean_1"), "mean {signal}", None, False),
    "difference":     (("difference",), "Δ mean ({pos} − {neg})", 0.0, False),
    "roc_auc":        (("roc_auc",), "ROC AUC", 0.5, False),
    "matched_cindex": (("matched_cindex",), "matched C-index", 0.5, False),
    "lr_plus":        (("lr_plus",), "LR+ (threshold {thr:g})", 1.0, True),
    "lr_minus":       (("lr_minus",), "LR− (threshold {thr:g})", 1.0, True),
}
_COLORS = {"mean_0": "#3B8BD4", "mean_1": "#D85A30", "difference": "#5B4B8A",
           "roc_auc": "#2A8C6E", "matched_cindex": "#B8860B",
           "lr_plus": "#C0408A", "lr_minus": "#4A6FA5"}


def plot_binned_metrics(res, class_names=("negative", "positive"), ncols=2,
                        axes=None, save_path=None, save_name="binned_metrics",
                        y_pad_frac=0.04):
    """One panel per metric: mean line, pointwise CI (dark band + error bars)
    and simultaneous band (light band, dashed edge). Returns the axes array."""
    a = res.attrs
    level = 100 * (1 - a.get("alpha", 0.05))
    have = set(res["metric"])
    panels = [p for p, spec in _PANELS.items() if have & set(spec[0])]
    if axes is None:
        nrows = int(np.ceil(len(panels) / ncols))
        _, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.8 * nrows),
                               sharex=True, squeeze=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, panel in zip(axes, panels):
        mets, ylab, ref, logy = _PANELS[panel]
        lows, highs = [], []
        for m in mets:
            r = res[res["metric"] == m].sort_values("_bin")
            if r.empty:
                continue
            x, e, color = r["center"].to_numpy(), r["estimate"].to_numpy(), _COLORS[m]
            label = {"mean_0": f"true_label = 0 ({class_names[0]})",
                     "mean_1": f"true_label = 1 ({class_names[1]})"}.get(m)
            ax.fill_between(x, r["sim_low"], r["sim_high"], color=color, alpha=0.12,
                            edgecolor=color, linewidth=1, linestyle="--", zorder=1)
            ax.fill_between(x, r["ci_low"], r["ci_high"], color=color, alpha=0.28,
                            linewidth=0, zorder=2)
            ax.errorbar(x, e, yerr=[e - r["ci_low"], r["ci_high"] - e], fmt="o",
                        color=color, elinewidth=1.2, capsize=2.5, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                        label=label)
            ax.plot(x, e, color=color, lw=1.8, zorder=3)
            lows.append(r["sim_low"].to_numpy())
            highs.append(r["sim_high"].to_numpy())
        if ref is not None:
            ax.axhline(ref, color="gray", lw=0.7, ls="--", alpha=0.7)
        if logy:
            ax.set_yscale("log")
        if lows:
            y_lo, y_hi = np.nanmin(np.concatenate(lows)), np.nanmax(np.concatenate(highs))
            if logy:
                f = (y_hi / y_lo) ** y_pad_frac
                ax.set_ylim(y_lo / f, y_hi * f)
            else:
                pad = y_pad_frac * ((y_hi - y_lo) or 1)
                ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_ylabel(ylab.format(signal=a.get("signal_column", "signal"),
                                  pos=class_names[1], neg=class_names[0],
                                  thr=a.get("threshold", 0.5)))
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        if panel == "means":
            ax.legend(loc="best", frameon=False)

    for ax in axes[len(panels):]:
        ax.set_visible(False)
    xlabel = (f"{a.get('time_column', 'time')} (binned), {a.get('n_steps', '')} steps, "
              f"{a.get('bin_width', float('nan')):.2f} wide")
    for ax in axes[:len(panels)]:
        ax.set_xlabel(xlabel)
        ax.tick_params(labelbottom=True)
    fig = axes[0].figure
    fig.suptitle(f"Dark band / bars: pointwise {level:g}% CI   ·   "
                 f"light dashed band: simultaneous {level:g}% band "
                 f"(group bootstrap, B={a.get('n_boot', '?')})", fontsize=10)
    fig.tight_layout()

    if save_path:
        path = os.path.join(save_path, f"{save_name}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved binned metrics plot to {path}")
    return axes