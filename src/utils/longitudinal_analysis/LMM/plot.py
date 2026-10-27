import numpy as np
from .helpers import _population_curve, _slot_weights, significant_intervals,_interval_label
import pandas as pd

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