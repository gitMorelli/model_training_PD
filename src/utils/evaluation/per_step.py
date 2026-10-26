from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.metrics import roc_curve, roc_auc_score

def plot_probability_trajectories(df, n_steps=20, ax=None,
                                  class_names=("negative", "positive"),
                                  min_count=1, show_points=True,
                                  point_alpha=0.15, save_path=None,
                                  save_name="probability_trajectories", step_is_years=False,
                                  signal_column="probability_1",
                                  plot_difference=False, diff_ax=None,
                                  diff_save_name=None, y_pad_frac=0.02):
    """Average probability_1 across subjects on a binned case_dt axis.
    Rows are binned to N equal-width steps between min and max case_dt.
    Each class (true_label 0 / 1) gets a mean curve with a ±SEM band.
    Bins with fewer than `min_count` subjects are dropped from that class.
 
    The y-axis of each plot spans the lowest to the highest edge of the
    ±SEM bands drawn on it, padded by `y_pad_frac` of that span.
 
    If `plot_difference` is True, a second figure shows the difference
    mean(class 1) - mean(class 0) per bin, with error
    SEM_diff = sqrt(SEM_0^2 + SEM_1^2) (independent groups). Only bins where
    both classes survive the `min_count` filter are included. The difference
    plot is saved as `{diff_save_name}.png` (default `{save_name}_difference`)
    when `save_path` is given.
 
    Returns `stats`, or `(stats, diff)` when `plot_difference` is True.
    """
    def _band_ylim(ax, lows, highs, pad_frac):
        """Set y-limits to the lowest/highest band edge, padded by pad_frac of the span."""
        if not lows:
            return
        y_lo = float(np.nanmin(np.concatenate(lows)))
        y_hi = float(np.nanmax(np.concatenate(highs)))
        pad = pad_frac * ((y_hi - y_lo) or 1)
        ax.set_ylim(y_lo - pad, y_hi + pad)

    d = df.dropna(subset=(signal_column, "case_dt", "true_label")).copy()
    if d.empty:
        raise ValueError(f"no rows with {signal_column}, case_dt, and true_label")
 
    lo, hi = d["case_dt"].min(), d["case_dt"].max()
    if step_is_years:
        delta = hi - lo
        # number of equal-width bins so each is as close as possible to n_steps years
        # -> n_steps=2y with delta=6y -> 3 bins; delta=5y -> ceil(5/2)=3 bins
        n_steps = int(np.ceil(delta / n_steps))
        print(f"step_is_years=True, delta={delta}, n_steps={n_steps}")
    width_bin_years = (hi - lo) / n_steps
    print("Bin is {:.2f} years wide, from {:.2f} to {:.2f}".format(width_bin_years, lo, hi))
 
    edges = np.linspace(lo, hi, n_steps + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    d["_bin"] = np.clip(np.digitize(d["case_dt"], edges) - 1, 0, n_steps - 1)
 
    grp = d.groupby(["true_label", "_bin"])[signal_column]
    stats = grp.agg(mean="mean", std="std", n="count").reset_index()
    stats["sem"] = stats["std"] / np.sqrt(stats["n"])
    stats = stats[stats["n"] >= min_count]
 
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
 
    palette = {0: "#3B8BD4", 1: "#D85A30"}
    band_lows, band_highs = [], []
    for label, name in zip((0, 1), class_names):
        s = stats[stats["true_label"] == label].sort_values("_bin")
        if s.empty:
            continue
        x = centers[s["_bin"].to_numpy()]
        m = s["mean"].to_numpy()
        e = s["sem"].fillna(0).to_numpy()
        color = palette[label]
        band_lows.append(m - e)
        band_highs.append(m + e)
 
        # raw rows behind the curve — jittered horizontally within each bin
        if show_points:
            raw = d[d["true_label"] == label]
            if not raw.empty:
                bin_w = edges[1] - edges[0]
                rng = np.random.default_rng(label)
                jitter = rng.uniform(-0.3, 0.3, len(raw)) * bin_w
                ax.scatter(centers[raw["_bin"].to_numpy()] + jitter,
                           raw[signal_column], s=8, color=color,
                           alpha=point_alpha, linewidths=0, zorder=1)
 
        # ±SEM band with a visible edge, then the mean line, then the markers
        ax.fill_between(x, m - e, m + e, color=color, alpha=0.25,
                        edgecolor=color, linewidth=1.2, zorder=2)
        ax.errorbar(x, m, yerr=e, fmt="o", color=color, ecolor=color,
                    elinewidth=1.4, capsize=3, markersize=6,
                    markeredgecolor="white", markeredgewidth=1,
                    label=f"true_label = {label} ({name})", zorder=3)
        ax.plot(x, m, color=color, lw=1.8, zorder=3)
 
    xlabel = f"case_dt (binned), {n_steps} steps, {width_bin_years:.2f} years wide"
    ax.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(signal_column)
    _band_ylim(ax, band_lows, band_highs, y_pad_frac)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="best", frameon=False)
 
    if save_path:
        fig_path = os.path.join(save_path, f"{save_name}.png")
        # save the figure that owns `ax`, not whatever figure happens to be current
        ax.figure.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved probability trajectories plot to {fig_path}")
 
    if not plot_difference:
        return stats
 
    # ---------------- difference curve: class 1 − class 0 ----------------
    s0 = stats[stats["true_label"] == 0].set_index("_bin")
    s1 = stats[stats["true_label"] == 1].set_index("_bin")
    common = s0.index.intersection(s1.index).sort_values().to_numpy()
 
    diff = pd.DataFrame({
        "_bin": common,
        "center": centers[common],
        "mean_0": s0.loc[common, "mean"].to_numpy(),
        "mean_1": s1.loc[common, "mean"].to_numpy(),
        "sem_0": s0.loc[common, "sem"].to_numpy(),
        "sem_1": s1.loc[common, "sem"].to_numpy(),
        "n_0": s0.loc[common, "n"].to_numpy(),
        "n_1": s1.loc[common, "n"].to_numpy(),
    })
    diff["diff"] = diff["mean_1"] - diff["mean_0"]
    # independent groups -> variances add; NaN SEM (n=1) treated as 0, as in the main plot
    diff["sem"] = np.sqrt(diff["sem_0"].fillna(0) ** 2 + diff["sem_1"].fillna(0) ** 2)
 
    if diff.empty:
        print("No bins contain both classes; difference plot skipped.")
        return stats, diff
 
    if diff_ax is None:
        _, diff_ax = plt.subplots(figsize=(9, 5))
 
    x = diff["center"].to_numpy()
    m = diff["diff"].to_numpy()
    e = diff["sem"].to_numpy()
    color = "#5B4B8A"
 
    diff_ax.fill_between(x, m - e, m + e, color=color, alpha=0.25,
                         edgecolor=color, linewidth=1.2, zorder=2)
    diff_ax.errorbar(x, m, yerr=e, fmt="o", color=color, ecolor=color,
                     elinewidth=1.4, capsize=3, markersize=6,
                     markeredgecolor="white", markeredgewidth=1,
                     label=f"{class_names[1]} − {class_names[0]} (±SEM)", zorder=3)
    diff_ax.plot(x, m, color=color, lw=1.8, zorder=3)
 
    diff_ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.6)
    diff_ax.set_xlabel(xlabel)
    diff_ax.set_ylabel(f"Δ {signal_column} ({class_names[1]} − {class_names[0]})")
    diff_ax.set_xlim(ax.get_xlim())  # align x-axis with the main plot
    _band_ylim(diff_ax, [m - e], [m + e], y_pad_frac)
    diff_ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    diff_ax.legend(loc="best", frameon=False)
 
    if save_path:
        name = diff_save_name or f"{save_name}_difference"
        diff_path = os.path.join(save_path, f"{name}.png")
        diff_ax.figure.savefig(diff_path, dpi=300, bbox_inches="tight")
        print(f"Saved difference plot to {diff_path}")
 
    return stats, diff
 
def roc_auc_over_time(
    df,
    step=1.0,
    window=1.0,
    id_col="unique_id",
    time_col="case_dt",
    pred_col="probability_1",
    true_col="true_value",
    percentiles=(5, 95),
    min_ids=20,
    plot=True,
    axes=None,
    axes_window=None,
    save_path=None,
    save_name="roc_auc_over_time.png",
    save_name_window=None,
    dpi=300,
    cmap="viridis",
    fontsize=8,
):
    """
    ROC-AUC as a function of time relative to diagnosis, in two variants.
 
    Cumulative: for each t, every id contributes its LAST row with
        case_dt <= t (ids with no row at or before t are dropped).
    Windowed: same selection, but only ids whose last row falls in
        (t - window, t] are kept, i.e. ids with a recent prediction.
 
    Parameters
    ----------
    df : DataFrame with id, time, score and true-value columns.
    step : float, spacing of the time grid (years).
    window : float, width of the (t - window, t] interval for the windowed
        variant. Set to None to skip it.
    pred_col : score column (probability of class 1 recommended).
    percentiles : (low, high) percentiles of case_dt used to define the range.
        The bounds are snapped outward to multiples of `step`.
    min_ids : minimum number of ids required to report an AUC at a given t.
    plot : whether to draw the figures.
    axes, axes_window : optional pairs of Axes (auc_ax, roc_ax) for the
        cumulative and windowed figures.
    save_path : optional directory where the figures are saved (created if
        missing). If given, figures are drawn even when plot=False.
    save_name : file name of the cumulative figure; extension sets the format.
    save_name_window : file name of the windowed figure. Defaults to
        save_name with "_window<window>" appended before the extension.
    dpi : resolution of the saved figures.
    cmap : colormap used to color the ROC curves by t.
    fontsize : base font size for ticks, labels and legends.
 
    Returns
    -------
    (res, res_window): DataFrames with columns t, roc_auc, n_ids, n_pos,
    n_neg, ordered from the latest t down to the earliest. res_window is
    None when window is None.
    """

    # Helpers #
    def _compute_auc_over_grid(data, grid, id_col, time_col, pred_col, true_col,
                           min_ids, window=None):
        """
        For each t in grid take, per id, the last row with case_dt <= t.
        If window is given, keep only ids whose last row has case_dt > t - window.
        """
        times = data[time_col].to_numpy()
        rows, curves = [], {}
        for t in grid:
            last = data[times <= t + 1e-9].drop_duplicates(subset=id_col, keep="last") #keep the last row per subject
            if window is not None:
                last = last[last[time_col] > t - window + 1e-9]
    
            y_true = last[true_col].to_numpy()
            y_score = last[pred_col].to_numpy()
            n_pos = int((y_true == 1).sum())
            n_neg = int((y_true == 0).sum())
    
            if len(last) >= min_ids and n_pos > 0 and n_neg > 0:
                auc = roc_auc_score(y_true, y_score)
                fpr, tpr, _ = roc_curve(y_true, y_score)
                curves[t] = (fpr, tpr, auc)
            else:
                auc = np.nan
    
            rows.append({"t": t, "roc_auc": auc, "n_ids": len(last),
                        "n_pos": n_pos, "n_neg": n_neg})
        return pd.DataFrame(rows), curves
    
    
    def _plot_auc_over_time(res, curves, step, t_min, t_max, title, ylabel,
                            axes=None, cmap="viridis", fontsize=8):
        if axes is None:
            fig, (ax, ax_roc) = plt.subplots(
                1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.7, 1]}
            )
        else:
            ax, ax_roc = axes
            fig = ax.figure
    
        # ---- Left panel: AUC vs t, with stacked class counts ----
        ax2 = ax.twinx()
        width = step * 0.6
        ax2.bar(res["t"], res["n_neg"], width=width, color="tab:blue",
                alpha=0.2, label="n true=0")
        ax2.bar(res["t"], res["n_pos"], width=width, bottom=res["n_neg"],
                color="tab:red", alpha=0.2, label="n true=1")
        ax2.set_ylabel("Number of ids", fontsize=fontsize + 1)
        ax2.tick_params(labelsize=fontsize)
    
        ax.plot(res["t"], res["roc_auc"], marker="o", ms=4, color="black",
                label="ROC-AUC")
        ax.axhline(0.5, color="grey", ls="--", lw=1, label="Chance")
        ax.axvline(0, color="red", ls=":", lw=1, label="Diagnosis")
        ax.set_xlabel("Years relative to diagnosis (t)", fontsize=fontsize + 1)
        ax.set_ylabel(ylabel, fontsize=fontsize + 1)
        ax.set_xticks(res["t"])
        ax.tick_params(labelsize=fontsize)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
    
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=fontsize)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    
        # ---- Right panel: overlaid ROC curves colored by t ----
        norm = Normalize(vmin=t_min, vmax=t_max)
        colormap = plt.get_cmap(cmap)
        for t, (fpr, tpr, _) in sorted(curves.items()):
            ax_roc.plot(fpr, tpr, color=colormap(norm(t)), lw=1.2, alpha=0.85)
        ax_roc.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_aspect("equal")
        ax_roc.set_xlabel("False positive rate", fontsize=fontsize + 1)
        ax_roc.set_ylabel("True positive rate", fontsize=fontsize + 1)
        ax_roc.tick_params(labelsize=fontsize)
        ax_roc.grid(alpha=0.3)
    
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_roc, fraction=0.046, pad=0.04)
        cbar.set_label("t (years relative to diagnosis)", fontsize=fontsize + 1)
        cbar.ax.tick_params(labelsize=fontsize)
    
        fig.suptitle(title, fontsize=fontsize + 3)
        fig.tight_layout()
        return fig
    
    
    def _save_or_close(fig, plot, save_path, save_name, dpi):
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, save_name), dpi=dpi,
                        bbox_inches="tight")
        if not plot:
            plt.close(fig)
 

    data = (
        df[[id_col, time_col, pred_col, true_col]]
        .dropna(subset=[time_col])
        .sort_values([id_col, time_col], kind="mergesort")
    )
 
    # Range from percentiles, snapped to the step grid
    lo, hi = np.percentile(data[time_col], percentiles)
    t_min = np.floor(lo / step) * step
    t_max = np.ceil(hi / step) * step
    n_steps = int(round((t_max - t_min) / step))
    grid = t_max - step * np.arange(n_steps + 1)  # from t_max down to t_min
 
    common = dict(id_col=id_col, time_col=time_col, pred_col=pred_col,
                  true_col=true_col, min_ids=min_ids)
    res, curves = _compute_auc_over_grid(data, grid, **common)
    res_window, curves_window = (None, None)
    if window is not None:
        res_window, curves_window = _compute_auc_over_grid(
            data, grid, window=window, **common
        )
 
    if plot or save_path is not None:
        style = dict(step=step, t_min=t_min, t_max=t_max, cmap=cmap,
                     fontsize=fontsize)
 
        fig = _plot_auc_over_time(
            res, curves,
            title="Last prediction per id with case_dt ≤ t",
            ylabel="ROC-AUC (last prediction with case_dt ≤ t)",
            axes=axes, **style,
        )
        _save_or_close(fig, plot, save_path, save_name, dpi)
 
        if window is not None:
            if save_name_window is None:
                stem, ext = os.path.splitext(save_name)
                save_name_window = f"{stem}_window{window:g}{ext or '.png'}"
            fig_w = _plot_auc_over_time(
                res_window, curves_window,
                title=f"Last prediction per id with t − {window:g} < case_dt ≤ t",
                ylabel=f"ROC-AUC (last prediction in (t − {window:g}, t])",
                axes=axes_window, **style,
            )
            _save_or_close(fig_w, plot, save_path, save_name_window, dpi)
 
    return res, res_window

