import time
from tkinter.font import names
import os
import pandas as pd
import time
import numpy as np 
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Sequence
import contextlib
import io

from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from src.utils.data_loading_utils import load_representations_handedness
from src.utils.model_utils import get_sklearn_model
from src.utils.data_loading_utils import prepare_exclusion_sets_PD, return_file_paths
from src.utils.data_loading_utils import questionnaires_to_keep
from src.debug.PD_model_evaluation import  analyze_results
from sklearn.inspection import permutation_importance
from src.debug.PD_model_evaluation import tee_stdout, plot_probability_trajectories
from src.scripts.t_on_representations import balance_data, reshape_features, mask_features, add_info_to_df 

HOME_PATH = "/home/a_morelli/models/model_training_logs/PD/"
params = {

    "input_file":os.path.join(HOME_PATH, "feature_extraction/17092026/statistics_PD.csv"),
    "signal_column": 'cell_gap_run_v_mean_mean', #'cell_contour_jitter_rms_mean', #cell_sigma_x_mean
    
    "class_col": 'diag_park_final1_quest',
    #To put for compatibility with code
    "grouped": False,
    'filter_missing': 'last_q', #all

    "seed": 42,

    "balanced_data": False,
    'balance_validation': False, #if True the validation set is balanced, if False it is not balanced
    "balancing_factor": 3,

    'mask_features': True, #if True it will mask the features according to the logic in the function mask_features
    'censor_time': 'all_matched',#'pre_diagnosis', #'all_matched',#'first_and_last',#'successive','last_successive_and_previous',#'last_and_successive', #'all', 'pre_diagnosis', 'pre_diagnosis_1y', 'last_and_previous','last_and_successive'
    'filter_modality' : 'digit_original', #'X_original' 'digit_original' 'text_original'
}

params['list_of_ids_paths'], params['data_folder'], params['grid_dict_path'] = return_file_paths('PD', 
                                                                                                 False, False)

def get_type_of_input(params):
    out = None
    if 'feature_extraction' in params["input_file"]:
        out = "feature"
    else:
        out = "DL"
    return out
params["type_of_input"] = get_type_of_input(params)


VERBOSE = True
SAVE_PATH = "/home/a_morelli/vscode_projects/model_training/data/time_series_analysis"
EXPERIMENT = "time_series_comparison" #"plot_property" #"time_series_comparison"
os.makedirs(SAVE_PATH, exist_ok=True)

def main():
    start = time.time()

    #splits = ["train", "val", "test"]
    
    reference_df = pd.read_parquet(params['list_of_ids_paths'])
    reference_df = balance_data(reference_df, params, column="unique_id")

    df = load_file(params) 
    df = balance_data(df, params)

    if params["type_of_input"] == "feature": #i save in the same format as the representation file -> subj_id,q,mo,rep -> i can use the same logic
        df,properties = reshape_features(df, keep_cols=["split"])
        if params["mask_features"]:
            df = mask_features(df, params) #masking features for the representation
        df = add_info_to_df(df, params) #adding the class_col to the df
        #print("[DEBUG: Columns]",df.columns)
        #print(df.head())
        #returns a df with the following columns: ['subject_id', 'split', 'q', 'modality', 'rep', 'unique_id','diag_park_final1_quest'],
        #position in rep is given by the properties list, which is a list of tuples (prop_name, prop_value)
        print(properties)
        #assert 1==0
        df = df[df['split'] == 'val'] #only keep validation set for the analysis
        df = add_case_dt(df, reference_df, properties, n_slots=13) #adding the case_dt to the df
    else:
        params['signal_column']='probability_1'
        properties = None
    
    print(EXPERIMENT)
    if EXPERIMENT == "time_series_comparison":
        #normalize the signal columns to have mean 0 and std 1
        for p in properties:
            df[p] = (df[p] - df[p].mean()) / df[p].std()
        compare_signals(df, signal_columns=properties, n_steps=10, step_is_years=False,
                    class_names=("negative", "positive"), min_count=1, power=2,
                    save_path=SAVE_PATH, save_name="signal_comparison", only_common_rows=False)
    elif EXPERIMENT == "plot_property":
        for n in [5, 10, 15, 20]:
            statistics,diff = plot_probability_trajectories(df, n_steps=n, ax=None,
                                                class_names=("negative", "positive"),
                                                min_count=1, save_path=SAVE_PATH, save_name=f"{params['signal_column']}_probability_trajectories_{n}", 
                                                signal_column=params['signal_column'], plot_difference=True, show_points=False)
            score = weighted_abs_diff_sum(diff)
            print(f"Score for {params['signal_column']}: {score}")


# ----------------------------------------------------------------------------
# helpers signal comaprison
# ----------------------------------------------------------------------------
def _resolve_edges(case_dt, n_steps, step_is_years):
    """Same binning logic as plot_probability_trajectories."""
    lo, hi = case_dt.min(), case_dt.max()
    if step_is_years:
        n_steps = int(np.ceil((hi - lo) / n_steps))
    return np.linspace(lo, hi, n_steps + 1), n_steps


def _binned_diff(d, sig, edges, min_count):
    """Per-bin class-1 minus class-0 difference, without plotting."""
    n_bins = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    bins = np.clip(np.digitize(d["case_dt"], edges) - 1, 0, n_bins - 1)

    st = (d.assign(_bin=bins)
           .groupby(["true_label", "_bin"])[sig]
           .agg(mean="mean", std="std", n="count")
           .reset_index())
    st["sem"] = st["std"] / np.sqrt(st["n"])
    st = st[st["n"] >= min_count]

    s0 = st[st["true_label"] == 0].set_index("_bin")
    s1 = st[st["true_label"] == 1].set_index("_bin")
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
    diff["sem"] = np.sqrt(diff["sem_0"].fillna(0) ** 2 + diff["sem_1"].fillna(0) ** 2)
    return diff


def _bh_qvalues(p):
    """Benjamini-Hochberg adjusted p-values (NaNs preserved)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q

def compare_signals(df, signal_columns, n_steps=20, step_is_years=False,
                    class_names=("negative", "positive"), min_count=1, power=2,
                    top_k=8, heatmap_value="z",
                    pages=True, per_page=6, max_page_signals=None,
                    save_path=None, save_name="signal_comparison", only_common_rows=False):
    """Compare class separation across many signals.

    Outputs
    -------
    summary  : DataFrame, one row per signal, ranked by weighted z
               (score, se, z, p, BH q-value, n_bins)
    heatmap  : signals (rows, ranked) x bins (columns), colored by per-bin
               z = diff/sem (heatmap_value="z") or raw diff ("diff")
    overlay  : all difference curves in light gray, top_k highlighted+labeled
    pages    : multi-page PDF with the full curves + difference plot per
               signal, in rank order, `per_page` signals per page
               (only written if save_path is given)

    Returns (summary, diffs, figs).
    """
    signal_columns = list(signal_columns)

    if only_common_rows:
        # only keep rows where all signals are present
        d = df.dropna(subset=signal_columns + ["case_dt", "true_label"])
    else:
        d = df.copy()
    print(f"Using {len(d)} rows with all {len(signal_columns)} signals present "
          f"(dropped {len(df) - len(d)}).")

    edges, n_steps_eff = _resolve_edges(d["case_dt"], n_steps, step_is_years)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # ---- compute everything (no plotting) ----
    diffs, rows = {}, []
    for sig in signal_columns:
        diff = _binned_diff(d, sig, edges, min_count)
        diffs[sig] = diff
        if diff.empty:
            rows.append({"signal": sig, "score": np.nan, "se": np.nan,
                         "z": np.nan, "n_bins": 0})
            continue
        score, se = weighted_abs_diff_sum(diff, power=power, return_se=True)
        rows.append({"signal": sig, "score": score, "se": se,
                     "z": score / se if se > 0 else np.nan,
                     "n_bins": len(diff)})

    summary = pd.DataFrame(rows)
    summary["p"] = 2 * norm.sf(summary["z"])
    summary["q_bh"] = _bh_qvalues(summary["p"])
    summary = (summary.sort_values("z", ascending=False, na_position="last")
                      .reset_index(drop=True))
    ranked = summary["signal"].tolist()
    figs = {}

    # ---- 1. heatmap: signals x bins ----
    mat = np.full((len(ranked), len(centers)), np.nan)
    for i, sig in enumerate(ranked):
        dd = diffs[sig]
        if dd.empty:
            continue
        if heatmap_value == "z":
            vals = dd["diff"] / dd["sem"].replace(0, np.nan)
        else:
            vals = dd["diff"]
        mat[i, dd["_bin"].to_numpy()] = vals.to_numpy()

    finite = np.abs(mat[np.isfinite(mat)])
    vmax = np.percentile(finite, 98) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    fig_h, ax_h = plt.subplots(
        figsize=(max(6, 0.5 * len(centers) + 3), max(4, 0.16 * len(ranked) + 1.5)))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e0e0e0")  # gray = bin missing for one of the classes
    im = ax_h.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap,
                     vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_h.set_yticks(range(len(ranked)))
    ax_h.set_yticklabels(ranked, fontsize=6)
    step = max(1, len(centers) // 10)
    ax_h.set_xticks(range(0, len(centers), step))
    ax_h.set_xticklabels([f"{c:.1f}" for c in centers[::step]], fontsize=8)
    ax_h.set_xlabel("case_dt (bin center)")
    cbar = fig_h.colorbar(im, ax=ax_h, pad=0.01)
    cbar.set_label(("per-bin z (diff / SEM)" if heatmap_value == "z" else "diff")
                   + f"  [{class_names[1]} − {class_names[0]}]")
    ax_h.set_title("Signals ranked by weighted z (top = strongest separation)")
    fig_h.tight_layout()
    figs["heatmap"] = fig_h

    # ---- 2. overlay: gray background + highlighted top_k ----
    top = [s for s in ranked[:top_k] if not diffs[s].empty]
    others = [s for s in ranked if s not in top and not diffs[s].empty]

    fig_o, ax_o = plt.subplots(figsize=(11, 5.5))
    for sig in others:
        dd = diffs[sig]
        ax_o.plot(dd["center"], dd["diff"], color="0.75", lw=0.7,
                  alpha=0.6, zorder=1)
    tab = plt.get_cmap("tab10")
    for k, sig in enumerate(top):
        dd = diffs[sig]
        x, m, e = dd["center"].to_numpy(), dd["diff"].to_numpy(), dd["sem"].to_numpy()
        c = tab(k % 10)
        ax_o.fill_between(x, m - e, m + e, color=c, alpha=0.15, linewidth=0, zorder=2)
        ax_o.plot(x, m, "o-", color=c, lw=1.8, ms=4, zorder=3,
                  label=f"#{k + 1} {sig}")
    if others:
        ax_o.plot([], [], color="0.75", lw=0.8, label=f"other {len(others)} signals")
    ax_o.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.6)
    ax_o.set_xlabel("case_dt (binned)")
    ax_o.set_ylabel(f"Δ signal ({class_names[1]} − {class_names[0]})")
    ax_o.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax_o.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    fig_o.tight_layout()
    figs["overlay"] = fig_o

    # ---- save ----
    if save_path:
        fig_h.savefig(os.path.join(save_path, f"{save_name}_heatmap.png"),
                      dpi=300, bbox_inches="tight")
        fig_o.savefig(os.path.join(save_path, f"{save_name}_overlay.png"),
                      dpi=300, bbox_inches="tight")
        summary.to_csv(os.path.join(save_path, f"{save_name}_summary.csv"),
                       index=False)

    # ---- 3. paged detail plots (PDF) ----
    if pages and save_path:
        sigs = ranked if max_page_signals is None else ranked[:max_page_signals]
        pdf_path = os.path.join(save_path, f"{save_name}_pages.pdf")
        with PdfPages(pdf_path) as pdf:
            for start in range(0, len(sigs), per_page):
                chunk = sigs[start:start + per_page]
                fig_p, axes = plt.subplots(len(chunk), 2,
                                           figsize=(12, 2.6 * len(chunk)),
                                           squeeze=False)
                for i, sig in enumerate(chunk):
                    rank = start + i  # summary is sorted, index == rank - 1
                    with contextlib.redirect_stdout(io.StringIO()):  # silence prints
                        plot_probability_trajectories(
                            d, n_steps=n_steps_eff, step_is_years=False,
                            ax=axes[i, 0], diff_ax=axes[i, 1],
                            plot_difference=True, signal_column=sig,
                            class_names=class_names, min_count=min_count,
                            show_points=False, save_path=None)
                    axes[i, 0].set_title(f"#{rank + 1}  {sig}", fontsize=9, loc="left")
                    axes[i, 1].set_title(f"z = {summary.loc[rank, 'z']:.2f}   "
                                         f"q = {summary.loc[rank, 'q_bh']:.3g}",
                                         fontsize=9, loc="left")
                    for a in axes[i]:
                        if i > 0 and a.get_legend() is not None:
                            a.get_legend().remove()
                        if i < len(chunk) - 1:
                            a.set_xlabel("")
                fig_p.tight_layout()
                pdf.savefig(fig_p)
                plt.close(fig_p)
        print(f"Saved {len(sigs)} signal pages to {pdf_path}")
    elif pages:
        print("pages=True needs save_path; skipped the PDF.")

    if save_path:
        print(f"Saved heatmap, overlay and summary to {save_path}")

    return summary, diffs, figs

def weighted_abs_diff_sum(diff, power=2, normalize=True, min_sem=None,
                          return_se=False):
    """|sum_i w_i * diff_i| over bins, with w_i = 1 / sem_i**power.

    Parameters
    ----------
    diff : DataFrame
        Output of plot_probability_trajectories(..., plot_difference=True),
        needs columns "diff" and "sem".
    power : float
        Weight exponent. 2 = inverse-variance weighting (statistically optimal
        for independent bins), 1 = inverse-SEM weighting (milder down-weighting).
    normalize : bool
        If True, divide by sum(w): the result is a weighted mean difference,
        in the same units as the signal and comparable across n_steps.
        If False, return the raw weighted sum (scales with number of bins
        and with the size of the weights).
    min_sem : float or None
        Floor applied to SEMs before weighting. Bins with SEM = 0 (single-
        subject bins) would otherwise get infinite weight. Default: the
        smallest positive SEM present.
    return_se : bool
        If True, also return the standard error of the signed weighted
        sum/mean (before taking the absolute value), assuming independent bins.

    Returns
    -------
    float, or (float, float) if return_se.
    """
    d = diff[["diff", "sem"]].dropna()
    if d.empty:
        raise ValueError("no bins with both diff and sem")

    vals = d["diff"].to_numpy(dtype=float)
    sem = d["sem"].to_numpy(dtype=float)

    if min_sem is None:
        positive = sem[sem > 0]
        if positive.size == 0:
            raise ValueError("all SEMs are zero; pass min_sem explicitly")
        min_sem = positive.min()
    sem = np.maximum(sem, min_sem)

    w = sem ** (-float(power))
    total = np.sum(w * vals)
    se = np.sqrt(np.sum((w * sem) ** 2))  # error propagation, independent bins

    if normalize:
        total /= w.sum()
        se /= w.sum()  # for power=2 this equals 1 / sqrt(sum(w))

    result = abs(total)
    return (result, se) if return_se else result

def add_case_dt(result_df, train_df, properties, n_slots=13):
    """Attach case_dt to result_df by matching (unique_id, slot=i) to
    train_df's case_dt_dateq{i+1}. 
    Rename columns for using the trajectory analysis function"""
    cols = [f"case_dt_dateq{i}" for i in range(1, n_slots + 1)]

    long = (train_df[["unique_id", *cols]]
            .melt(id_vars="unique_id", value_vars=cols,
                var_name="_col", value_name="case_dt"))
    long["q"] = long["_col"].str.removeprefix("case_dt_dateq").astype(int) - 1
    #rename columns to match result_df
    long = long.rename(columns={"unique_id": "subject_id"})

    results_df = result_df.merge(long[["subject_id", "q", "case_dt"]],
                        on=["subject_id", "q"], how="left")
    
    results_df = results_df.rename(columns={"diag_park_final1_quest": "true_label"})

    for p in properties:
        #get index of the property of interest in the properties list
        prop_index = properties.index(p)
        #build a column for that  property using the rep column
        results_df[p] = results_df["rep"].apply(lambda x: x[prop_index])
    #drop the rep column
    results_df = results_df.drop(columns=["rep"])
    return results_df

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

def load_file(params):
    if params["type_of_input"] == "representation":
        pass
    elif params["type_of_input"] == "feature":
        df = pd.read_csv(params["input_file"])
    return df

'''def questionnaires_to_keep_wrapper(r, censor_time, original_data, sid, subject_images):
    questionnaire_info = {q: {'case_dt_dateq': r[f'case_dt_dateq{q}']} for q in range(1, 14)}
    keep = questionnaires_to_keep(r['last_avail_q'], censor_time, questionnaire_info,
                                  original_data, sid, subject_images,
                                  r['case_grid_pattern'], r['rempli_seulq12'])
    return keep'''

if __name__ == "__main__":
    main()
    