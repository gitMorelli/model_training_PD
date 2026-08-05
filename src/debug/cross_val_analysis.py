import tarfile
import time
import io
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import pandas as pd
import torch.nn as nn
import time
import webdataset as wds
import glob
from tqdm import tqdm
import torch.optim as optim
from torchvision import models
import torchvision.utils as vutils
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassRecall
import random
import shutil
#from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
import re
import signal
import sys
import json
import pickle
# 4. Compute and display metrics using scikit-learn
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import sys, os, contextlib
import glob, os, re

from src.utils.data_loading_utils import melt_df, prepare_exclusion_sets_PD, load_grid_dict, prepare_loaders_PD, return_file_paths
from src.utils.data_loading_utils import prepare_handedness_dataset, prepare_handedness_dataset_all, generate_exclusion_set_val
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, TiledJoinedModels
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_dataset
from src.utils.image_processing import ResizeLongestSide, get_augmentation_transform, get_transforms, get_mu_std
from src.utils.training_utils import BestMetricTracker, ModelPDGrouped, ModelPDClassification, ClearCache
from src.utils.model_utils import SequenceQuestionnaireModel, SetQuestionnaireModel
from src.scripts.train_PD_model import model_initialization
from src.debug.PD_model_evaluation import analyze_results, tee_stdout,return_model_info

SOURCE_PATH = "/home/a_morelli/models/model_training_logs/PD/cross_val"
CHECKPOINT_PATH = os.path.join(SOURCE_PATH, "resnet18_model_results/checkpoints")
version='2'
analyze_pre = True #if False all  timesteps are considered in the analysis, if true only the pre_diagnosis ones (only useful if i have
#per timestep results!)
params_path = os.path.join(CHECKPOINT_PATH,f"v_{version}", "exp_params.pkl")
#open and save as exp_params dict
with open(params_path, 'rb') as f:
    exp_params = pd.read_pickle(f)

VERBOSE = False
CLASS_COL = exp_params['class_col'] 
SCORE_COLS = ["probability_0", "probability_1"]   # order matters: col 0 = class 0
LABEL_COL  = "true_label"

def main():
    results = {}

    load_folder = os.path.join(CHECKPOINT_PATH, f"v_{version}")
    for path in glob.glob(os.path.join(load_folder, "fold_*_results.csv")):
        m = re.search(r"fold_(\d+)_results\.csv$", os.path.basename(path))
        if m:
            fold = int(m.group(1))
            results[fold] = pd.read_csv(path)

    results = dict(sorted(results.items()))  # optional: order by fold number
    if 'slot' in results[0].columns:
        has_slots=True
    else:
        has_slots=False
    

    log_path = os.path.join(load_folder, f"stats.txt") #copy prints also to a log file in the checkpoint folder
    with tee_stdout(log_path):
        return_model_info(exp_params)
        if has_slots:
            print("-----> this test has per_step results, plotting the per_step curves and auc by bin")
            pooled_df = concat_folds(results, fold_col="fold", subject_col="unique_id", extra_cols=(),
                 check_disjoint=True, verbose=True)
            plot_fold_comparison(pooled_df, fold_col="fold", n_steps=20,
                                 subject_col="unique_id", class_names=("negative", "positive"),
                                 min_count=1, min_folds=None, show_pooled=True,
                                 axes=None, save_path=load_folder, fname="fold_trajectories.png")
            plot_fold_grid(pooled_df, fold_col="fold", n_steps=20,
                           subject_col="unique_id", class_names=("negative", "positive"),
                           min_count=1, ncols=3, save_path=load_folder, fname="fold_grid.png")
            auc_by_bin = fold_auc_by_bin(pooled_df, fold_col="fold", n_steps=20, subject_col="unique_id", min_per_class=5)
            auc_by_bin.to_csv(os.path.join(load_folder, "fold_auc_by_bin.csv"), index=False)
            print(f"Saved fold AUC by bin to {os.path.join(load_folder, 'fold_auc_by_bin.csv')}")

            if analyze_pre:
                print(f"----> This test had per_step results and analyze_pre is set to True -> filtering the results to only consider the last timestep before diagnosis for each unique_id")
                for fold, df in results.items():
                    #load the metadata parquet file
                    metadata = pd.read_parquet(exp_params['list_of_ids_paths'])
                    results[fold] = filter_lastq_results(df, metadata)
        print("############## Pooled CV analysis ##############")
        analyze_cv_pooled(results, results_df=None,
                        pos_label=1, threshold=None, strategy="youden",
                        target_recall=0.90, plot=True, out_dir_path=load_folder)

        print("\n############## Per-fold CV analysis ##############")
        analyze_cv_per_fold(results, pos_label=1, threshold=None, strategy="youden",
                            target_recall=0.90, out_dir_path=load_folder)

        print("\n############## Per-fold CV analysis (1:1 matched) ##############")
        analyze_cv_per_fold(results, pos_label=1, threshold=None, strategy="youden",
                            target_recall=0.90, out_dir_path=load_folder, matched=True)
        
        print("\n############## per_step analysis ##############")


# per step cross val
PALETTE = {0: "#3B8BD4", 1: "#D85A30"}
REQUIRED = ["case_dt", "probability_1", "true_label"]
SLOT_COLS = ["unique_id", "slot", "true_label", "predicted_label",
             "probability_0", "probability_1", "case_dt"] 
 
# --------------------------------------------------------------------------- #
# assembling the OOF frame
# --------------------------------------------------------------------------- #
def concat_folds(frames, fold_col="fold", subject_col=None, extra_cols=(),
                 check_disjoint=True, verbose=True):
    """Stack per-fold prediction frames into one OOF frame with a fold column.
 
    frames : list/tuple of DataFrames  -> folds labelled 0..k-1 in order
             dict {label: DataFrame}   -> labels used as given (fold names,
                                          seeds, "outer0_inner2", ...)
 
    Validates the columns the downstream functions rely on, and — because
    these are meant to be out-of-fold predictions — warns if the same subject
    appears in more than one fold, which would mean the folds are not disjoint
    and the pooled curve double-counts.
    """
    if isinstance(frames, dict):
        items = list(frames.items())
    else:
        items = list(enumerate(frames))
    if not items:
        raise ValueError("no frames given")
 
    need = list(REQUIRED) + ([subject_col] if subject_col else []) \
        + list(extra_cols)
    keep = need + []  # columns carried through; add anything else you need
 
    out = []
    for label, f in items:
        if f is None or len(f) == 0:
            if verbose:
                print(f"fold {label}: empty, skipped")
            continue
        missing = [c for c in need if c not in f.columns]
        if missing:
            raise KeyError(f"fold {label} is missing columns: {missing}")
        if fold_col in f.columns and not (f[fold_col] == label).all():
            raise ValueError(
                f"fold {label} already has a '{fold_col}' column with "
                f"different values; rename it or pass a different fold_col")
        g = f.loc[:, [c for c in keep if c in f.columns]].copy()
        g[fold_col] = label
        out.append(g)
 
    if not out:
        raise ValueError("all frames were empty")
    df = pd.concat(out, ignore_index=True)
 
    # ---- sanity checks ----------------------------------------------------
    df["true_label"] = df["true_label"].astype(int)
    bad = set(df["true_label"].unique()) - {0, 1}
    if bad:
        raise ValueError(f"true_label has unexpected values: {sorted(bad)}")
    p = df["probability_1"]
    if p.notna().any() and (p.min() < 0 or p.max() > 1):
        print(f"WARNING: probability_1 outside [0, 1] "
              f"(min={p.min():.3g}, max={p.max():.3g}) — logits, not probs?")
 
    n_na = df[REQUIRED].isna().any(axis=1).sum()
    if n_na and verbose:
        print(f"WARNING: {n_na} rows have NaN in {REQUIRED} "
              f"and will be dropped downstream")
 
    if subject_col is not None:
        per_subj = df.groupby(subject_col)[fold_col].nunique()
        leaked = per_subj[per_subj > 1]
        if check_disjoint and len(leaked):
            print(f"WARNING: {len(leaked)} subjects appear in more than one "
                  f"fold — these are not disjoint OOF splits. "
                  f"e.g. {list(leaked.index[:5])}")
        dup = df.duplicated(subset=[subject_col, "case_dt", fold_col]).sum()
        if dup and verbose:
            print(f"WARNING: {dup} duplicated ({subject_col}, case_dt) rows "
                  f"within a fold")
 
    if verbose:
        by_fold = (df.groupby(fold_col)
                     .agg(rows=("probability_1", "size"),
                          pos_rate=("true_label", "mean"),
                          dt_min=("case_dt", "min"),
                          dt_max=("case_dt", "max")))
        if subject_col is not None:
            by_fold["subjects"] = df.groupby(fold_col)[subject_col].nunique()
        print(by_fold.to_string(float_format=lambda v: f"{v:.3g}"))
 
    return df.reset_index(drop=True)
def load_folds(paths, fold_col="fold", subject_col=None, labels=None,
               reader=None, **kwargs):
    """Same thing, reading from disk. Fold labels default to the file stems.
 
    paths  : list of file paths (csv / parquet), or a glob string
    labels : optional explicit labels, same length as paths
    """
    import glob as _glob
 
    if isinstance(paths, str):
        paths = sorted(_glob.glob(paths))
    if not paths:
        raise ValueError("no files matched")
 
    def _read(p):
        if reader is not None:
            return reader(p)
        return pd.read_parquet(p) if p.endswith((".parquet", ".pq")) \
            else pd.read_csv(p)
 
    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(labels) != len(paths):
        raise ValueError("labels and paths differ in length")
    return concat_folds({lab: _read(p) for lab, p in zip(labels, paths)},
                        fold_col=fold_col, subject_col=subject_col, **kwargs)
def filter_lastq_results(csv_data, metadata):
    #i can drop the nans cause a timestep is na only if the grid_pattern was 1 and the rempli_pattern was 0 (which is unreliable)
    csv_data = csv_data.dropna(subset=['case_dt'])
    
    #add the case_control, rempli_pattern, case_pattern, grid_pattern, case_grid_pattern columns to the nan_case_dt_rows dataframe by merging with the metadata dataframe on the unique_id column
    csv_data = csv_data.merge(metadata[['unique_id', 'last_avail_q']], on='unique_id', how='left')
    
    filtered = (csv_data[csv_data["slot"] <= csv_data["last_avail_q"] - 1]
         .sort_values("slot")
         .groupby("unique_id", as_index=False)
         .tail(1)
         .sort_index())

    return filtered
# --------------------------------------------------------------------------- #
# binning + stats, per_step functions
# --------------------------------------------------------------------------- #
def make_edges(df, n_steps=20, lo=None, hi=None):
    """Global bin edges. Compute ONCE from the pooled OOF frame and reuse."""
    d = df.dropna(subset=REQUIRED)
    if d.empty:
        raise ValueError("no rows with case_dt, probability_1, and true_label")
    lo = d["case_dt"].min() if lo is None else lo
    hi = d["case_dt"].max() if hi is None else hi
    return np.linspace(lo, hi, n_steps + 1)
def assign_bins(df, edges):
    d = df.dropna(subset=REQUIRED).copy()
    d["_bin"] = np.clip(np.digitize(d["case_dt"], edges) - 1, 0, len(edges) - 2)
    d["center"] = (0.5 * (edges[:-1] + edges[1:]))[d["_bin"].to_numpy()]
    return d
def fold_curves(df, edges, fold_col="fold", subject_col=None, min_count=1):
    """Mean probability_1 per (fold, true_label, bin).
 
    If subject_col is given, rows are collapsed to one value per subject per
    bin first, so `n` counts subjects rather than repeated timesteps and the
    within-fold SEM is not inflated by within-subject correlation.
    """
    d = assign_bins(df, edges)
    keys = [fold_col, "true_label", "_bin", "center"]
    if subject_col is not None:
        d = (d.groupby(keys + [subject_col], as_index=False)["probability_1"]
               .mean())
    g = (d.groupby(keys)["probability_1"]
           .agg(mean="mean", std="std", n="count")
           .reset_index())
    g["sem"] = g["std"] / np.sqrt(g["n"])
    return g[g["n"] >= min_count].sort_values(keys).reset_index(drop=True)
def across_folds(curves, value="mean", min_folds=None):
    """Unweighted mean of the per-fold means, with between-fold spread.
 
    `sem` here is fold-to-fold variability (n = number of folds contributing to
    that bin), which is what you want when asking whether the folds agree. It
    is NOT the subject-level SEM from `fold_curves`.
    """
    out = (curves.groupby(["true_label", "_bin", "center"])[value]
                 .agg(mean="mean", std="std", n_folds="count")
                 .reset_index())
    out["sem"] = out["std"] / np.sqrt(out["n_folds"])
    if min_folds is not None:
        out = out[out["n_folds"] >= min_folds]
    return out.sort_values(["true_label", "_bin"]).reset_index(drop=True)
def pooled_curve(df, edges, subject_col=None, min_count=1):
    """Count-weighted curve over all OOF rows: the estimate you'd report."""
    dummy = df.assign(_all=0)
    c = fold_curves(dummy, edges, fold_col="_all",
                    subject_col=subject_col, min_count=min_count)
    return c.drop(columns="_all")
def separation(curves, fold_col="fold"):
    """Delta(bin) = mean(class 1) - mean(class 0), one row per (fold, bin)."""
    w = curves.pivot_table(index=[fold_col, "_bin", "center"],
                           columns="true_label", values="mean")
    w = w.dropna(subset=[c for c in (0, 1) if c in w.columns])
    if not {0, 1}.issubset(w.columns):
        raise ValueError("need both true_label 0 and 1 present in every fold")
    return (w.assign(delta=w[1] - w[0])
             .reset_index()[[fold_col, "_bin", "center", "delta"]]) 
# --------------------------------------------------------------------------- #
# plotting per_step curves
# --------------------------------------------------------------------------- #
def plot_fold_comparison(df, fold_col="fold", n_steps=20, subject_col=None,
                         class_names=("negative", "positive"), min_count=1,
                         min_folds=None, show_pooled=True, axes=None,
                         save_path=None, fname="fold_trajectories.png"):
    """Left: per-class curves, one thin line per fold + pooled bold curve.
    Right: separation (class1 - class0) per fold + across-fold mean +/- SEM.
 
    Returns a dict of the underlying frames.
    """
    edges = make_edges(df, n_steps)
    curves = fold_curves(df, edges, fold_col=fold_col,
                         subject_col=subject_col, min_count=min_count)
    summary = across_folds(curves, min_folds=min_folds)
    pooled = pooled_curve(df, edges, subject_col=subject_col,
                          min_count=min_count)
    sep = separation(curves, fold_col=fold_col)
    sep_summary = (sep.groupby(["_bin", "center"])["delta"]
                      .agg(mean="mean", std="std", n_folds="count")
                      .reset_index())
    sep_summary["sem"] = sep_summary["std"] / np.sqrt(sep_summary["n_folds"])
    if min_folds is not None:
        sep_summary = sep_summary[sep_summary["n_folds"] >= min_folds]
 
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax0, ax1 = axes
 
    # ---- left panel -------------------------------------------------------
    for label, name in zip((0, 1), class_names):
        color = PALETTE[label]
        for _, fc in curves[curves["true_label"] == label].groupby(fold_col):
            fc = fc.sort_values("_bin")
            ax0.plot(fc["center"], fc["mean"], color=color, lw=1.0,
                     alpha=0.45, zorder=2)
        s = summary[summary["true_label"] == label].sort_values("_bin")
        if not s.empty:
            e = s["sem"].fillna(0).to_numpy()
            ax0.fill_between(s["center"], s["mean"] - e, s["mean"] + e,
                             color=color, alpha=0.18, linewidth=0, zorder=1)
        if show_pooled:
            p = pooled[pooled["true_label"] == label].sort_values("_bin")
            ax0.plot(p["center"], p["mean"], color=color, lw=2.6,
                     marker="o", markersize=5, markeredgecolor="white",
                     markeredgewidth=1,
                     label=f"true_label = {label} ({name}), pooled OOF",
                     zorder=4)
    ax0.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.6)
    ax0.set_xlabel("case_dt (binned)")
    ax0.set_ylabel("probability_1")
    ax0.set_title("Per-fold curves (thin) vs pooled OOF (bold)\n"
                  "band = ±SEM across folds", fontsize=10)
    ax0.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax0.legend(loc="best", frameon=False, fontsize=9)
 
    # ---- right panel ------------------------------------------------------
    for fold, fs in sep.groupby(fold_col):
        fs = fs.sort_values("_bin")
        ax1.plot(fs["center"], fs["delta"], lw=1.0, alpha=0.55,
                 label=f"fold {fold}")
    ss = sep_summary.sort_values("_bin")
    e = ss["sem"].fillna(0).to_numpy()
    ax1.fill_between(ss["center"], ss["mean"] - e, ss["mean"] + e,
                     color="black", alpha=0.12, linewidth=0)
    ax1.plot(ss["center"], ss["mean"], color="black", lw=2.4, marker="o",
             markersize=5, markeredgecolor="white", markeredgewidth=1,
             label="mean across folds")
    ax1.axhline(0.0, color="gray", lw=0.7, ls="--", alpha=0.6)
    ax1.set_xlabel("case_dt (binned)")
    ax1.set_ylabel("mean prob (class 1) − mean prob (class 0)")
    ax1.set_title("Class separation per fold", fontsize=10)
    ax1.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax1.legend(loc="best", frameon=False, fontsize=8, ncol=2)
 
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, fname)
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved fold comparison plot to {fig_path}")
 
    return {"curves": curves, "summary": summary, "pooled": pooled,
            "separation": sep, "separation_summary": sep_summary,
            "edges": edges} 
def plot_fold_grid(df, fold_col="fold", n_steps=20, subject_col=None,
                   class_names=("negative", "positive"), min_count=1,
                   ncols=3, save_path=None, fname="fold_grid.png"):
    """One panel per fold, shared axes — for spotting a single bad fold."""
    edges = make_edges(df, n_steps)
    curves = fold_curves(df, edges, fold_col=fold_col,
                         subject_col=subject_col, min_count=min_count)
    folds = sorted(curves[fold_col].unique())
    nrows = int(np.ceil(len(folds) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows),
                            sharex=True, sharey=True, squeeze=False)
    for ax, fold in zip(axs.ravel(), folds):
        fc = curves[curves[fold_col] == fold]
        for label, name in zip((0, 1), class_names):
            s = fc[fc["true_label"] == label].sort_values("_bin")
            if s.empty:
                continue
            e = s["sem"].fillna(0).to_numpy()
            ax.fill_between(s["center"], s["mean"] - e, s["mean"] + e,
                            color=PALETTE[label], alpha=0.22, linewidth=0)
            ax.plot(s["center"], s["mean"], color=PALETTE[label], lw=1.8,
                    marker="o", markersize=4, label=f"{label} ({name})")
        ax.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.6)
        ax.set_title(f"fold {fold}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    for ax in axs.ravel()[len(folds):]:
        ax.axis("off")
    axs[0, 0].legend(loc="best", frameon=False, fontsize=8)
    fig.supxlabel("case_dt (binned)")
    fig.supylabel("probability_1")
    fig.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, fname)
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved fold grid plot to {fig_path}")
    return curves 
def fold_auc_by_bin(df, fold_col="fold", n_steps=20, subject_col=None,
                    min_per_class=5):
    """Per-fold, per-bin AUC — a scale-free companion to the mean curves.
 
    Useful because two folds can differ in calibration (curves shifted) while
    ranking cases identically (AUC unchanged), which the mean curves hide.
    """
    from sklearn.metrics import roc_auc_score
 
    edges = make_edges(df, n_steps)
    d = assign_bins(df, edges)
    if subject_col is not None:
        d = (d.groupby([fold_col, "true_label", "_bin", "center", subject_col],
                       as_index=False)["probability_1"].mean())
    rows = []
    for (fold, b, c), g in d.groupby([fold_col, "_bin", "center"]):
        y = g["true_label"].to_numpy()
        if min(np.sum(y == 0), np.sum(y == 1)) < min_per_class:
            continue
        rows.append({fold_col: fold, "_bin": b, "center": c,
                     "auc": roc_auc_score(y, g["probability_1"].to_numpy()),
                     "n": len(g)})
    return pd.DataFrame(rows)

#----- Others -------------------
def _to_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

def analyze_cv_pooled(results, results_df=None, **kwargs):
    # results: dict {fold: DataFrame}  (from your earlier loader)
    frames = [results[f] for f in sorted(results)]
    pooled = pd.concat(frames, ignore_index=True)

    scores, labels = _scores_labels_from_df(pooled)
    return analyze_results(scores, labels, results_df=pooled,
                           split="cv_oof", **kwargs)

def analyze_cv_per_fold(results, metrics=("pr_auc", "roc_auc", "youden_j","balanced_acc","f1_positive",
                                          "sensitivity", "specificity", "precision_positive","accuracy"), matched=False, **kwargs):
    per_fold = {}
    for f in sorted(results):
        print(f"\n########## FOLD {f} ##########")
        fold_results = results[f]
        #get the unique_id with label 1
        label_1_ids = fold_results.loc[fold_results[LABEL_COL] == 1, "unique_id"].unique()
        #get the unique_id with label 0
        label_0_ids = fold_results.loc[fold_results[LABEL_COL] == 0, "unique_id"].unique()
        #get the minimum number of samples between the two classes
        min_samples = min(len(label_1_ids), len(label_0_ids))
        #sample the unique_id with label 0 to match the number of samples with label 1
        if matched:
            label_0_ids_matched = np.random.choice(label_0_ids, size=min_samples, replace=False)
            #filter the fold_results to keep only the matched samples
            fold_results = fold_results[fold_results["unique_id"].isin(np.concatenate([label_1_ids, label_0_ids_matched]))]
        scores, labels = _scores_labels_from_df(fold_results)
        per_fold[f] = analyze_results(scores, labels, results_df=fold_results,
                                    split=f"fold_{f}", plot=False, **kwargs)

    print("\n########## CV SUMMARY ##########")
    for m in metrics:
        vals = np.array([per_fold[f][m] for f in per_fold])
        print(f"{m:>10s}: {vals.mean():.3f} ± {vals.std():.3f}  "
            f"(folds: {', '.join(f'{v:.3f}' for v in vals)})")
    return per_fold

def _scores_labels_from_df(df):
    scores = df[SCORE_COLS].to_numpy(dtype=float)   # (N, 2)
    labels = df[LABEL_COL].to_numpy().astype(int)   # (N,)
    return scores, labels

if __name__ == "__main__":
    main()
