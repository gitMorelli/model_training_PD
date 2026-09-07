import tarfile
import time
import io
import torch
from PIL import Image
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
import random
import shutil
#from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import json

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

from src.utils.data_loading_utils import load_representations_handedness
from src.utils.model_utils import get_sklearn_model
from src.utils.data_loading_utils import prepare_exclusion_sets_PD, return_file_paths


params = {
    "type_of_repr": "feature", # representation, feature
    "problem": "PD",
    "class_col": 'diag_park_final1_quest',
    #To put for compatibility with code
    "grouped": False,
    'filter_missing': 'all',

    "model": "clip-vit-large-patch14-inter",
    "loaded_timestamp": "27082026",
    "seed": 42,

    "selected_model": "logreg",
    "sklearn_model_parameters": {},
    "selected_pipeline":None,

    #for representation
    "representation_type": "concat", #concat, mod, mean
    "use_pca": False,
    "n_components": 50,
    "n_splits": 5,
    "compare_all": False, #if True it will score all 6 combinations instead of just the one above

    "balanced_data": False,
    'balance_validation': False, #if True the validation set is balanced, if False it is not balanced
    "balancing_factor": 3,
}
params["source_path"] = os.path.join("/home/a_morelli/models/model_training_logs",params["problem"],
                                     f"{params['type_of_repr']}_extraction")  
if params["type_of_repr"] == "representation":
    params["source_path"] = os.path.join(params["source_path"], params['model'])
params["source_path"] = os.path.join(params["source_path"], params["loaded_timestamp"])

params['list_of_ids_paths'], params['data_folder'], params['grid_dict_path'] = return_file_paths(params['problem'], 
                                                                                                 False, False)

VERBOSE = True

def main():
    args = get_args()

    #splits = ["train", "val", "test"]
    
    df = load_file(params) 

    df = balance_data(df, params)

    if params["type_of_repr"] == "feature": #i save in the same format as the representation file -> subj_id,q,mo,rep -> i can use the same logic
        df,_ = reshape_features(df, keep_cols=["split"])

    df = add_info_to_df(df, params) #adding the class_col to the df

    process_representation(df, params)

def reshape_features(
    df,
    modalities=("text_original", "digit_original", "X_original"),
    id_col="subject_id",
    keep_cols=None,
    sort_col=None,
    verbose=True,
):
    """
    Wide -> long: q{N}_{modality}_{property} columns -> [<kept cols>, q, modality, rep]
 
    rep is a 1-D float array whose i-th entry is always the same property, for every
    (q, modality) block and every subject. That fixed ordering is the whole point.
 
    Parameters
    ----------
    modalities : listed explicitly because both modality and property contain
        underscores, so "q1_text_original_is_uniform" cannot be split unambiguously.
    id_col     : subject identifier, always kept and used for sorting.
    keep_cols  : extra non-feature columns to carry through (label, group, site, ...).
        None  -> keep every column that did not match the pattern.
        []    -> keep only id_col.
        [...] -> keep exactly these (plus id_col).
        Values are duplicated once per (q, modality) row.
    sort_col   : property name to sort `rep` by. Default None = alphabetical.
 
    Returns
    -------
    long_df : one row per (subject, q, modality)
    props   : the property names, in rep-array order. Keep this - it is the only
              thing that maps a rep index back to a feature name.
    """
    import re

    pat = re.compile(r"^q(\d+)_(" + "|".join(map(re.escape, modalities)) + r")_(.+)$")
 
    # Parse every column into (q, modality, property); ignore anything that doesn't match.
    parsed = {}
    for c in df.columns:
        m = pat.match(c)
        if m:
            parsed[c] = (int(m.group(1)), m.group(2), m.group(3))
    if not parsed:
        raise ValueError(f"no columns matched - check `modalities`: {list(modalities)}")
 
    unmatched = [c for c in df.columns if c not in parsed]
 
    # --- decide what rides along -----------------------------------------
    if keep_cols is None:
        keep = unmatched                          # everything non-feature
    else:
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise KeyError(f"keep_cols not in df: {missing}")
        keep = list(keep_cols)
    if id_col not in keep:                        # id_col is never optional
        keep.insert(0, id_col)
    if id_col not in df.columns:
        raise KeyError(f"id_col {id_col!r} not in df")
 
    dropped = [c for c in unmatched if c not in keep]
 
    meta = pd.DataFrame(parsed.values(), index=parsed.keys(),
                        columns=["q", "modality", "prop"])
 
    # ONE global property list -> identical ordering in every block.
    props = sorted(meta["prop"].unique())
    if sort_col is not None:                      # optional: pin one property first
        props = [sort_col] + [p for p in props if p != sort_col]
 
    if verbose:
        print(f"matched {len(parsed)} feature columns")
        print(f"keeping {len(keep)}: {keep}")
        if dropped:
            print(f"dropping {len(dropped)}: {dropped[:8]}{' ...' if len(dropped) > 8 else ''}")
        print(f"{len(props)} properties | q in {sorted(meta['q'].unique())} "
              f"| {meta['modality'].nunique()} modalities")
 
    # Positional alignment: every block is built from df's rows in original order,
    # so a plain concat lines up. Safer than merging on id_col, which would fan out
    # if id_col were ever non-unique.
    meta_df = df[keep].reset_index(drop=True)
 
    frames = []
    for (q, mod), grp in meta.groupby(["q", "modality"]):
        # prop -> column name for this block; reindex so a property missing from
        # this block becomes a NaN slot instead of shifting the whole array.
        lookup = pd.Series(grp.index.values, index=grp["prop"].values)
        cols = lookup.reindex(props)
 
        block = pd.DataFrame(index=range(len(df)), columns=props, dtype=float)
        present = cols.dropna()
        # to_numeric coerces bools -> 0/1 and any stray strings -> NaN
        block[present.index] = (df[present.values]
                                .apply(pd.to_numeric, errors="coerce")
                                .to_numpy())
 
        frames.append(pd.concat([
            meta_df,
            pd.DataFrame({"q": q, "modality": mod,
                          "rep": list(block.to_numpy(dtype=float))}),
        ], axis=1))
 
    long_df = (pd.concat(frames, ignore_index=True)
                 .sort_values([id_col, "modality", "q"])
                 .reset_index(drop=True))
 
    # rep last, metadata first
    long_df = long_df[[c for c in long_df.columns if c != "rep"] + ["rep"]]
 
    # ---- checks ---------------------------------------------------------
    D = len(props)
    assert long_df["rep"].map(len).eq(D).all(), "ragged rep arrays"
    expected = df[id_col].nunique() * meta.groupby(["q", "modality"]).ngroups
    if verbose:
        print(f"{long_df.shape[0]} rows (expected {expected}) | rep dim = {D}")
        if long_df.shape[0] != expected:
            print("  -> mismatch: id_col is probably not unique in df")
 
        # all-NaN or constant properties are dead weight; drop them from the input
        stacked = np.stack(long_df["rep"].to_numpy())
        with np.errstate(invalid="ignore"):
            nan_frac, sd = np.isnan(stacked).mean(0), np.nanstd(stacked, 0)
        print("all-NaN properties:", [p for p, f in zip(props, nan_frac) if f == 1.0])
        print("constant properties:", [p for p, s in zip(props, sd) if s == 0])
 
    return long_df, props

def balance_data(df, params, verbose=VERBOSE):
    exclusion_set, val_exclusion_set, _ = prepare_exclusion_sets_PD(
        params, verbose=verbose, class_col=params["class_col"])
    
    #remove rows with subject_id in either exclusion_set or val_exclusion_set
    unique_before = df[df['split'] == 'train']['subject_id'].nunique()
    complete_exclusion_set = exclusion_set.union(val_exclusion_set)
    df = df[~df['subject_id'].isin(complete_exclusion_set)]
    if verbose:
        print(f"Number of unique ids before exclusion (train split): {unique_before}")
        print(f"Number of unique ids after exclusion (train split): {df[df['split'] == 'train']['subject_id'].nunique()}")
    
    return df

def add_info_to_df(df, params):
    original_data = pd.read_parquet(params['list_of_ids_paths'])
    id_column_original = 'unique_id'
    id_column_df = 'subject_id'

    #add the params['class_col'] column to df by merging with original_data on the id columns
    df = df.merge(original_data[[id_column_original, params['class_col']]],
                  left_on=id_column_df, right_on=id_column_original, how='left')
    return df

def process_representation(df, params):
    REPRESENTATION = params["representation_type"]  # "concat", "mod", "mean"
    USE_PCA        = params["use_pca"]     # True -> add PCA before the ridge
    N_COMPONENTS   = params["n_components"]        # only used when USE_PCA (capped at n_subjects - 1)
    COMPARE_ALL    = params["compare_all"]      # score all 6 combinations instead of just the one above
    
    D    = df["rep"].iloc[0].shape[0]  # e.g. 1024 for CLIP
    SEED = params["seed"]

    n_splits = params.get("n_splits", 5)  # for KFold CV; not used if COMPARE_ALL
    # =============================================================================
    
    df = df[df['split'].isin(['train','val'])]  # i am discarding the test split for performing the crossval
    
    
    # ----------------------------------------------------------------------------
    # STEP 1 - long df -> one feature row per subject
    # ----------------------------------------------------------------------------
    
    # Collapse the two grouping axes (13 q x 3 modality) into one label, e.g. "eeg_q7".
    df["key"] = df["modality"].astype(str) + "_q" + df["q"].astype(str)
    
    # Sorted -> deterministic column order, stable across reruns and at predict time.
    keys     = sorted(df["key"].unique())
    subjects = np.sort(df["subject_id"].unique())
    
    # drop_duplicates guards against two rows for the same (subject, key), which would
    # make the index non-unique and break the reindex below.
    s = (df.drop_duplicates(["subject_id", "key"])
        .set_index(["subject_id", "key"])["rep"])
    
    # Force onto the COMPLETE subject x key grid: missing combos become NaN rather
    # than silently shortening a row and shifting every later feature by 1024.
    s = s.reindex(pd.MultiIndex.from_product([subjects, keys]))
    
    blocks = np.stack([
        np.asarray(r, dtype=float) if isinstance(r, (np.ndarray, list)) else np.full(D, np.nan)
        for r in s.values
    ])
    Bl = blocks.reshape(len(subjects), len(keys), D)      # (n, n_keys, D) block view
    n  = len(subjects)
    
    # One target per subject, reordered to match the row order of the features.
    y = (df.drop_duplicates("subject_id")
        .set_index("subject_id")
        .loc[subjects, params["class_col"]]
        .to_numpy())

    
    mod_of_key = np.array([k.rsplit("_q", 1)[0] for k in keys])
    
    with np.errstate(invalid="ignore"):                  # all-NaN blocks -> NaN, imputed later
        FEATURES = {
            # keep every block separately: most information, hardest to fit
            "concat": blocks.reshape(n, len(keys) * D),
            # keep the modality distinction, pool over q
            "mod":    np.concatenate(
                        [np.nanmean(Bl[:, mod_of_key == m, :], axis=1)
                        for m in np.unique(mod_of_key)], axis=1),
            # pool everything: fewest features, strongest prior
            "mean":   np.nanmean(Bl, axis=1),
        }
    
    print(f"{n} subjects | {len(keys)} keys")
    for k, v in FEATURES.items():
        print(f"  {k:7s} -> {v.shape[1]:6d} features")
    
    ########## CHECKS ###############
    # 1. Is y sane? Constant, near-constant, or a weird dtype all produce this.
    print("Checks ---- >")
    print(pd.Series(y).describe())
    print(pd.Series(y).nunique(), "unique values")

    # 2. Are the features actually varying across subjects?
    X = FEATURES["mod"]
    print("NaN fraction:", np.isnan(X).mean())
    print("per-column std - min/median/max:",
        np.nanstd(X, 0).min(), np.median(np.nanstd(X, 0)), np.nanstd(X, 0).max())
    print("duplicate rows:", len(X) - len(np.unique(np.round(X, 6), axis=0)))

    # 3. Is y aligned with X? This is the classic silent failure.
    chk = df.drop_duplicates("subject_id").set_index("subject_id").loc[subjects, params["class_col"]]
    print("aligned:", np.array_equal(chk.to_numpy(), y), "| index match:", (chk.index == subjects).all())

    # 4. Does ANY single feature correlate with the target at all?
    from scipy.stats import pearsonr
    r = np.array([pearsonr(X[:, j], y)[0] for j in range(0, X.shape[1], 8)])
    print("max |r| over sampled features:", np.nanmax(np.abs(r)).round(4), flush=True)
    ##################################
    
    # ----------------------------------------------------------------------------
    # STEP 2 - pipeline + subject-level CV
    # ----------------------------------------------------------------------------
    
    def build_pipe(use_pca=False, n_components=50):
        steps = [SimpleImputer(strategy="mean"), StandardScaler()]
        if use_pca:
            steps.append(PCA(n_components=min(n_components, n - 1), random_state=SEED))
        steps.append(LogisticRegressionCV(
            Cs=np.logspace(-4, 2, 20), penalty="l2", scoring="roc_auc",
            max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=SEED))
        return make_pipeline(*steps)
    
    # One row = one subject, so plain KFold is leakage-free. Splitting the original
    # long df would put the same subject on both sides of the split.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    
    SCORING = "roc_auc"   # chance level = 0.5 (R2's zero-baseline no longer applies)

    def score(rep, use_pca, verbose=True):
        X  = FEATURES[rep]
        sc = cross_val_score(build_pipe(use_pca), X, y, cv=cv, scoring=SCORING)
        if verbose:
            tag = f"pca{N_COMPONENTS}" if use_pca else "no-pca"
            print(f"{rep:7s} {tag:7s}  auc = {sc.mean():6.4f} +/- {sc.std():.4f}")
        return sc.mean()


    if COMPARE_ALL:
        print("\n5-fold stratified CV, identical folds:")
        results = {(r, p): score(r, p) for r in FEATURES for p in (False, True)}
        REPRESENTATION, USE_PCA = max(results, key=results.get)
        best = results[(REPRESENTATION, USE_PCA)]
        print(f"\nbest: {REPRESENTATION} / pca={USE_PCA}  auc = {best:.4f}"
            f"  ({best - 0.5:+.4f} over chance)")
    else:
        result = score(REPRESENTATION, USE_PCA)

    # Empirical chance level. Should land at ~0.50 - anything clearly above it means
    # the pipeline is leaking, and the score above is not trustworthy either.
    perm = np.random.default_rng(SEED).permutation(y)
    print("permuted labels:", cross_val_score(
        build_pipe(USE_PCA), FEATURES[REPRESENTATION], perm, cv=cv, scoring=SCORING).mean().round(4))


    # Refit on all data; the CV score above is the generalisation estimate.
    pipe = build_pipe(USE_PCA).fit(FEATURES[REPRESENTATION], y)

    # Sanity checks worth reading:
    #   C at either edge of the Cs grid -> widen np.logspace
    #   low explained variance          -> PCA is discarding too much
    print("\nC:", pipe[-1].C_[0])          # C_ is an array (one entry per class)
    if USE_PCA:
        print("explained variance:",
            pipe.named_steps["pca"].explained_variance_ratio_.sum().round(3))

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

def load_file(params):
    if params["type_of_repr"] == "representation":
        df = pd.read_parquet(os.path.join(params["source_path"], f"representations_{params['problem']}.parquet"))
    elif params["type_of_repr"] == "feature":
        df = pd.read_csv(os.path.join(params["source_path"], f"statistics_{params['problem']}.csv"))
    return df


if __name__ == "__main__":
    main()
    