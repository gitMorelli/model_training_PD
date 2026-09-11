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
import pickle

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
from src.utils.data_loading_utils import questionnaires_to_keep
from src.debug.PD_model_evaluation import  analyze_results

params = {
    'save_dir_root': '/home/a_morelli/models/model_training_logs/PD/feature_extraction/trained_models',

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
    "representation_type": "mean", #concat, mod, mean
    "use_pca": False,
    "n_components": 50,
    "n_splits": 5,
    "compare_all": False, #if True it will score all 6 combinations instead of just the one above

    "balanced_data": False,
    'balance_validation': False, #if True the validation set is balanced, if False it is not balanced
    "balancing_factor": 3,

    'censor_time': 'all_matched',#'pre_diagnosis', #'all_matched',#'first_and_last',#'successive','last_successive_and_previous',#'last_and_successive', #'all', 'pre_diagnosis', 'pre_diagnosis_1y', 'last_and_previous','last_and_successive'
    'filter_modality' : 'digit_original', #'X_original' 'digit_original' 'text_original'
}
'''
Notes
q is from 1 to 13 in the reshaped df, same for the list_of_ids_paths
'''
params["source_path"] = os.path.join("/home/a_morelli/models/model_training_logs",params["problem"],
                                     f"{params['type_of_repr']}_extraction")  
if params["type_of_repr"] == "representation":
    params["source_path"] = os.path.join(params["source_path"], params['model'])
params["source_path"] = os.path.join(params["source_path"], params["loaded_timestamp"])

params['list_of_ids_paths'], params['data_folder'], params['grid_dict_path'] = return_file_paths(params['problem'], 
                                                                                                 False, False)

VERBOSE = True

def main():
    start = time.time()

    #splits = ["train", "val", "test"]
    
    df = load_file(params) 

    df = balance_data(df, params)

    time_taken = time.time() - start
    print(f"----- > Time taken to load and balance the data: {time_taken:.2f} seconds", flush=True)

    #keep only 100 unique subject_ids for testing in the train and 20 in the validation and test sets
    '''train_subjects = df[df['split'] == 'train']['subject_id'].unique()[:100]
    val_subjects = df[df['split'] == 'val']['subject_id'].unique()[:20]
    df = df[df['subject_id'].isin(train_subjects) | df['subject_id'].isin(val_subjects)]'''

    if params["type_of_repr"] == "feature": #i save in the same format as the representation file -> subj_id,q,mo,rep -> i can use the same logic
        df,properties = reshape_features(df, keep_cols=["split"])
    
    time_taken = time.time() - start
    print(f"----- > Time taken to reshape the data: {time_taken:.2f} seconds", flush=True)
    
    
    df = mask_features(df, params) #masking features for the representation

    time_taken = time.time() - start
    print(f"----- > Time taken to mask the features: {time_taken:.2f} seconds", flush=True)

    df = add_info_to_df(df, params) #adding the class_col to the df

    print(f"Completed preprocessing", flush=True)

    results_df = process_representation(df, params, verbose=VERBOSE)

    all_probs, all_labels = results_df["probability_0"].to_numpy(), results_df["true_label"].to_numpy()

    save_dir = get_save_path(params)


    analyze_results(all_probs, all_labels, results_df, split="validation",
                        pos_label=1, threshold=None, strategy="youden",
                        target_recall=0.90, plot=True, out_dir_path=save_dir)

    save_results(save_dir, params, results_df)


def get_save_path(params):
    save_dir_model = os.path.join(params['save_dir_root'], params['model'])
    os.makedirs(save_dir_model, exist_ok=True)

    #get the subfolders, they are in the form v_{number}, get the last one and increment it by 1
    subfolders = [f for f in os.listdir(save_dir_model) if os.path.isdir(os.path.join(save_dir_model, f)) and f.startswith('v_')]
    if subfolders:
        last_folder = max(subfolders, key=lambda x: int(x.split('_')[1]))
        new_folder = f"v_{int(last_folder.split('_')[1]) + 1}"
    else:
        new_folder = "v_1"

    save_dir = os.path.join(save_dir_model, new_folder)

    os.makedirs(save_dir, exist_ok=True)

    return save_dir

def save_results(save_dir, params, results_df):
    #save the params in a pkl file
    params_path = os.path.join(save_dir, "params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Params saved to {params_path}")

    #save the results in a csv file
    results_path = os.path.join(save_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

def mask_features(df_source, params):
    original_data = pd.read_parquet(params['list_of_ids_paths'])
    original_id_column = 'unique_id'

    def mask_if_modality_missing(
            long_df,
            required_modality,
            id_col="subject_id",
            q_col="q",
            modality_col="modality",
            rep_col="rep",
            inplace=False,
            verbose=True,
        ):
            """
            For each (subject, q): if `required_modality`'s row is missing (either
            absent entirely, or present but all-NaN, e.g. from an earlier masking
            step), NaN-out the rep of every OTHER modality for that same (subject, q).

            Parameters
            ----------
            long_df : output of reshape_features (optionally already passed through
                mask_dropped_questionnaires).
            required_modality : the modality whose presence gates the others,
                e.g. "digit_original".
            inplace : if False (default), operate on a copy.

            Returns
            -------
            long_df with `rep` replaced by all-NaN on every non-`required_modality`
            row of a (subject, q) group where `required_modality` was missing.
            """
            if not inplace:
                long_df = long_df.copy()

            D = len(long_df[rep_col].iloc[0])

            def _is_missing(rep):
                arr = np.asarray(rep, dtype=float)
                return arr.size == 0 or np.isnan(arr).all()

            n_masked = 0
            n_groups_gated = 0

            for (subject_id, q), idx in long_df.groupby([id_col, q_col]).groups.items():
                rows = long_df.loc[idx]
                req_rows = rows[rows[modality_col] == required_modality]

                # missing = no such row at all, or the row exists but rep is all-NaN
                required_missing = req_rows.empty or req_rows[rep_col].map(_is_missing).all()

                if required_missing:
                    other_idx = rows.index[rows[modality_col] != required_modality]
                    if len(other_idx):
                        n_groups_gated += 1
                        n_masked += len(other_idx)
                        long_df.loc[other_idx, rep_col] = pd.Series(
                            [np.full(D, np.nan) for _ in range(len(other_idx))],
                            index=other_idx,
                        )

            if verbose:
                print(f"masked {n_masked} rows across {n_groups_gated} (subject, q) "
                    f"groups missing modality {required_modality!r}")

            return long_df

    def mask_dropped_questionnaires(
        long_df,
        get_questionnaires_to_keep,
        id_col="subject_id",
        q_col="q",
        rep_col="rep",
        inplace=False,
        verbose=True,
        **kwargs,
    ):
        """
        NaN-out rep rows whose `q` is not in that subject's keep-list.

        Parameters
        ----------
        long_df : output of reshape_features (one row per subject/q/modality,
            `rep_col` holding a fixed-length float array per row).
        get_questionnaires_to_keep : callable
            get_questionnaires_to_keep(subject_id, **kwargs) -> iterable of q's to KEEP
            for that subject. Anything else is set to NaN. `**kwargs` lets you
            pass through whatever extra arguments this function needs; they are
            forwarded unchanged on every call.
        id_col, q_col, rep_col : column names, matching reshape_features's output.
        inplace : if False (default), operate on a copy.

        Returns
        -------
        long_df with `rep` replaced by an all-NaN vector (same length as before)
        on every row whose q was not in that subject's keep-list.
        """
        if not inplace:
            long_df = long_df.copy()

        D = len(long_df[rep_col].iloc[0])
        n_masked = 0

        for subject_id, idx in long_df.groupby(id_col).groups.items():
            
            #get the arguments for the current subject
            subject_row = original_data.loc[original_data[original_id_column] == subject_id]
            questionnaire_info = {}
            for q in range(1,14):
                questionnaire_info[q] = {
                    'case_dt_dateq': subject_row[f'case_dt_dateq{q}'].iloc[0],
                }
            last_q = subject_row['last_avail_q'].iloc[0]
            censor_time = params['censor_time']
            subject_images = [i for i in range(1,14) ] #placeholder, subject_images was used to check in the tar file
            case_grid_pattern = subject_row['case_grid_pattern'].iloc[0]
            rempli_seulq12 = subject_row['rempli_seulq12'].iloc[0]
            
            keep = set(get_questionnaires_to_keep(last_q, censor_time, questionnaire_info, original_data,subject_id,subject_images, 
                                                  case_grid_pattern, rempli_seulq12))
            rows = long_df.loc[idx]
            drop_mask = ~rows[q_col].isin(keep)

            if drop_mask.any():
                drop_idx = rows.index[drop_mask]
                n_masked += len(drop_idx)
                # one fresh NaN array per row, so nothing shares a reference
                long_df.loc[drop_idx, rep_col] = pd.Series(
                    [np.full(D, np.nan) for _ in range(len(drop_idx))],
                    index=drop_idx,
                )

        if verbose:
            print(f"masked {n_masked}/{len(long_df)} rows "
                f"({long_df.groupby(id_col).ngroups} subjects checked)")

        return long_df
    
    df = df_source.copy()  # don't overwrite the original
    df = mask_if_modality_missing(df, required_modality=params['filter_modality'], inplace=True)
    df = mask_dropped_questionnaires(df, get_questionnaires_to_keep=questionnaires_to_keep, inplace=True)

    return df

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
    
    Missing properties in a (q, modality) block become NaN slots in the rep array
    Missing (q, modality) blocks become NaN rows in the long_df. 
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

def process_representation(df, params, verbose=False):
    REPRESENTATION = params["representation_type"]  # "concat", "mod", "mean"
    USE_PCA        = params["use_pca"]              # True -> add PCA before the model
    N_COMPONENTS   = params["n_components"]         # only used when USE_PCA (capped at n_train - 1)
    COMPARE_ALL    = params["compare_all"]          # run every representation instead of just the one above

    D    = df["rep"].iloc[0].shape[0]  # e.g. 1024 for CLIP
    SEED = params["seed"]

    # .copy() avoids SettingWithCopyWarning when adding the "key" column below
    df = df[df["split"].isin(["train", "val"])].copy()

    # ----------------------------------------------------------------------------
    # STEP 1 - long df -> one feature row per subject
    # ----------------------------------------------------------------------------
    df["key"] = df["modality"].astype(str) + "_q" + df["q"].astype(str)

    keys     = sorted(df["key"].unique())
    subjects = np.sort(df["subject_id"].unique())

    s = (df.drop_duplicates(["subject_id", "key"])
           .set_index(["subject_id", "key"])["rep"])
    s = s.reindex(pd.MultiIndex.from_product([subjects, keys]))

    blocks = np.stack([
        np.asarray(r, dtype=float) if isinstance(r, (np.ndarray, list)) else np.full(D, np.nan)
        for r in s.values
    ])
    Bl = blocks.reshape(len(subjects), len(keys), D)
    n  = len(subjects)

    subj_info = (df.drop_duplicates("subject_id")
                   .set_index("subject_id")
                   .loc[subjects])
    y = subj_info[params["class_col"]].to_numpy()

    # ---- NEW: one split per subject, aligned with the feature rows ----
    # A subject appearing in both train and val would leak information.
    n_splits_per_subj = df.groupby("subject_id")["split"].nunique()
    leaky = n_splits_per_subj[n_splits_per_subj > 1].index.tolist()
    if leaky:
        raise ValueError(f"{len(leaky)} subjects appear in both train and val, e.g. {leaky[:5]}")

    split_of_subj = subj_info["split"].to_numpy()
    train_mask = split_of_subj == "train"
    val_mask   = split_of_subj == "val"
    n_train    = int(train_mask.sum())
    print(f"train subjects: {n_train} | val subjects: {int(val_mask.sum())}")

    mod_of_key = np.array([k.rsplit("_q", 1)[0] for k in keys])

    with np.errstate(invalid="ignore"):
        FEATURES = {
            "concat": blocks.reshape(n, len(keys) * D),
            "mod":    np.concatenate(
                          [np.nanmean(Bl[:, mod_of_key == m, :], axis=1)
                           for m in np.unique(mod_of_key)], axis=1),
            "mean":   np.nanmean(Bl, axis=1),
        }

    print(f"{n} subjects | {len(keys)} keys")
    for k, v in FEATURES.items():
        print(f"  {k:7s} -> {v.shape[1]:6d} features")

    ########## CHECKS ###############
    if verbose:
        print("Checks ---- >")
        print(pd.Series(y).describe())
        print(pd.Series(y).nunique(), "unique values")

        X = FEATURES["mod"]
        print("NaN fraction:", np.isnan(X).mean())
        print("per-column std - min/median/max:",
              np.nanstd(X, 0).min(), np.median(np.nanstd(X, 0)), np.nanstd(X, 0).max())
        print("duplicate rows:", len(X) - len(np.unique(np.round(X, 6), axis=0)))

        chk = df.drop_duplicates("subject_id").set_index("subject_id").loc[subjects, params["class_col"]]
        print("aligned:", np.array_equal(chk.to_numpy(), y), "| index match:", (chk.index == subjects).all())

        from scipy.stats import pearsonr
        r = np.array([pearsonr(X[:, j], y)[0] for j in range(0, X.shape[1], 8)])
        print("max |r| over sampled features:", np.nanmax(np.abs(r)).round(4), flush=True)
    ##################################

    # ----------------------------------------------------------------------------
    # STEP 2 - pipeline, fit on train, predict on val
    # ----------------------------------------------------------------------------
    def build_pipe(use_pca=False, n_components=50, model_name="logreg", model_params=None):
        model = get_sklearn_model(model_name, **(model_params or {}))

        steps = [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
        ]
        if use_pca:
            # PCA is fit on the training subjects only, so cap by n_train, not n
            steps.append(("pca", PCA(n_components=min(n_components, n_train - 1),
                                     random_state=SEED)))
        steps.append((model_name, model))
        return Pipeline(steps)

    def fit_predict(rep_name):
        X = FEATURES[rep_name]
        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]

        pipeline = build_pipe(use_pca=USE_PCA, n_components=N_COMPONENTS,
                              model_name=params["selected_model"],
                              model_params=params["sklearn_model_parameters"])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)

        if hasattr(pipeline, "predict_proba"):
            proba   = pipeline.predict_proba(X_val)
            classes = list(pipeline.classes_)
            # probability of class 0 if it exists, otherwise of the first class
            col = classes.index(0) if 0 in classes else 0
            prob_0 = proba[:, col]
        else:
            print(f"[{rep_name}] model has no predict_proba -> probability_0 set to NaN")
            prob_0 = np.full(len(X_val), np.nan)

        return pd.DataFrame({
            "unique_id":       subjects[val_mask],
            "true_label":      y_val,
            "predicted_label": y_pred,
            "probability_0":   prob_0,
        })

    if COMPARE_ALL:
        return {rep: fit_predict(rep) for rep in FEATURES}
    return fit_predict(REPRESENTATION)

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
    