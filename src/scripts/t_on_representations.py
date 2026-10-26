import time
import io
from tkinter.font import names
import os
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, field
from typing import Optional, Sequence
import re
from joblib import Parallel, delayed

from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import check_scoring


from src.utils.models import get_sklearn_model
from src.utils.data_loading import prepare_exclusion_sets_PD, return_file_paths, return_dt_path
from src.utils.data_loading import questionnaires_to_keep
from src.utils.evaluation import  analyze_results, plot_probability_trajectories
from sklearn.inspection import permutation_importance
from src.debug.PD_model_evaluation import tee_stdout

params = {

    "type_of_repr": "feature", # representation, feature

    "problem": "PD",
    "class_col": 'diag_park_final1_quest',
    #To put for compatibility with code
    "grouped": False,
    'filter_missing': 'last_q', #all
    'compute_per_step': True,

    "model": "clip-vit-large-patch14-inter",
    "loaded_timestamp": "17092026",
    "seed": 42,

    #model
    "selected_model": "xgb",

    #for representation
    "representation_type": "concat", #concat, mod, mean
    "impute_missing": False,#False leaves the Nan, True imputes with mean 
    "use_pca": False, #requires imputing
    "n_components": 50,
    "n_splits": 5,
    "compare_all": False, #if True it will score all 6 combinations instead of just the one above
    "max_nan_frac":0.95, #features with a fraction of NaN values greater than this threshold will be dropped. 
    "nan_ref": "train", # "train", "all=train+val"

    #feature importance
    "importance":"permutation", # auto = only computes it if the model has it, "permutation" = always compute permutation importance (model agnostic)
    "group_features": True, #if True it will group the features according to the logic in the function make_groups
    "n_repeats":10,

    "balanced_data": False,
    'balance_validation': False, #if True the validation set is balanced, if False it is not balanced
    "balancing_factor": 3,

    'mask_features': True, #if True it will mask the features according to the logic in the function mask_features
    'censor_time': 'pre_diagnosis',#'pre_diagnosis', #'all_matched',#'first_and_last',#'successive','last_successive_and_previous',#'last_and_successive', #'all', 'pre_diagnosis', 'pre_diagnosis_1y', 'last_and_previous','last_and_successive'
    'filter_modality' : 'digit_original', #'X_original' 'digit_original' 'text_original'
}

if params["selected_model"] == "logreg":
    params["sklearn_model_parameters"] = {
            #logreg
            "class_weight": "balanced", #None, 'balanced', {0: 1, 1: 3}
            "C": 1.0,
            "max_iter": 2000,
        }
elif params["selected_model"] == "xgb":
    params["sklearn_model_parameters"] = {
            #xgb
            "scale_pos_weight": 10, 
            "eval_metric": "aucpr", #auc, logloss
            "max_delta_step": 1, #or 1
            "min_child_weight": 1

        }
elif params["selected_model"] == "lgbm":
    params["sklearn_model_parameters"] = {
            #lgbm
            "class_weight": "balanced", #None, 'balanced', {0: 1, 1: 3}
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
    
    reference_df = pd.read_parquet(params['list_of_ids_paths'])
    reference_df = balance_data(reference_df, params, column="unique_id")
    check_available_subjects(reference_df, column="unique_id")

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
    else:
        properties = None
    
    time_taken = time.time() - start
    print(f"----- > Time taken to reshape the data: {time_taken:.2f} seconds", flush=True)
    
    
    if params["mask_features"]:
        df,keep_dict = mask_features(df, params) #masking features for the representation

    time_taken = time.time() - start
    print(f"----- > Time taken to mask the features: {time_taken:.2f} seconds", flush=True)

    df = add_info_to_df(df, params) #adding the class_col to the df

    check_available_subjects(df, column="subject_id")

    print(f"Completed preprocessing", flush=True)

    save_dir = get_save_path(params)
    log_path = os.path.join(save_dir, f"stats.txt") #copy prints also to a log file in the checkpoint folder
    with tee_stdout(log_path):
        results_df, all_probs, all_labels, imp_df, pipeline = process_representation(df, params, verbose=VERBOSE, prop_names=properties)

        analyze_results(all_probs, all_labels, results_df, split="validation",
                            pos_label=1, threshold=None, strategy="youden",
                            target_recall=0.90, plot=True, out_dir_path=save_dir)
        
        if imp_df is not None:
            if params.get("group_features", False):   
                imp_group = imp_df.groupby("group")["importance_mean"].sum().sort_values(ascending=False)
                print(f"Feature importance per group:\n{imp_group}")
            else:
                per_prop = imp_df.groupby("prop")["importance"].sum().sort_values(ascending=False)
                per_key  = imp_df.groupby("key")["importance"].sum().sort_values(ascending=False)
                print(f"Feature importance per property:\n{per_prop}")
                print(f"Feature importance per key:\n{per_key}")
            imp_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
            print(f"Feature importance saved to {os.path.join(save_dir, 'feature_importance.csv')}")
        
        if params.get('compute_per_step',False):
            #[WARNING] I should modify the code so that the per_step results can be computed post_diagnosis even if the model was trained pre_diagnosis
            time_df_path = return_dt_path(params['list_of_ids_paths'])
            time_df = pd.read_parquet(time_df_path)
            results_per_step = process_per_step(pipeline, df, time_df, keep_dict,
                                                params, verbose=VERBOSE, prop_names=properties)
            for n in [5, 10, 15, 20]:
                statistics,diff = plot_probability_trajectories(results_per_step, n_steps=n, ax=None,
                                                    class_names=("negative", "positive"),
                                                    min_count=1, save_path=save_dir, 
                                                    save_name=f"probability_trajectories_{n}", plot_difference=True, show_points=True)
        else:
            results_per_step = None
        save_results(save_dir, params, results_df, results_per_step=results_per_step)

def check_available_subjects(df, column="subject_id"):
    #get the number of unique subject_ids in the train split with the class_col == 0 and with te class col==1
    print(f"[TEST] Number of unique subject_ids in the train split with class_col == 0: {df[(df['split'] == 'train') & (df[params['class_col']] == 0)][column].nunique()}")
    print(f"[TEST] Number of unique subject_ids in the train split with class_col == 1: {df[(df['split'] == 'train') & (df[params['class_col']] == 1)][column].nunique()}")
    print(f"[TEST] Number of unique subject_ids in the val split with class_col == 0: {df[(df['split'] == 'val') & (df[params['class_col']] == 0)][column].nunique()}")
    print(f"[TEST] Number of unique subject_ids in the val split with class_col == 1: {df[(df['split'] == 'val') & (df[params['class_col']] == 1)][column].nunique()}")
    print(f"[TEST] Number of unique subject_ids in the test split with class_col == 0: {df[(df['split'] == 'test') & (df[params['class_col']] == 0)][column].nunique()}")
    print(f"[TEST] Number of unique subject_ids in the test split with class_col == 1: {df[(df['split'] == 'test') & (df[params['class_col']] == 1)][column].nunique()}")

def get_save_path(params):
    save_dir_model = os.path.join(params['source_path'], 'trained_models', params['selected_model'])
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

def save_results(save_dir, params, results_df, results_per_step=None):
    #save the params in a pkl file
    params_path = os.path.join(save_dir, "params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Params saved to {params_path}")

    #save the results in a csv file
    results_path = os.path.join(save_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    if results_per_step is not None:
        results_per_step_path = os.path.join(save_dir, "results_per_step.csv")
        results_per_step.to_csv(results_per_step_path, index=False)
        print(f"Results per step saved to {results_per_step_path}")

def mask_features(df_source, params):
    original_data = pd.read_parquet(params['list_of_ids_paths'])
    # O(1) per-subject lookup instead of a full scan each time
    orig = (original_data.drop_duplicates('unique_id')
                         .set_index('unique_id')
                         .to_dict('index'))

    df = df_source.copy()
    reps = np.stack(df['rep'].to_numpy()).astype(float)   # (N, D)
    row_missing = np.isnan(reps).all(axis=1)

    # --- 1) modality gate, fully vectorized ---
    is_req = (df['modality'] == params['filter_modality']).to_numpy()
    req_present = (pd.Series(is_req & ~row_missing, index=df.index)
                     .groupby([df['subject_id'], df['q']])
                     .transform('any')
                     .to_numpy())
    gate_mask = ~req_present & ~is_req

    # --- 2) questionnaire keep-list: loop only over subjects, no pandas inside ---
    per_subject = dict(tuple(original_data.groupby('unique_id', sort=False)))
    censor_time = params['censor_time']
    subject_images = list(range(1, 14))
    keep_set = set()
    keep_dict = {}
    for sid in df['subject_id'].unique():
        sub = per_subject[sid]          # tiny DataFrame, O(1) lookup
        r = sub.iloc[0]                 # scalar fields come from the same slice
        questionnaire_info = {q: {'case_dt_dateq': r[f'case_dt_dateq{q}']} for q in range(1, 14)}
        keep = questionnaires_to_keep(r['last_avail_q'], censor_time, questionnaire_info,
                                      sub, sid, subject_images,
                                      r['case_grid_pattern'], r['rempli_seulq12'])
        keep_set.update((sid, q) for q in keep)
        keep_dict[sid] = keep

    kept = np.fromiter(((s, q) in keep_set for s, q in zip(df['subject_id'], df['q'])),
                       dtype=bool, count=len(df))

    # --- apply both masks at once, write back once ---
    final_mask = gate_mask | ~kept
    print(f"gate masked {gate_mask.sum()}, keep-list masked {(~kept).sum()}, "
          f"total {final_mask.sum()}/{len(df)}")
    reps[final_mask] = np.nan
    df['rep'] = list(reps)
    return df, keep_dict

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

def balance_data(df, params, verbose=VERBOSE, column = "subject_id"):
    exclusion_set, val_exclusion_set, _ = prepare_exclusion_sets_PD(
        params, verbose=verbose, class_col=params["class_col"])
    
    #remove rows with subject_id in either exclusion_set or val_exclusion_set
    unique_before = df[df['split'] == 'train'][column].nunique()
    unique_before_val = df[df['split'] == 'val'][column].nunique()
    complete_exclusion_set = exclusion_set.union(val_exclusion_set)
    df = df[~df[column].isin(complete_exclusion_set)]
    if verbose:
        print(f"Number of unique ids before exclusion (train split): {unique_before}")
        print(f"Number of unique ids after exclusion (train split): {df[df['split'] == 'train'][column].nunique()}")
        print(f"Number of unique ids before exclusion (val split): {unique_before_val}")
        print(f"Number of unique ids after exclusion (val split): {df[df['split'] == 'val'][column].nunique()}")
    
    return df

def add_info_to_df(df, params):
    original_data = pd.read_parquet(params['list_of_ids_paths'])
    id_column_original = 'unique_id'
    id_column_df = 'subject_id'

    #add the params['class_col'] column to df by merging with original_data on the id columns
    df = df.merge(original_data[[id_column_original, params['class_col']]],
                  left_on=id_column_df, right_on=id_column_original, how='left')
    return df

# ============================================================================
# features

@dataclass
class FeatureBundle:
    """One feature row per subject, plus everything needed to evaluate it."""
    FEATURES: dict                  # rep name -> (n_subjects, n_features)
    names: dict                     # rep name -> (n_features,) array of str
    y: np.ndarray                   # (n_subjects,)
    subjects: np.ndarray            # (n_subjects,) ids, sorted
    split_of_subj: np.ndarray       # (n_subjects,) "train" / "val"
    keys: list                      # column keys, sorted (pre-filter layout)
    mod_of_key: np.ndarray          # modality of each key
    D: int
    dropped: dict = field(default_factory=dict)  # rep name -> pd.Series(nan_frac, index=dropped names)
    max_nan_frac: Optional[float] = None

    @property
    def n(self) -> int:
        return len(self.subjects)

    @property
    def train_mask(self) -> np.ndarray:
        return self.split_of_subj == "train"

    @property
    def val_mask(self) -> np.ndarray:
        return self.split_of_subj == "val"

def drop_nan_columns(X, names, max_nan_frac, ref_mask=None):
    """Drop columns whose NaN fraction exceeds max_nan_frac.

    The fraction is computed over the rows in ref_mask (all rows if None).
    Returns the filtered X, the filtered names, and a Series of the dropped
    names -> NaN fraction, sorted from most to least missing.
    """
    ref = X if ref_mask is None else X[ref_mask]
    nan_frac = np.isnan(ref).mean(axis=0)
    keep = nan_frac <= max_nan_frac          # strictly "more than" is dropped
    dropped = (pd.Series(nan_frac[~keep], index=names[~keep], name="nan_frac")
                 .sort_values(ascending=False))
    return X[:, keep], names[keep], dropped

def build_feature_names(rep_name, keys, mod_of_key, prop_names):
    """Names in the exact column order that build_subject_features produces.

    prop_names[d] describes dimension d of a single `rep` vector.
    """
    prop_names = np.asarray(prop_names, dtype=object)
    if rep_name == "concat":
        # blocks.reshape(n, len(keys) * D) is key-major, then dimension
        return np.array([f"{k}|{p}" for k in keys for p in prop_names])
    if rep_name == "mod":
        # must match the np.unique(mod_of_key) order used to build FEATURES
        return np.array([f"{m}|{p}" for m in np.unique(mod_of_key) for p in prop_names])
    if rep_name == "mean":
        return np.array([f"mean|{p}" for p in prop_names])
    raise ValueError(f"unknown representation: {rep_name}")


def build_subject_features(
    df: pd.DataFrame,
    params: dict,
    prop_names: Optional[Sequence] = None,
    max_nan_frac: Optional[float] = None,
    nan_ref: str = "train",
    verbose: bool = False,
) -> FeatureBundle:
    """...
    max_nan_frac : drop columns with a NaN fraction strictly greater than this
                   (e.g. 0.3). None disables filtering.
    nan_ref      : "train" computes NaN fractions on train subjects only (no
                   peeking at val), "all" uses every subject.
    """
    if max_nan_frac is not None and not 0.0 <= max_nan_frac <= 1.0:
        raise ValueError(f"max_nan_frac must be in [0, 1], got {max_nan_frac}")
    if nan_ref not in ("train", "all"):
        raise ValueError(f"nan_ref must be 'train' or 'all', got {nan_ref!r}")

    D = np.asarray(df["rep"].iloc[0]).shape[0]

    # .copy() avoids SettingWithCopyWarning when adding the "key" column below
    df = df[df["split"].isin(["train", "val"])].copy()
    df["key"] = df["modality"].astype(str) + "_q" + df["q"].astype(str)

    keys = sorted(df["key"].unique())
    subjects = np.sort(df["subject_id"].unique())

    s = (df.drop_duplicates(["subject_id", "key"])
           .set_index(["subject_id", "key"])["rep"])
    s = s.reindex(pd.MultiIndex.from_product([subjects, keys]))

    blocks = np.stack([
        np.asarray(r, dtype=float) if isinstance(r, (np.ndarray, list)) else np.full(D, np.nan)
        for r in s.values
    ])
    Bl = blocks.reshape(len(subjects), len(keys), D)
    n = len(subjects)

    subj_info = (df.drop_duplicates("subject_id")
                   .set_index("subject_id")
                   .loc[subjects])
    y = subj_info[params["class_col"]].to_numpy()

    # A subject appearing in both train and val would leak information.  This
    # check also guarantees one row per subject downstream, which is why the CV
    # evaluator can use StratifiedKFold rather than GroupKFold.
    n_splits_per_subj = df.groupby("subject_id")["split"].nunique()
    leaky = n_splits_per_subj[n_splits_per_subj > 1].index.tolist()
    if leaky:
        raise ValueError(
            f"{len(leaky)} subjects appear in both train and val, e.g. {leaky[:5]}"
        )

    split_of_subj = subj_info["split"].to_numpy()
    mod_of_key = np.array([k.rsplit("_q", 1)[0] for k in keys])

    with np.errstate(invalid="ignore"):
        FEATURES = {
            "concat": blocks.reshape(n, len(keys) * D),
            "mod":    np.concatenate(
                          [np.nanmean(Bl[:, mod_of_key == m, :], axis=1)
                           for m in np.unique(mod_of_key)], axis=1),
            "mean":   np.nanmean(Bl, axis=1),
        }

    # Positional fallback keeps every downstream path uniform: importances are
    # always named, they are just uninformative without real prop_names.
    if prop_names is None:
        prop_names = [f"d{j}" for j in range(D)]
    prop_names = np.asarray(prop_names, dtype=object)
    if len(prop_names) != D:
        raise ValueError(f"prop_names has length {len(prop_names)}, expected D={D}")

    names = {}
    for rep, X in FEATURES.items():
        nm = build_feature_names(rep, keys, mod_of_key, prop_names)
        if len(nm) != X.shape[1]:
            raise AssertionError(f"[{rep}] {len(nm)} names vs {X.shape[1]} columns")
        names[rep] = nm

    # --- NaN-based column filtering ---------------------------------------
    # Done after naming so matrix and names are filtered with the same mask.
    dropped = {}
    if max_nan_frac is not None:
        ref_mask = (split_of_subj == "train") if nan_ref == "train" else None
        if ref_mask is not None and not ref_mask.any():
            raise ValueError("nan_ref='train' but there are no train subjects")
        for rep in list(FEATURES):
            n_before = FEATURES[rep].shape[1]
            X, nm, dr = drop_nan_columns(FEATURES[rep], names[rep], max_nan_frac, ref_mask)
            if X.shape[1] == 0:
                raise ValueError(
                    f"[{rep}] all {n_before} features exceed max_nan_frac={max_nan_frac}"
                )
            FEATURES[rep], names[rep], dropped[rep] = X, nm, dr

    print(f"{n} subjects | {len(keys)} keys | "
          f"train {int((split_of_subj == 'train').sum())} | "
          f"val {int((split_of_subj == 'val').sum())}")
    for k, v in FEATURES.items():
        msg = f"  {k:7s} -> {v.shape[1]:6d} features"
        if max_nan_frac is not None:
            msg += f"  ({len(dropped[k])} dropped, NaN frac > {max_nan_frac:.0%} on {nan_ref})"
        print(msg)
        if verbose and len(dropped.get(k, ())):
            print(dropped[k].head(10).to_string())

    if verbose:
        _run_checks(df, FEATURES, y, subjects, params)

    return FeatureBundle(
        FEATURES=FEATURES, names=names, y=y, subjects=subjects,
        split_of_subj=split_of_subj, keys=keys, mod_of_key=mod_of_key, D=D,
        dropped=dropped, max_nan_frac=max_nan_frac,
    )


def _run_checks(df, FEATURES, y, subjects, params):
    from scipy.stats import pearsonr

    print("Checks ---- >")
    print(pd.Series(y).describe())
    print(pd.Series(y).nunique(), "unique values")

    X = FEATURES["mod"]
    print("NaN fraction:", np.isnan(X).mean())
    print("per-column std - min/median/max:",
          np.nanstd(X, 0).min(), np.median(np.nanstd(X, 0)), np.nanstd(X, 0).max())
    print("duplicate rows:", len(X) - len(np.unique(np.round(X, 6), axis=0)))

    chk = (df.drop_duplicates("subject_id")
             .set_index("subject_id")
             .loc[subjects, params["class_col"]])
    print("aligned:", np.array_equal(chk.to_numpy(), y),
          "| index match:", (chk.index == subjects).all())

    r = np.array([pearsonr(X[:, j], y)[0] for j in range(0, X.shape[1], 8)])
    print("max |r| over sampled features:", np.nanmax(np.abs(r)).round(4), flush=True)
# ============================================================================

# ============================================================================
# Pipeline and importance helpers
def build_pipe(params, n_train, n_features):
    """n_train is now an argument, not a closure variable.

    The PCA cap depends on the size of the *current* training set, which changes
    from fold to fold.  Reading it from an enclosing scope was silently wrong
    under CV rather than an error.
    """
    model_name = params["selected_model"]
    model = get_sklearn_model(model_name, **(params.get("sklearn_model_parameters") or {}))

    steps = []

    if params.get("impute_missing",False):
        steps.append(("impute", SimpleImputer(strategy="mean")))
    
    steps.append(("scale", StandardScaler()))

    if params.get("use_pca"):
        n_comp = min(params["n_components"], n_train - 1, n_features)
        steps.append(("pca", PCA(n_components=n_comp, random_state=params["seed"])))
    steps.append((model_name, model))
    return Pipeline(steps)


def intrinsic_importance(pipeline):
    """Per-input-column importance from the fitted estimator, or None."""
    est = pipeline.steps[-1][1]
    if hasattr(est, "coef_"):
        w = np.atleast_2d(est.coef_)                    # (n_classes, n_out)
    elif hasattr(est, "feature_importances_"):
        w = np.atleast_2d(est.feature_importances_)
    else:
        return None

    if "pca" in pipeline.named_steps:
        # components_ is (n_components, n_features_in): project back, then abs.
        # Doing abs before the projection would be wrong.
        w = w @ pipeline.named_steps["pca"].components_
    return np.abs(w).mean(axis=0)


def _default_scoring(pipeline, y):
    if len(np.unique(y)) == 2 and hasattr(pipeline, "predict_proba"):
        return "roc_auc"
    return "balanced_accuracy"


def perm_importance(pipeline, X, y, seed=0, n_repeats=20, scoring=None, n_jobs=-1):
    """Model-agnostic. Pass raw X with NaNs; the imputer lives in the pipeline."""
    scoring = scoring or _default_scoring(pipeline, y)
    print(f"Computing permutation importance with {n_repeats} repeats, scoring={scoring}")
    r = permutation_importance(
        pipeline, X, y,
        n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs,
        scoring=scoring,
    )
    return r.importances_mean, r.importances_std

def make_groups(columns, patterns, ignore_case=True, add_singletons=False):
    """Build {group_name: [columns]} from {group_name: regex}.

    Example: make_groups(X.columns, {"size": r"size", "color": r"colou?r"})
    With add_singletons=True, every column not matched by any pattern
    becomes its own one-column group, so you can compare groups vs. the rest.
    """
    flags = re.IGNORECASE if ignore_case else 0
    groups = {}
    for name, pat in patterns.items():
        cols = [c for c in columns if re.search(pat, str(c), flags)]
        if not cols:
            raise ValueError(f"No columns match group {name!r} (pattern {pat!r})")
        groups[name] = cols
    if add_singletons:
        used = {c for cols in groups.values() for c in cols}
        groups.update({c: [c] for c in columns if c not in used})
    
    #assert that every group has the same number of columns, otherwise the permutation importance will be biased
    group_sizes = [len(cols) for cols in groups.values()]
    if len(set(group_sizes)) != 1:
        raise ValueError(f"All groups must have the same number of columns, but got group sizes: {group_sizes}")
    return groups


def _group_drops(pipeline, X, y, scorer, cols, n_repeats, seed, baseline):
    rng = np.random.default_rng(seed)
    X_perm = X.copy()
    drops = np.empty(n_repeats)
    for i in range(n_repeats):
        idx = rng.permutation(len(X))
        # Same row permutation for every column in the group:
        # within-group structure is kept, link to target/other features is broken.
        shuffled = X[cols].iloc[idx]
        shuffled.index = X.index  # keep dtypes (incl. categoricals, NaNs)
        X_perm[cols] = shuffled
        drops[i] = baseline - scorer(pipeline, X_perm, y)
    return drops


def group_perm_importance(pipeline, X, y, groups, seed=0, n_repeats=20,
                          scoring=None, n_jobs=-1):
    """Permutation importance for groups of columns.

    pipeline : fitted estimator/pipeline (imputer etc. inside it)
    X        : raw DataFrame (NaNs allowed), ideally a held-out set
    groups   : dict {group_name: list of column names}, e.g. from make_groups
    scoring  : anything sklearn accepts (str, callable scorer, None -> .score)

    Returns a DataFrame sorted by mean importance, plus the raw
    (n_groups, n_repeats) array of score drops.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame so groups can use column names")
    missing = {c for cols in groups.values() for c in cols} - set(X.columns)
    if missing:
        raise KeyError(f"Columns not in X: {sorted(missing)}")

    scorer = check_scoring(pipeline, scoring=scoring)
    baseline = scorer(pipeline, X, y)

    # Independent, reproducible random streams per group
    seeds = np.random.SeedSequence(seed).spawn(len(groups))
    drops = Parallel(n_jobs=n_jobs)(
        delayed(_group_drops)(pipeline, X, y, scorer, list(cols), n_repeats, s, baseline)
        for cols, s in zip(groups.values(), seeds)
    )
    drops = np.vstack(drops)

    result = pd.DataFrame({
        "group": list(groups.keys()),
        "n_columns": [len(c) for c in groups.values()],
        "importance_mean": drops.mean(axis=1),
        "importance_std": drops.std(axis=1),
        "columns": [list(c) for c in groups.values()],
    }).sort_values("importance_mean", ascending=False, ignore_index=True)
    return result, drops

def _predict_frame(pipeline, X, ids, y_true, tag=""):
    y_pred = pipeline.predict(X)
    out = {"unique_id": ids, "true_label": y_true, "predicted_label": y_pred}

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        classes = list(pipeline.classes_)
        if len(classes) == 2:
            col = classes.index(0) if 0 in classes else 0
            out["probability_0"] = proba[:, col]
            out["probability_1"] = 1 - proba[:, col]
        else:
            # prob_1 = 1 - prob_0 is meaningless beyond two classes
            print(f"[{tag}] {len(classes)} classes -> storing per-class columns")
            for j, c in enumerate(classes):
                out[f"probability_{c}"] = proba[:, j]
    else:
        print(f"[{tag}] model has no predict_proba -> probability columns set to NaN")
        out["probability_0"] = np.full(len(X), np.nan)
        out["probability_1"] = np.full(len(X), np.nan)

    return pd.DataFrame(out)


def _split_name_cols(imp_df):
    parts = imp_df["feature"].str.split("|", n=1, expand=True)
    imp_df["key"] = parts[0]
    imp_df["prop"] = parts[1]
    return imp_df
# ============================================================================

# ============================================================================
# Single train/val split
def eval_holdout(bundle: FeatureBundle, params: dict, rep: str, importance="auto", prop_names=None):
    """Respects the `split` column.  Returns (results_df, imp_df)."""
    X, names = bundle.FEATURES[rep], bundle.names[rep]
    #print(names)
    #assert 1==0
    tr, va = bundle.train_mask, bundle.val_mask
    X_train, y_train = X[tr], bundle.y[tr]
    X_val, y_val = X[va], bundle.y[va]

    pipeline = build_pipe(params, n_train=int(tr.sum()), n_features=X.shape[1])
    pipeline.fit(X_train, y_train)

    results_df = _predict_frame(pipeline, X_val, bundle.subjects[va], y_val, tag=rep)

    imp = None if importance == "permutation" else intrinsic_importance(pipeline)
    imp_group=None
    if imp is None:
        print("Computing permutation importance on the holdout set...")
        if params.get("group_features", False):
            print("Grouping features for permutation importance...")
            regex =  {name: re.escape(name) for name in prop_names}
            groups = make_groups(names, regex,
                     add_singletons=True)   # ungrouped columns become their own groups
            #build X as a DataFrame with columns as names
            X_val_df = pd.DataFrame(X_val, columns=names)
            imp_group, drops = group_perm_importance(pipeline, X_val_df, y_val, groups,
                                            scoring=_default_scoring(pipeline, y_val),
                                            n_repeats=params.get("n_repeats", 20))
        else:
            m, s = perm_importance(pipeline, X_val, y_val, seed=params["seed"], n_repeats=params.get("n_repeats", 20))
            imp_df = pd.DataFrame({"feature": names, "importance": m, "std": s})
    else:
        print("Using intrinsic importance from the fitted model...")
        imp_df = pd.DataFrame({"feature": names, "importance": imp})

    if imp_group is not None:
        imp_df = imp_group
    else:
        imp_df = _split_name_cols(imp_df)
        imp_df = imp_df.sort_values("importance", ascending=False, ignore_index=True)
    return results_df, imp_df, pipeline
# ============================================================================

# ============================================================================
# Repeated stratified CV
def eval_cv(bundle: FeatureBundle, params: dict, rep: str,
            n_splits=5, n_repeats=5, importance="auto"):
    """Ignores the `split` column and pools every subject.

    Cross-validating on the train subset only would mean permanently holding out
    data you never look at, so the pooling is deliberate, not an oversight.
    Returns (oof_df, imp_df).
    """
    X, names, y = bundle.FEATURES[rep], bundle.names[rep], bundle.y

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=params["seed"])
    oof, imps = [], []

    for i, (tr, te) in enumerate(cv.split(X, y)):
        pipeline = build_pipe(params, n_train=len(tr), n_features=X.shape[1])
        pipeline.fit(X[tr], y[tr])

        frame = _predict_frame(pipeline, X[te], bundle.subjects[te], y[te], tag=f"{rep}/f{i}")
        frame["fold"] = i % n_splits
        frame["repeat"] = i // n_splits
        oof.append(frame)

        imp = None if importance == "permutation" else intrinsic_importance(pipeline)
        if imp is None:
            imp, _ = perm_importance(pipeline, X[te], y[te],
                                     seed=params["seed"] + i, n_repeats=10)
        imps.append(imp)

    imps = np.vstack(imps)
    # rank 0 = most important within a fold; median rank is far more stable than
    # the mean magnitude at small n
    ranks = np.argsort(np.argsort(-imps, axis=1), axis=1)

    imp_df = pd.DataFrame({
        "feature": names,
        "importance_mean": np.nanmean(imps, axis=0),
        "importance_std": np.nanstd(imps, axis=0),
        "rank_median": np.median(ranks, axis=0),
        "top50_frac": (ranks < 50).mean(axis=0),
    })
    imp_df = _split_name_cols(imp_df)
    imp_df = imp_df.sort_values("rank_median", ignore_index=True)

    return pd.concat(oof, ignore_index=True), imp_df
# ============================================================================


def cv_summary(oof_df, metric="accuracy"):
    """Per-fold scores. Each subject appears n_repeats times in oof_df, so never
    score the pooled frame directly."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    fn = {"accuracy": accuracy_score,
          "balanced_accuracy": balanced_accuracy_score}[metric]

    rows = []
    for (rep_i, fold), g in oof_df.groupby(["repeat", "fold"]):
        row = {"repeat": rep_i, "fold": fold, metric: fn(g.true_label, g.predicted_label)}
        if "probability_1" in g and g.probability_1.notna().all() and g.true_label.nunique() == 2:
            row["roc_auc"] = roc_auc_score(g.true_label, g.probability_1)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================================
# Wrappers
def process_representation(df, params, verbose=False, prop_names=None):
    bundle = build_subject_features(df, params, prop_names=prop_names, verbose=verbose, max_nan_frac=params.get("max_nan_frac", None),
                                    nan_ref=params.get("nan_ref", "train"))
    importance = params.get("importance", "auto")

    print("Dropped features (based on NaN fraction):")
    print(bundle.dropped[params["representation_type"]] if bundle.dropped else "None")

    if params["compare_all"]:
        return {rep: eval_holdout(bundle, params, rep, importance)
                for rep in bundle.FEATURES}

    results_df, imp_df, pipeline = eval_holdout(bundle, params, params["representation_type"], importance, prop_names=prop_names)
    all_labels = results_df["true_label"].to_numpy()
    all_probs = np.stack([results_df["probability_0"].to_numpy(),
                          results_df["probability_1"].to_numpy()], axis=1)
    return results_df, all_probs, all_labels, imp_df, pipeline

def process_per_step(pipeline,df,reference_df,keep_dict, params, verbose=False, prop_names=None):
    bundle = build_subject_features(df, params, prop_names=prop_names, verbose=verbose, max_nan_frac=params.get("max_nan_frac", None),
                                    nan_ref=params.get("nan_ref", "train"))

    print("Dropped features (based on NaN fraction):")
    print(bundle.dropped[params["representation_type"]] if bundle.dropped else "None")

    if params["compare_all"]:
        raise NotImplementedError("compare_all is not implemented for process_per_step")

    rep = params["representation_type"]
    if rep not in ['concat']:
        raise ValueError(f"Representation type {rep} is not supported for process_per_step. ")
    
    X, names = bundle.FEATURES[rep], bundle.names[rep]
    tr, va = bundle.train_mask, bundle.val_mask

    X_val, y_val = X[va], bundle.y[va]

    #i precompute the predictions for each step (prediction is made keeping all available data till that step)
    X_temp = X_val.copy()
    all_results = []
    for i in range(13, 0, -1): 
        #create a copy of X_val in which all features corresponding to q{i} are set to NaN
        q_i_mask = np.array([f"q{i+1}|" in name for name in names])
        X_temp[:, q_i_mask] = np.nan
        results_df = _predict_frame(pipeline, X_temp, bundle.subjects[va], y_val, tag=rep)
        results_df["step"] = i
        all_results.append(results_df)
    all_results_df = pd.concat(all_results, ignore_index=True)

    rows = []
    for i in range(len(X_val)):
        unique_id = bundle.subjects[va][i]
        true_label = y_val[i]
        steps = keep_dict.get(unique_id, [])
        sorted_steps = sorted(steps)
        
        #for each available step return a prediction, then remove that step and repeat
        for q in sorted_steps:
            row_result = all_results_df[(all_results_df["unique_id"] == unique_id) & (all_results_df["step"] == q)]
            row = {
                "unique_id": unique_id,
                "true_label": true_label,
                "q": q,
                "predicted_label": row_result["predicted_label"].values[0],
                "probability_0": row_result["probability_0"].values[0],
                "probability_1": row_result["probability_1"].values[0],
            }
            rows.append(row)
    results_df = pd.DataFrame(rows)

    """Attach case_dt to result_df by matching (unique_id, slot=i) to
    reference_df's case_dt_dateq{i+1}. 
    Rename columns for using the trajectory analysis function"""
    cols = [f"case_dt_dateq{i}" for i in range(1,14)]

    long = (reference_df[["unique_id", *cols]]
            .melt(id_vars="unique_id", value_vars=cols,
                var_name="_col", value_name="case_dt"))
    long["q"] = long["_col"].str.removeprefix("case_dt_dateq").astype(int) - 1

    results_df = results_df.merge(long[["unique_id", "q", "case_dt"]],
                        on=["unique_id", "q"], how="left")

    return results_df
# ============================================================================

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

'''def questionnaires_to_keep_wrapper(r, censor_time, original_data, sid, subject_images):
    questionnaire_info = {q: {'case_dt_dateq': r[f'case_dt_dateq{q}']} for q in range(1, 14)}
    keep = questionnaires_to_keep(r['last_avail_q'], censor_time, questionnaire_info,
                                  original_data, sid, subject_images,
                                  r['case_grid_pattern'], r['rempli_seulq12'])
    return keep'''

if __name__ == "__main__":
    main()
    