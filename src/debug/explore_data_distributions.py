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
# 4. Compute and display metrics using scikit-learn
from sklearn.metrics import classification_report, confusion_matrix
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_numeric_dtype
import seaborn as sns
from scipy import stats
import numpy as np
import html
from functools import cached_property

import src.debug.debug_utils.pd_training_data_analysis as pd_train_analysis
#PATHS

metadata = {
    "experiment": "PD", #'handedness', 'pd'
    "show_preview": True,
    "analyze_training_data": True,
    'analyze_statistics_data': True,

    "run_validity_checks": True,
    "generate_num_statistics": False,
    "generate_correlations": False,
    "run_metadata_analysis": False,
    "run_confidence_prediction_analysis": False,
    "save_file": False,

}

if metadata["experiment"] == "handedness":
    metadata['training_set'] = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv"
    metadata['statistics_path'] = "/home/a_morelli/datasets/handedness/sharded_data_statistics/statistics_all_no_grids_png_whitebg_20260612-182050.csv"
    metadata['predictions_path'] = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/resnet18_model_results/checkpoints/v_30/predictions.csv"
    OUT_PATH = "/home/a_morelli/vscode_projects/model_training/data/inspect_statistics/handedness"
    #MODEL_SPECIFIC_OUT_PATH = os.path.dirname(metadata['predictions_path'])
elif metadata["experiment"] == "PD":
    metadata['training_set'] = "/home/a_morelli/datasets/id_lists/PD_training_set_8_7_26.parquet"
    metadata['statistics_path'] = "/home/a_morelli/datasets/id_lists/statistics/statistics_PD_10072026.csv"
    OUT_PATH = "/home/a_morelli/vscode_projects/model_training/data/inspect_statistics/pd"
    #MODEL_SPECIFIC_OUT_PATH = os.path.dirname(metadata['predictions_path'])

QUESTIONNAIRES = [str(q) for q in range(1,14)]

def main(metadata):
    args = get_args()

    #load files
    training_data, statistics_df, predictions_df = None, None, None
    if 'training_set' in metadata:
        if metadata['training_set'].endswith('.csv'):
            training_data = pd.read_csv(metadata['training_set'])
        elif metadata['training_set'].endswith('.parquet'):
            training_data = pd.read_parquet(metadata['training_set'])
        print(training_data.head())
    if 'statistics_path' in metadata:
        statistics_df = pd.read_csv(metadata['statistics_path'])
    if 'predictions_path' in metadata:
        predictions_df = pd.read_csv(metadata['predictions_path'])

    if metadata["show_preview"]:
        print("Generating preview of dataframes... ")
        preview_dataframes([training_data, training_data[training_data['split']=='train'],training_data[training_data['split']=='val'],
                            statistics_df, predictions_df],
                            ["training_data","training_data_train","training_data_val",
                               "statistics_df", "predictions_df"], 
                            output_dir=os.path.join(OUT_PATH,'dfs_preview'),
                            max_unique_to_list=20, top_n_categories=15,sample_rows=5,force_categorical=['case_control','last_avail_q','at_least_warning','diag_park_final1_quest'],)
    
    if metadata['analyze_training_data']:
        print("Analyzing training data... ")
        if metadata['experiment'] == "PD":
            run_pd_training_data_analysis(training_data,out_path=OUT_PATH)
        elif metadata['experiment'] == "handedness":
            print("Handedness training data analysis has to be re-implemented")

    #merge data from other tables
    #statistics_df = merge_dfs(matching_ids_df, statistics_df, predictions_df) 
    if metadata['analyze_statistics_data']:
        merge_statistics_with_training_data(training_data, statistics_df)
    
    '''
    if save_file:
        out_path = os.path.join(OUT_PATH, "merged_statistics_w_predictions_w_original.csv")
        statistics_df.to_csv(out_path, index=False)
        print(f"Saved merged statistics with predictions to {out_path}")
        out_path = os.path.join(MODEL_SPECIFIC_OUT_PATH, "merged_statistics_w_predictions_w_original.csv")
        statistics_df.to_csv(out_path, index=False)
        print(f"Saved copy of merged statistics in the model folder")
        #save metadata to json
        with open(os.path.join(MODEL_SPECIFIC_OUT_PATH, "merged_statistics_metadata.json"), 'w') as f:
            import json
            json.dump(metadata, f, indent=4)
        print("Saved metadata to json in the model folder")
    '''
    

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()


def melt_df(df,modality,threshold=1):
    exclusion_set = set()
    avail_columns=[f'q_{q}_num_{modality}' for q in QUESTIONNAIRES]
    df_source = df[['ident_projet', 'lateralite','split'] + avail_columns]
    df_long = df_source.melt(
        id_vars=['ident_projet', 'lateralite','split'], 
        value_vars=avail_columns,
        var_name='original_col', 
        value_name='score'
    )
    print(f"Length of melted df before filtering: {len(df_long)}")

    # 3. Extract the 'q' number from the column name
    # This regex looks for 'q_' followed by digits at the start of the string
    df_long['questionnaire'] = df_long['original_col'].str.extract(r'^q_(\d+)_').astype(int)

    df_long['ident_projet'] = df_long['ident_projet'].astype(str) + '_' + df_long['questionnaire'].astype(str)

    # 2. Filter rows where the score/value is >= 1
    df_filtered = df_long[df_long['score'] < threshold]
    ident_projets_to_exclude = set(df_filtered['ident_projet'].unique())
    df_long = df_long[df_long['score'] >= threshold]

    print(f"Length of melted df after filtering: {len(df_long)}")

    # 4. Drop the temporary columns to get your final desired structure
    new_df = df_long[['ident_projet', 'lateralite','split']].reset_index(drop=True)

    return new_df,ident_projets_to_exclude

def describe_df(df, name="df"):
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "first_value": df.iloc[0] if len(df) else None,
        "example_type": [type(df[c].iloc[0]).__name__ if len(df) else None
                         for c in df.columns],
        "n_nulls": df.isna().sum(),
    })
    summary.index.name = f"{name} columns"
    return summary

def plot_distributions(
    df,
    columns=None,
    out_dir="distributions",
    bins=50,
    max_categories=30,
    treat_as_categorical=None,
    save=True,
    show=False,
):
    """
    Plot and (optionally) save the distribution of selected columns.

    Numeric columns -> histogram. Object/low-cardinality columns -> bar chart
    of value counts. Returns {column_name: saved_path}.

    Parameters
    ----------
    columns : list[str] | None      columns to plot (default: all)
    out_dir : str                   directory to write PNGs into
    bins : int                      histogram bins for numeric columns
    max_categories : int            cap on bars shown for categorical columns
    treat_as_categorical : set|None force these columns to bar-chart mode
                                    (useful for int label/category columns)
    save, show : bool               write to disk / display interactively
    """
    columns = list(df.columns) if columns is None else columns
    treat_as_categorical = set(treat_as_categorical or [])
    if save:
        os.makedirs(out_dir, exist_ok=True)

    paths = {}
    for col in columns:
        s = df[col].dropna()
        fig, ax = plt.subplots(figsize=(7, 4))

        is_categorical = (
            col in treat_as_categorical
            or not is_numeric_dtype(s)
            or s.nunique() <= 2  # binary-ish ints -> bars read better
        )

        if s.empty:
            ax.text(0.5, 0.5, "all NaN / empty", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        elif is_categorical:
            vc = s.value_counts()
            clipped = vc.iloc[:max_categories]
            clipped.plot.bar(ax=ax)
            ax.set_ylabel("count")
            if len(vc) > max_categories:
                ax.set_title(f"{col}  (top {max_categories} of {len(vc)})")
            else:
                ax.set_title(col)
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.hist(s, bins=bins)
            ax.set_ylabel("count")
            ax.set_title(
                f"{col}  (n={len(s)}, "
                f"mean={s.mean():.3g}, std={s.std():.3g})"
            )

        # null annotation so you don't forget missing data exists
        n_null = df[col].isna().sum()
        if n_null:
            ax.annotate(
                f"{n_null} NaN ({n_null / len(df):.1%})",
                xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color="firebrick",
            )

        fig.tight_layout()

        if save:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in col)
            path = os.path.join(out_dir, f"{safe}.png")
            fig.savefig(path, dpi=120)
            paths[col] = path
        if show:
            plt.show()
        plt.close(fig)
    '''
    example usage
    # Print distributions 
    # numeric image stats
    plot_distributions(
        statistics_df,
        columns=["width", "height", "ratio", "mean_intensity",
                "ink_density_binary", "probability_1"],
        out_dir=OUT_PATH,
    )

    # label/category columns -> force bar charts even though they're ints
    plot_distributions(
        statistics_df,
        columns=["modality_type", "questionnaire", "true_label",
                "predicted_label", "format", "split"],
        treat_as_categorical={"true_label", "predicted_label"},
        out_dir=OUT_PATH,
    )
    '''

    return paths


def merge_dfs(matching_ids_df, statistics_df, predictions_df):
    #Merge statistics_df with matching_ids_df 
    # Map modality_type values -> column suffix used in matching_ids_df
    modality_map = {
        "x": "X",
        "hand": "text",
        "number_random": "digit",
        "hand_sentences_full": "sent",
    }

    # Index the lookup table by id for fast access
    m = matching_ids_df.set_index("ident_projet")

    # Build the target column name for each row of statistics_df
    q_num = statistics_df["questionnaire"].str.removeprefix("q")      # "q1" -> "1"
    mod   = statistics_df["modality_type"].map(modality_map)          # "x"  -> "X"
    col_names = "q_" + q_num + "_num_" + mod                          # -> "q_1_num_X"


    # Look up m.loc[subject_id, col_name] for each row
    statistics_df["num"] = [
        m.at[sid, col]
        for sid, col in zip(statistics_df["subject_id"], col_names)
    ]
    #map q_n_grid_file_avail t the lines with the corresponding questionnaire and subject_id
    statistics_df["grid_file_avail"] = [
        m.at[sid, f"q_{q}_grid_file_avail"]
        for sid, q in zip(statistics_df["subject_id"], q_num)
    ]
    # grid_file_category is a single column per id -> straight map by subject_id
    statistics_df["grid_file_category"] = statistics_df["subject_id"].map(
        m["grid_file_category"]
    )
    # add the lateralite column to check the label association remained correct
    statistics_df["lateralite"] = statistics_df["subject_id"].map(
        m["lateralite"]
    )

    #Merge predictions_df with matching_ids_df
    # --- 1. Split ident_projet "XX_YY" -> ident_projet "XX" + questionnaire "YY" ---
    parts = predictions_df["ident_projet"].str.rsplit("_", n=1, expand=True)
    predictions_df["ident_projet"]  = parts[0]   # "D3B2E9R1"
    predictions_df["questionnaire"] = parts[1]   # "1"  (overwrites the all-NaN column)

    # --- 2. Add prediction columns to statistics_df, joined on (id, questionnaire) ---
    cols_to_add = ["true_label", "predicted_label", "probability_0", "probability_1", "split"]

    # statistics_df.questionnaire is "q1"; predictions_df.questionnaire is "1" -> normalize
    statistics_df["_q_key"] = statistics_df["questionnaire"].str.removeprefix("q")

    statistics_df = statistics_df.merge(
        predictions_df[["ident_projet", "questionnaire", *cols_to_add]]
            .rename(columns={"ident_projet": "subject_id", "questionnaire": "_q_key"}),
        on=["subject_id", "_q_key"],
        how="left",
    ).drop(columns="_q_key")

    return statistics_df

######## Statistics analysis ######
def merge_statistics_with_training_data(df1, df2):
    # Merge training_data=df1 and statistics_df=df2 to get the split information
    # 1. Check that the merge keys are unique in each table
    assert df1["unique_id"].is_unique, "unique_id is NOT unique in df1"
    assert df2["subject_id"].is_unique, "subject_id is NOT unique in df2"

    # 2. Left merge to preserve ALL rows of df1
    #    suffixes: keep df1's names clean, tag any overlapping df2 columns
    merged = df1.merge(
        df2,
        left_on="unique_id",
        right_on="subject_id",
        how="left",
        suffixes=("", "_df2"),
    )

    # 3. Check the shared 'split' column agrees where a match was found
    #    (rows in df1 with no match in df2 will have NaN in the df2 columns)
    matched = merged[merged["subject_id"].notna()]
    mismatch = matched[matched["split"] != matched["split_df2"]]

    if len(mismatch) > 0:
        print(f"WARNING: 'split' disagrees on {len(mismatch)} rows:")
        print(mismatch[["unique_id", "subject_id", "split", "split_df2"]])
    else:
        print("'split' agrees on all matched rows.")
        # drop the redundant duplicate now that it's verified
        merged = merged.drop(columns=["split_df2"])

    return merged

######## Train analysis #########
def run_pd_training_data_analysis(training_data,out_path):
    # Analyze the training data for PD experiment
    r = pd_train_analysis.Report("training_data analysis")
    p=pd_train_analysis.Profiler(training_data)
    r.add("Subject count", pd_train_analysis.n_subjects(p))
    r.add("Filled periods per subject", pd_train_analysis.filled_periods_per_subject(p))

    r.add("Split fraction", pd_train_analysis.split_fraction(p))
    r.add("Check splitting of ids",pd_train_analysis.ids_in_multiple_splits(p, "unique_id"))
    r.add("Check splitting of groups",pd_train_analysis.group_split_leakage(p))
    r.add("Test contamination via matched groups", pd_train_analysis.test_group_contamination(p))

    r.add("Check if matching was correct", pd_train_analysis.check_matching(p, tol_vars={"relative_age": 1.1},  # add age-band tolerance
                 return_offenders=False) )
    r.add("Group size distribution", pd_train_analysis.group_sizes(p)) 
    res = pd_train_analysis.analyze(p, by=["split"], properties=["id_appearances"])
    r.add("Id appearances within each split", res)
    r.add("Check case-control balance",pd_train_analysis.analyze(p, by=['split'], properties=['case_control_balance']))

    r.add("Statistics on rempli_pattern vs grid_pattern", pd_train_analysis.rempli_vs_grid_mismatch(p, strict=True))
    res = pd_train_analysis.analyze(p, by=["case_control"], properties=["pattern_flags"])
    r.add("Pattern-position counts per case_control", res)

    pd_train_analysis.save_q_num_figures(p, outdir=os.path.join(out_path,'q_num_figures'))
    r.add("q_num missing", pd_train_analysis.q_num_missing(p))   # into a report

    pd_train_analysis.save_case_dt_figures(p, outdir=os.path.join(out_path,'case_dt_figures'))   # {'paths': [...], 'missing': {...}}
    r.add("case_dt missing", pd_train_analysis.case_dt_missing(p))        # registered property -> report

    r.write(os.path.join(out_path,'training_data_analysis.txt'))

######## Preview ################
def preview_dataframes(
    dfs,
    names,
    output_dir=".",
    max_unique_to_list=20,
    top_n_categories=15,
    sample_rows=5,
    force_categorical=None,
):
    """
    Write an exploratory HTML overview of each DataFrame to its own file.

    Parameters
    ----------
    dfs : list of (pd.DataFrame or None)
        DataFrames to preview. None entries are skipped.
    names : list of str
        Output file base-name for each df (same length as `dfs`).
    output_dir : str
        Directory where the reports are written (created if missing).
    max_unique_to_list : int
        For categorical columns, list every unique value only if there are
        at most this many; otherwise show the top-N value counts instead.
    top_n_categories : int
        How many of the most frequent values to show for high-cardinality
        categorical columns.
    sample_rows : int
        Number of rows to include as a head() sample.
    force_categorical : list of str or None
        Column names to treat as categorical even if numeric (e.g. IDs,
        zip codes, encoded labels).

    Returns
    -------
    list of str
        Paths of the HTML files that were written.
    """
    if len(dfs) != len(names):
        raise ValueError(
            f"dfs and names must have equal length "
            f"({len(dfs)} vs {len(names)})."
        )

    os.makedirs(output_dir, exist_ok=True)
    force_categorical = set(force_categorical or [])
    esc = html.escape
    written = []

    template = """<!DOCTYPE html>
    <html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title} — overview</title>
    <style>
    :root {{
        --bg:#ffffff; --fg:#1a1a1a; --muted:#6b7280; --line:#e5e7eb;
        --accent:#4f46e5; --ok:#059669; --warn:#d97706; --bad:#dc2626;
    }}
    * {{ box-sizing:border-box; }}
    body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
        Helvetica,Arial,sans-serif; margin:0; padding:2rem; max-width:1100px;
        margin-inline:auto; color:var(--fg); background:var(--bg); }}
    h1 {{ font-size:1.8rem; margin:0 0 1.2rem; }}
    h2 {{ font-size:1.25rem; margin:2rem 0 .75rem;
        border-bottom:2px solid var(--line); padding-bottom:.35rem; }}
    h3 {{ font-size:1rem; margin:1.4rem 0 .4rem; }}
    code {{ background:#f3f4f6; padding:.1rem .35rem; border-radius:4px;
        font-size:.85em; }}
    .cards {{ display:flex; gap:1rem; flex-wrap:wrap; }}
    .card {{ flex:1; min-width:130px; border:1px solid var(--line);
        border-radius:10px; padding:1rem; text-align:center; }}
    .card-val {{ font-size:1.5rem; font-weight:700; color:var(--accent); }}
    .card-lbl {{ font-size:.8rem; color:var(--muted);
        text-transform:uppercase; letter-spacing:.03em; margin-top:.25rem; }}
    table {{ border-collapse:collapse; width:100%; font-size:.88rem;
        margin:.5rem 0; }}
    th, td {{ text-align:left; padding:.4rem .6rem;
        border-bottom:1px solid var(--line); }}
    th {{ background:#f9fafb; font-weight:600; position:sticky; top:0; }}
    td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .ok {{ color:var(--ok); }} .warn {{ color:var(--warn); }}
    .bad {{ color:var(--bad); font-weight:600; }}
    .badge {{ background:var(--accent); color:#fff; font-size:.7rem;
        padding:.15rem .5rem; border-radius:999px; font-weight:600;
        vertical-align:middle; }}
    .muted {{ color:var(--muted); font-size:.85rem; margin:.2rem 0; }}
    table.vc {{ max-width:640px; }}
    td.bar {{ width:35%; padding-left:0; }}
    td.bar div {{ background:var(--accent); height:12px; border-radius:3px;
        min-width:2px; opacity:.75; }}
    .scroll {{ overflow-x:auto; }}
    .data {{ font-size:.82rem; }}
    .data td, .data th {{ white-space:nowrap; }}
    </style></head>
    <body>{body}</body></html>
    """

    for df, name in zip(dfs, names):
        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"'{name}' is not a DataFrame or None.")

        n = len(df)
        parts = []

        # ---- header / shape --------------------------------------------
        mem = df.memory_usage(deep=True).sum() / 1e6
        parts.append(f"<h1>{esc(str(name))}</h1>")
        parts.append('<div class="cards">')
        for label, value in [
            ("Rows", f"{df.shape[0]:,}"),
            ("Columns", f"{df.shape[1]}"),
            ("Duplicated rows", f"{df.duplicated().sum():,}"),
            ("Memory", f"{mem:.2f} MB"),
        ]:
            parts.append(
                f"<div class='card'><div class='card-val'>{value}</div>"
                f"<div class='card-lbl'>{esc(label)}</div></div>"
            )
        parts.append("</div>")

        # ---- per-column dtype + missingness ----------------------------
        parts.append("<h2>Columns</h2>")
        rows = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            miss_pct = (missing / n * 100) if n else 0
            cls = "ok" if miss_pct == 0 else "warn" if miss_pct < 20 else "bad"
            rows.append(
                f"<tr><td>{esc(str(col))}</td>"
                f"<td><code>{esc(str(df[col].dtype))}</code></td>"
                f"<td class='num'>{n - missing:,}</td>"
                f"<td class='num'>{missing:,}</td>"
                f"<td class='num {cls}'>{miss_pct:.1f}%</td></tr>"
            )
        parts.append(
            "<table><thead><tr><th>Column</th><th>Dtype</th>"
            "<th>Non-null</th><th>Missing</th><th>Missing %</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )

        # ---- split numeric vs categorical ------------------------------
        numeric_cols, categorical_cols = [], []
        for col in df.columns:
            is_num = pd.api.types.is_numeric_dtype(df[col])
            if is_num and col not in force_categorical:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        # ---- numerical description -------------------------------------
        if numeric_cols:
            parts.append("<h2>Numerical variables</h2>")
            desc = df[numeric_cols].describe().T
            desc.insert(0, "missing", df[numeric_cols].isna().sum())
            desc["skew"] = df[numeric_cols].skew(numeric_only=True)
            parts.append(
                desc.to_html(
                    classes="data",
                    float_format=lambda x: f"{x:,.4g}",
                    border=0,
                )
            )

        # ---- categorical variables -------------------------------------
        if categorical_cols:
            parts.append("<h2>Categorical variables</h2>")
            for col in categorical_cols:
                nunique = int(df[col].nunique(dropna=True))
                parts.append(
                    f"<h3>{esc(str(col))} "
                    f"<span class='badge'>{nunique:,} unique</span></h3>"
                )
                vc = df[col].value_counts(dropna=False)
                capped = nunique > max_unique_to_list
                shown = vc.head(top_n_categories) if capped else vc
                if capped:
                    parts.append(
                        f"<p class='muted'>Showing top {top_n_categories} "
                        f"of {nunique:,} values.</p>"
                    )
                vrows = []
                for val, cnt in shown.items():
                    label = "NaN" if pd.isna(val) else str(val)
                    pct = cnt / n * 100 if n else 0
                    vrows.append(
                        f"<tr><td>{esc(label)}</td>"
                        f"<td class='num'>{cnt:,}</td>"
                        f"<td class='num'>{pct:.1f}%</td>"
                        f"<td class='bar'><div style='width:{pct:.1f}%'></div></td>"
                        f"</tr>"
                    )
                parts.append(
                    "<table class='vc'><thead><tr><th>Value</th><th>Count</th>"
                    "<th>%</th><th></th></tr></thead><tbody>"
                    + "".join(vrows) + "</tbody></table>"
                )

        # ---- sample rows -----------------------------------------------
        if sample_rows > 0:
            parts.append(f"<h2>Sample (first {sample_rows} rows)</h2>")
            parts.append('<div class="scroll">')
            parts.append(df.head(sample_rows).to_html(classes="data", border=0))
            parts.append("</div>")

        # ---- write file -------------------------------------------------
        report = template.format(title=esc(str(name)), body="\n".join(parts))
        path = os.path.join(output_dir, f"{name}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        written.append(path)

    return written

#### OUTDATED handedness ####
def preview_of_dataframes_outdated(matching_ids_df, statistics_df, predictions_df):

    print(len(matching_ids_df), "rows in matching_ids_df")
    print(len(statistics_df), "rows in statistics_df")
    print(len(predictions_df), "rows in predictions_df")
    #print(matching_ids_df["split"].describe())

    #Predictions
    print("Preliminary analysis of the predictions_df:")
    val_pred=predictions_df[predictions_df['split']=='val']
    train_pred=predictions_df[predictions_df['split']=='train']
    print(f"predictions_df: {len(val_pred)} rows in val split, {len(train_pred)} rows in train split")
    #count how many rows in train_pred have label==0 and how much have label==1
    train_1_count = len(train_pred[train_pred['true_label']==1])
    train_0_count = len(train_pred[train_pred['true_label']==0])
    print(f"predictions_df: {train_0_count} rows in train split with true_label==0, {train_1_count} rows in train split with true_label==1")
    #count na values
    na_count = train_pred['true_label'].isna().sum()
    print(f"predictions_df: {na_count} rows with true_label==NaN")
    #count how many times each ident_projet appears. Print the unique values of these counts
    value_counts = predictions_df['ident_projet'].value_counts()
    unique_counts = value_counts.unique()
    print("Unique counts of ident_projet in matching_ids_df:", unique_counts)
    print('|'*50)
    #assert 1==0 , "stop here to check the unique counts of ident_projet in matching_ids_df"
     

    #Matching_ids
    print("matching_ids_df subject_id value counts:")
    row_counts = matching_ids_df['ident_projet'].value_counts()
    print(row_counts)
    print("statistics_df subject_id value counts:")
    row_counts = statistics_df['subject_id'].value_counts()
    print(row_counts)
    print(len(statistics_df[statistics_df['subject_id']=='D3B2E9R1'])) #expected 39-6=33 because it has two missing questionnaires 
    #instead i have 29; But i see that 4 are 0 -> 39-6-4=29 -> it is
    #print the row for subject_id D3B2E9R1
    row = matching_ids_df[matching_ids_df['ident_projet']=='D3B2E9R1']
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(row)
    '''for col in row.columns:
        print(col, row[col].values)'''
    '''print("predictions_df ident_projet value counts:")
    row_counts = predictions_df['ident_projet'].value_counts()
    print(row_counts)'''

    #print(predictions_df.describe() ) # quick check that it loaded correctly and has expected columns
    
    for df, name in [(matching_ids_df, "matching_ids_df"), (statistics_df, "statistics_df"), (predictions_df, "predictions_df")]:
        print(describe_df(df, name))
        print()  # spacer between tables

def validity_checks_outdated(statistics_df):
    row_counts = statistics_df['subject_id'].value_counts()
    print(row_counts)

    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(statistics_df[statistics_df['subject_id']=='E4Q3Q8E1'])
    
    #check if there are subject_id, questionnaire, modality_type combinations that are duplicated
    duplicates = statistics_df.duplicated(subset=['subject_id', 'questionnaire', 'modality_type'], keep=False)
    assert not duplicates.any(), f"There are {len(duplicates)} duplicated subject_id, questionnaire, modality_type combinations in statistics_df out of {len(statistics_df)} rows"
    '''if duplicates.any():
        print(f"There are {len(statistics_df[duplicates])} duplicated subject_id, questionnaire, modality_type combinations in statistics_df out of {len(statistics_df)} rows")
        #keep only the first occurrence of each duplicated combination
        statistics_df = statistics_df[~duplicates | ~statistics_df.duplicated(subset=['subject_id', 'questionnaire', 'modality_type'], keep='first')]'''
    
    #show the number o missing values in the columns of interest for the train and val split
    columns_of_interest = ['predicted_label']
    val_missing = statistics_df[statistics_df['split'] == 'val'][columns_of_interest].isna().sum()
    train_missing = statistics_df[statistics_df['split'] == 'train'][columns_of_interest].isna().sum()
    print(f"Missing values in val split:\n{val_missing}")
    print(f"Missing values in train split:\n{train_missing}")
    #show the number of missing values in each column (print column names followed by the number of missing values and teh fraction of missing)
    missing_values = statistics_df.isna().sum()
    missing_values_fraction = missing_values / len(statistics_df)
    print("Missing values in statistics_df:")
    for col, missing, fraction in zip(missing_values.index, missing_values.values, missing_values_fraction.values):
        print(f"{col}: {missing} ({fraction:.2%})")

    #Check that subject_ids only appear in one split
    val_subject_ids = statistics_df[statistics_df['split'] == 'val']['subject_id'].unique()
    train_subject_ids = statistics_df[statistics_df['split'] == 'train']['subject_id'].unique()
    #assert that there is no overlap between val_subject_ids and train_subject_ids
    assert len(set(val_subject_ids).intersection(set(train_subject_ids))) == 0, "There are subject_ids that appear in both val and train splits"

    #Get the ratio of val_subject_ids to train_subject_ids
    ratio = len(val_subject_ids) / len(train_subject_ids)
    print(f"Ratio of val_subject_ids to train_subject_ids: {ratio:.4f}")

    #left handed vs right handed 
    print("#"*50)
    print("#"*50)
    print("Gettng statistics on left-handed vs right-handed in training split")
    temporary_df = statistics_df[statistics_df['split'] == 'train']
    left_handed_subject_ids = temporary_df[temporary_df['lateralite'] == 1]['subject_id'].unique()
    rigth_handed_subject_ids = temporary_df[temporary_df['lateralite'] == 0]['subject_id'].unique()
    #get the ratio of left_handed_subject_ids to rigth_handed_subject_ids
    ratio = len(left_handed_subject_ids) / len(rigth_handed_subject_ids)
    print(f"Number of left_handed_subject_ids: {len(left_handed_subject_ids)}")
    print(f"Number of rigth_handed_subject_ids: {len(rigth_handed_subject_ids)}")
    print(f"Ratio of left_handed_subject_ids to rigth_handed_subject_ids: {ratio:.4f}")
    #Select the digit modality and check how many rows have num <= 0 (should be 0 because statistics is created from datloader
    #and loader cannot load samples with num <= 0)
    less_than_0_df = temporary_df[(temporary_df['modality_type'] == 'number_random') & (temporary_df['num'] <= 0)]
    print(f"Number of rows with num <= 0 for digit modality: {len(less_than_0_df)}")
    #check if there are nan values in the num column for the digit modality
    nan_num_df = temporary_df[(temporary_df['modality_type'] == 'number_random') & (temporary_df['num'].isna())]
    print(f"Number of rows with num == NaN for digit modality: {len(nan_num_df)}")
    #Count the number of rows with lateratlite == 1 and lateralite == 0 and print the number and the ratio
    left_handed_rows = temporary_df[(temporary_df['modality_type'] == 'number_random') & (temporary_df['lateralite'] == 1)]
    right_handed_rows = temporary_df[(temporary_df['modality_type'] == 'number_random') & (temporary_df['lateralite'] == 0)]
    print(f"Number of rows with lateralite == 1: {len(left_handed_rows)}")
    print(f"Number of rows with lateralite == 0: {len(right_handed_rows)}")
    print(f"Ratio of rows with lateralite == 1 to lateralite == 0: {len(left_handed_rows) / len(right_handed_rows):.4f}")
    #repeat for hand column
    less_than_0_df = temporary_df[(temporary_df['modality_type'] == 'hand') & (temporary_df['num'] <= 0)]
    print(f"Number of rows with num <= 0 for hand modality: {len(less_than_0_df)}")
    left_handed_rows = temporary_df[(temporary_df['modality_type'] == 'hand') & (temporary_df['lateralite'] == 1)]
    right_handed_rows = temporary_df[(temporary_df['modality_type'] == 'hand') & (temporary_df['lateralite'] == 0)]
    print(f"Number of rows with lateralite == 1: {len(left_handed_rows)}")
    print(f"Number of rows with lateralite == 0: {len(right_handed_rows)}")
    print(f"Ratio of rows with lateralite == 1 to lateralite == 0: {len(left_handed_rows) / len(right_handed_rows):.4f}")
    print("#"*50)
    print("#"*50)
    
    category_1_df = statistics_df[statistics_df['grid_file_category'] == 1]
    category_2_df = statistics_df[statistics_df['grid_file_category'] == 2]
    category_3_df = statistics_df[statistics_df['grid_file_category'] == 3]
    #get unique ids in each category and check that they are separated
    unique_category_1_ids = category_1_df['subject_id'].unique()
    unique_category_2_ids = category_2_df['subject_id'].unique()
    unique_category_3_ids = category_3_df['subject_id'].unique()
    assert len(set(unique_category_1_ids).intersection(set(unique_category_2_ids))) == 0, "There are subject_ids that appear in both category 1 and category 2"
    assert len(set(unique_category_1_ids).intersection(set(unique_category_3_ids))) == 0, "There are subject_ids that appear in both category 1 and category 3"
    assert len(set(unique_category_2_ids).intersection(set(unique_category_3_ids))) == 0, "There are subject_ids that appear in both category 2 and category 3"

    #check that the category actually represents the availability
    agg_dict = {'grid_file_avail': 'sum'}
    agg_dict['grid_file_category'] = 'first'
    filtered_df = statistics_df[statistics_df['modality_type']=='x'].groupby('subject_id').agg(agg_dict).reset_index()
    category_1_df = filtered_df[filtered_df['grid_file_category'] == 1]
    category_2_df = filtered_df[filtered_df['grid_file_category'] == 2]
    category_3_df = filtered_df[filtered_df['grid_file_category'] == 3]
    print("Category 1:","\n",category_1_df.describe())
    print("Category 2:","\n",category_2_df.describe())
    print("Category 3:","\n",category_3_df.describe()) 

    #assert if the lateralite and label column have the same values 
    #first convert both to integers
    statistics_df['lateralite'] = statistics_df['lateralite'].astype(int)
    statistics_df['label'] = statistics_df['label'].astype(int)
    assert (statistics_df['lateralite'] == statistics_df['label']).all(), "The lateralite and label columns (label from matching df) do not have the same values"
    mask = statistics_df['true_label'].notna()
    statistics_df.loc[mask, 'true_label'] = statistics_df.loc[mask, 'true_label'].astype(int)
    assert (
        statistics_df.loc[mask, 'lateralite'] == statistics_df.loc[mask, 'true_label']
    ).all(), "The lateralite and true_label columns (label from prediction df) do not have the same values"
    #check if the label value is the same for all rows with the same subject_id
    assert statistics_df.groupby('subject_id')['lateralite'].nunique().max() == 1, "There are subject_ids that have multiple label values"

def inspect_num_statistics(statistics_df):
    #Get the properties of the statistics_df for the different gridfile classes
    #kep only the columns that contain num and the grid file category
    columns = statistics_df.columns
    filtered_columns = [col for col in columns if 'num' in col or 'grid_file_category' in col or 'subject_id' in col]
    filtered_df = statistics_df[filtered_columns] 
    #assert that for a given subject_id all rows with that subject_id have the same grid_file_category
    assert filtered_df.groupby('subject_id')['grid_file_category'].nunique().max() == 1, "There are subject_ids that have multiple grid_file_category values"
    #group on subject_id and get the sum of the num columns and keep the grid_file_category column
    agg_dict = {col: 'sum' for col in filtered_df.columns if 'num' in col}
    agg_dict['grid_file_category'] = 'first'
    filtered_df = filtered_df.groupby('subject_id').agg(agg_dict).reset_index()
    category_1_df = filtered_df[filtered_df['grid_file_category'] == 1]
    category_2_df = filtered_df[filtered_df['grid_file_category'] == 2]
    category_3_df = filtered_df[filtered_df['grid_file_category'] == 3]
    print("Category 1:","\n",category_1_df.describe())
    print("Category 2:","\n",category_2_df.describe())
    print("Category 3:","\n",category_3_df.describe()) 

def numerosity_analysis(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    def per_subject(df, by=None, value=None):
        keys = ['subject_id'] + ([by] if by else [])
        agg = (df.groupby(keys).size() if value is None
            else df.groupby(keys)[value].sum())
        return agg.rename('value').reset_index()

    def plot_dist(data, by=None, discrete=True, title='', fname=None):
        if by is None:
            fig = plt.figure(figsize=(7, 4))
            sns.histplot(data['value'], discrete=discrete,
                        bins=(None if discrete else 30))
            plt.xlabel('value per subject_id')
            plt.ylabel('number of subjects')
            plt.title(title)
        else:
            g = sns.displot(data, x='value', col=by, col_wrap=3,
                            discrete=discrete, height=3,
                            facet_kws={'sharey': False})
            g.set_axis_labels('value per subject_id', 'subjects')
            g.figure.suptitle(title, y=1.02)
            fig = g.figure
        plt.tight_layout()
        if fname:
            fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    # ----- counts (number of appearances) -----
    plot_dist(per_subject(df),                       title='1) appearances per subject',
            fname='1_appearances_per_subject.png')
    plot_dist(per_subject(df, by='questionnaire'),   by='questionnaire',
            title='2) appearances per subject, by questionnaire',
            fname='2_appearances_by_questionnaire.png')
    plot_dist(per_subject(df, by='modality_type'),   by='modality_type',
            title='3) appearances per subject, by modality',
            fname='3_appearances_by_modality.png')

    # ----- same, summing num -----
    plot_dist(per_subject(df, value='num'),                     discrete=False,
            title='4a) total num per subject',
            fname='4a_num_per_subject.png')
    plot_dist(per_subject(df, by='questionnaire', value='num'), by='questionnaire',
            discrete=False, title='4b) total num per subject, by questionnaire',
            fname='4b_num_by_questionnaire.png')
    plot_dist(per_subject(df, by='modality_type', value='num'), by='modality_type',
            discrete=False, title='4c) total num per subject, by modality',
            fname='4c_num_by_modality.png')
    
    # ------ Numerosity per modality and questionnaire -----

def correlation_analysis(df,out_dir):
    os.makedirs(out_dir, exist_ok=True)
    def slug(s):
        return ''.join(c if c.isalnum() else '_' for c in str(s))

    def corr_with_label(data, label_col='label' ,title='', fname=None):
        """data: columns 'value' (continuous) and 'label' (0/1)."""
        d = data.dropna(subset=['value', label_col])
        x = d['value'].to_numpy(float)
        y = d[label_col].to_numpy(int)
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        if n0 < 2 or n1 < 2:
            print(f'[skip] {title}: not enough in both classes (0:{n0}, 1:{n1})')
            return None

        r, p = stats.pointbiserialr(y, x)                       # == Pearson(label, num)
        u, p_mw = stats.mannwhitneyu(x[y == 1], x[y == 0], alternative='two-sided')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # panel 1 — distribution difference
        for lab, color in [(0, 'C0'), (1, 'C1')]:
            sns.kdeplot(x[y == lab], ax=axes[0], fill=True, alpha=0.4, color=color,
                        label=f'label {lab} (n={int((y==lab).sum())})')
        axes[0].set(xlabel='num', ylabel='density', title='distribution by label')
        axes[0].legend()

        # panel 2 — correlation plot (logistic fit)
        sns.regplot(x=x, y=y, ax=axes[1], logistic=True, color='C2',
                    scatter_kws={'alpha': 0.3}, y_jitter=0.03)
        axes[1].set(xlabel='num', ylabel='P(label = 1)', title='logistic fit')

        fig.suptitle(f'{title}\npoint-biserial r = {r:.3f} (p = {p:.3g})   |   '
                    f'Mann–Whitney p = {p_mw:.3g}', y=1.06)
        plt.tight_layout()
        if fname:
            fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        return {'r': r, 'p': p, 'p_mannwhitney': p_mw, 'n0': n0, 'n1': n1}
    
    def correlation_w_outcome(outcome_name='label'):
        labels = df.groupby('subject_id')[outcome_name].first()   # one label per subject

        total = (df.groupby('subject_id')['num'].sum().rename('value').reset_index()
            .merge(labels.rename(outcome_name), on='subject_id'))

        res_total = corr_with_label(total,label_col=outcome_name,
            title=f'total num (all modalities & questionnaires) vs {outcome_name}',
            fname=f'corr_total_num_{outcome_name}.png')
        
        per_mod = (df.groupby(['subject_id', 'modality_type'])['num'].sum()
                .rename('value').reset_index()
                .merge(labels.rename(outcome_name), on='subject_id'))

        rows = []
        for mod, sub in per_mod.groupby('modality_type'):
            res = corr_with_label(sub, label_col = outcome_name,title=f'num for modality "{mod}" vs {outcome_name}',
                                fname=f'corr_num_{slug(mod)}_{outcome_name}.png')
            if res:
                rows.append({'modality': mod, **res})

        summary = pd.DataFrame(rows).sort_values('r', key=abs, ascending=False)
        print(summary)   # all coefficients side by side
    
    correlation_w_outcome(outcome_name='label')
    correlation_w_outcome(outcome_name="predicted_label")

def metadata_analysis(df, out_dir):
    COLS = ['ratio', 'mean_intensity', 'ink_density_binary', 'area']
    os.makedirs(out_dir, exist_ok=True)

    def slug(s):
        return ''.join(c if c.isalnum() else '_' for c in str(s))

    def dist_overall(df, col, fname=None):
        fig = plt.figure(figsize=(7, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=40)
        plt.xlabel(col); plt.ylabel('count'); plt.title(f'{col} — overall')
        plt.tight_layout()
        if fname:
            fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.show(); plt.close(fig)
    
    def dist_overlay(df, col, by, fname=None):
        fig = plt.figure(figsize=(8, 4.5))
        for grp, sub in df.dropna(subset=[col]).groupby(by):
            sns.kdeplot(sub[col], label=str(grp), fill=False)
        plt.xlabel(col); plt.ylabel('density'); plt.title(f'{col} — by {by}')
        plt.legend(title=by, fontsize=8)
        plt.tight_layout()
        if fname:
            fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.show(); plt.close(fig)

    def dist_by(df, col, by, fname=None):
        g = sns.displot(df.dropna(subset=[col]), x=col, col=by, col_wrap=3,
                        kde=True, bins=40, height=3, facet_kws={'sharey': False,
                                                                'sharex': False})
        g.set_axis_labels(col, 'count')
        g.figure.suptitle(f'{col} — by {by}', y=1.02)
        plt.tight_layout()
        if fname:
            g.figure.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.show(); plt.close(g.figure)
    
    for col in COLS:
        dist_overall(df, col,                       fname=f'{slug(col)}_overall.png')
        dist_by(df, col, 'modality_type',           fname=f'{slug(col)}_by_modality.png')
        dist_by(df, col, 'questionnaire',           fname=f'{slug(col)}_by_questionnaire.png')

def confidence_prediction_analysis(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    CONF_THRESHOLD = 0.70   # winning prob must exceed this to count as "confident"

    # predicted class: 0 if probability_0 > 0.5 else 1
    df['pred'] = np.where(df['probability_0'] > 0.5, 0, 1)

    # confidence = probability assigned to the predicted class = max of the two
    df['confidence'] = df[['probability_0', 'probability_1']].max(axis=1)

    df['correct']   = df['pred'] == df['label']
    df['confident'] = df['confidence'] >= CONF_THRESHOLD

    # 2x2 category label
    df['category'] = np.select(
        [ df['correct'] &  df['confident'],
        df['correct'] & ~df['confident'],
        ~df['correct'] &  df['confident'],
        ~df['correct'] & ~df['confident']],
        ['confidently correct',
        'unconfidently correct',
        'confidently incorrect',
        'unconfidently incorrect'])
    print(df['category'].value_counts(), '\n')

    ct = pd.crosstab(df['correct'], df['confident'],
                    rownames=['correct'], colnames=['confident'], margins=True)
    print(ct)

    worst = df[df['category'] == 'confidently incorrect'].sort_values('confidence',
                                                                  ascending=False)
    print(f'{len(worst)} confidently incorrect rows')
    worst.head(10)

    fig = plt.figure(figsize=(8, 4.5))
    sns.histplot(data=df, x='confidence', hue='correct', bins=30,
                element='step', stat='count', common_norm=False)
    plt.axvline(CONF_THRESHOLD, color='k', ls='--', lw=1, label='threshold')
    plt.xlabel('confidence  (max prob)'); plt.title('confidence by correctness')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confidence_by_correctness.png'), dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main(metadata)