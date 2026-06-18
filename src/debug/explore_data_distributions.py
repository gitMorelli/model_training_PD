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

#PATHS
LIST_OF_IDS_PATH = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv"
STATISTICS_PATH = "/home/a_morelli/datasets/handedness/sharded_data_statistics/statistics_all_no_grids_png_whitebg_20260612-182050.csv"
PREDICTIONS_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/resnet18_model_results/checkpoints/v_30/predictions.csv"
OUT_PATH = "/home/a_morelli/vscode_projects/model_training/data/inspect_statistics"
MODEL_SPECIFIC_OUT_PATH = os.path.dirname(PREDICTIONS_PATH)

metadata = {
    "list_of_ids_path": LIST_OF_IDS_PATH,
    "statistics_path": STATISTICS_PATH,
    "predictions_path": PREDICTIONS_PATH,
}


QUESTIONNAIRES = [str(q) for q in range(1,14)]


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

def preview_of_dataframes(matching_ids_df, statistics_df, predictions_df):

    print(len(matching_ids_df), "rows in matching_ids_df")
    print(len(statistics_df), "rows in statistics_df")
    print(len(predictions_df), "rows in predictions_df")
    #print(matching_ids_df["split"].describe())


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
     

    #print id value counts for each dataframe
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

def validity_checks(statistics_df):
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

    left_handed_subject_ids = statistics_df[statistics_df['lateralite'] == 1]['subject_id'].unique()
    rigth_handed_subject_ids = statistics_df[statistics_df['lateralite'] == 0]['subject_id'].unique()
    #get the ratio of left_handed_subject_ids to rigth_handed_subject_ids
    ratio = len(left_handed_subject_ids) / len(rigth_handed_subject_ids)
    print(f"Ratio of left_handed_subject_ids to rigth_handed_subject_ids: {ratio:.4f}")
    
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


def main(show_preview=True,run_validity_checks=True,generate_num_statistics=False, generate_correlations=False, 
         run_metadata_analysis=False, run_confidence_prediction_analysis = False, save_file=True):
    args = get_args()

    matching_ids_df = pd.read_csv(LIST_OF_IDS_PATH)
    statistics_df = pd.read_csv(STATISTICS_PATH)
    predictions_df = pd.read_csv(PREDICTIONS_PATH)

    if show_preview:
        preview_of_dataframes(matching_ids_df, statistics_df, predictions_df)
        print("="*50)

    statistics_df = merge_dfs(matching_ids_df, statistics_df, predictions_df) 
    
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(statistics_df.head())
        print("="*50)

    if run_validity_checks:    
        validity_checks(statistics_df)
        print("="*50)
    
    if generate_num_statistics:
        inspect_num_statistics(statistics_df)
        numerosity_analysis(statistics_df, out_dir=os.path.join(OUT_PATH,'numerosity'))
        #add the numerosity distribution as is per-modality and the numerosity as is per-questionnaire

    if generate_correlations:
        correlation_analysis(statistics_df, out_dir=os.path.join(OUT_PATH,'correlation'))

    statistics_df['area'] = statistics_df['width'] * statistics_df['height']
    #the area values seem off (order of 1e7)
    if run_metadata_analysis:
        metadata_analysis(statistics_df, out_dir=os.path.join(OUT_PATH,'metadata'))

    if run_confidence_prediction_analysis:
        confidence_prediction_analysis(statistics_df, out_dir=os.path.join(OUT_PATH,'confidence_analysis'))

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
    

if __name__ == "__main__":
    main()