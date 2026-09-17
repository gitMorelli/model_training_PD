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


from src.utils.data_loading_utils import melt_df, prepare_exclusion_sets_PD, load_grid_dict, prepare_loaders_PD, return_file_paths
from src.utils.data_loading_utils import prepare_handedness_dataset, prepare_handedness_dataset_all, generate_exclusion_set_val
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, TiledJoinedModels
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_dataset
from src.utils.image_processing import ResizeLongestSide, get_augmentation_transform, get_transforms, get_mu_std
from src.utils.training_utils import BestMetricTracker, ModelPDGrouped, ModelPDClassification, ClearCache
from src.utils.model_utils import SequenceQuestionnaireModel, SetQuestionnaireModel
from src.scripts.train_PD_model import model_initialization

def get_last_best_checkpoint(checkpoint_dir,version):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir,f"v_{version}", "*best*.ckpt"))
    checkpoint_to_load = max(checkpoint_files, key=os.path.getctime) if checkpoint_files else None
    checkpoint_name = os.path.basename(checkpoint_to_load) if checkpoint_to_load else None
    checkpoint_to_load=f"v_{version}/{checkpoint_name}" if checkpoint_name else None
    print(f"Checkpoint to load: {checkpoint_name}", flush=True)
    return checkpoint_to_load

experiment = "PD"#"pre_trained_models/E3N" # "PD"
SOURCE_PATH = f"/home/a_morelli/models/model_training_logs/{experiment}/"
model_name = 'resnet18' #efficientnet_v2_s' #convnext_tiny'#'resnet50'#'FiveStageResidualStridedConvNet' #"FiveStageResidualStridedConvNet"
CHECKPOINT_PATH = f"/home/a_morelli/models/model_training_logs/{experiment}/{model_name}_model_results/checkpoints"
version='40'
override_parameters=True
old_run=False
params_path = os.path.join(CHECKPOINT_PATH,f"v_{version}", "exp_params.pkl")
#get the most recent ckpt file with best in the name 
checkpoint_to_load = get_last_best_checkpoint(CHECKPOINT_PATH,version)
#open and save as exp_params dict
with open(params_path, 'rb') as f:
    exp_params = pd.read_pickle(f) 

prefix=''
if override_parameters:
    print(f"Overriding parameters from {params_path}", flush=True)
    exp_params['censor_time'] = 'pre_diagnosis' #you can test models trained on all also on partial sequences
    prefix='pre_diagnosis'

#exp_params['filter_missing']='all'
#exp_params['censor_time']='all'

exp_params['predict_on_train'] = False
exp_params['balance_validation'] = False
exp_params['batch_size'] = 4
#'precision': "16-mixed",

if exp_params['pre_training']:
    exp_params['matched_validation'] = False
else:
    exp_params['matched_validation'] = True 

#PATHS
SOURCE_PATTERN = os.path.join(SOURCE_PATH,exp_params['data_folder'])
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")
SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")

VERBOSE = False
CLASS_COL = exp_params['class_col'] 

if old_run:
    exp_params['list_of_ids_paths'], exp_params['data_folder'], exp_params['grid_dict_path'] = return_file_paths('PD', False, False)
    print(exp_params['custom_pre_trained_weights'],flush=True) 
    exp_params['custom_pre_trained_weights_old'] = exp_params['custom_pre_trained_weights']
    exp_params['custom_pre_trained_weights'] = None #why do i need this?

def main(exp_params):
    args = get_args()
    worker = args.num_workers
    prefetch_factor = 4 if worker > 0 else None

    #fix all the seeds for reproducibility 
    torch.manual_seed(exp_params['seed'])
    random.seed(exp_params['seed'])
    #with lightning 
    L.seed_everything(exp_params['seed'], workers=True)

    #load grid_files for selecting chunks from the images during the dataloading
    grid_dict = load_grid_dict(exp_params)

    csv_data = pd.read_parquet(exp_params['list_of_ids_paths']) #if dataset is synthetic the list_of_ids_paths
    #is 

    exp_params['norm_mu'],exp_params['norm_std'] = get_mu_std(exp_params, verbose=VERBOSE)

    #exclude controls from the training if i want to reduce the asimmetry of the dataset (for example if i want to have a 1:1 ratio between cases and controls)
    exclusion_set, val_exclusion_set, counts = prepare_exclusion_sets_PD(exp_params,verbose=VERBOSE,class_col=CLASS_COL)

    #_,transform = get_model(name=exp_params['model'], pretrained=True)
    #transform = get_transforms(exp_params, transform)
    model, transform = model_initialization(None,exp_params,verbose=VERBOSE,val=True, **exp_params['model_parameters'])
    
    train_df = pd.read_parquet(exp_params['list_of_ids_paths'])
    val_exclusion_set = override_val_exclusion(train_df, val_exclusion_set, exp_params)

    train_loader,val_loader,_,_= prepare_loaders_PD(worker,prefetch_factor,exp_params,exclusion_set,val_exclusion_set, grid_dict, transform, 
                                                    SHARD_PATTERN_train=SHARD_PATTERN_train, SHARD_PATTERN_val=SHARD_PATTERN_val, train_df=train_df,
                                                    running_eval=True)
    
    if exp_params['matched_validation']:
        matched_val_loader = prepare_balanced_validation(worker,prefetch_factor,exp_params, grid_dict, transform)
    
    # 1. Gather predictions using the best checkpoint saved during training
    # Setting ckpt_path="best" tells Lightning to automatically find your top model
    ckpt_path=os.path.join(CHECKPOINT_PATH,checkpoint_to_load) 
    lit_model = litmodel_initialization_from_checkpoint(model, ckpt_path, exp_params)
    if prefix!='':
        save_path = os.path.join(os.path.dirname(ckpt_path), prefix)
    else:
        save_path = os.path.dirname(ckpt_path)

    tb_logger=False
    # 4. Initialize Trainer and Fit
    trainer = L.Trainer(
        max_epochs=1,
        logger = tb_logger,
        accelerator="auto"                # Automatically selects GPU/CPU/MPu
    )

    outputs = trainer.predict(lit_model, dataloaders=val_loader)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
    results_df, all_probs, all_preds, all_labels = get_result_df(outputs)
    
    print(f"Evaluating model on validation set using checkpoint: {ckpt_path}")

    log_path = os.path.join(save_path, f"stats.txt") #copy prints also to a log file in the checkpoint folder
    with tee_stdout(log_path):
        return_model_info(exp_params)

        analyze_results(all_probs, all_labels, results_df, split="validation",
                        pos_label=1, threshold=None, strategy="youden",
                        target_recall=0.90, plot=True, out_dir_path=save_path)
        
        if exp_params['matched_validation']:
            print("#" * 50)
            print(f"Evaluating model on matched validation set using checkpoint: {ckpt_path}")
            outputs = trainer.predict(lit_model, dataloaders=matched_val_loader)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
            results_df_matched, all_probs_matched, all_preds_matched, all_labels_matched = get_result_df(outputs)
            analyze_results(all_probs_matched, all_labels_matched, results_df_matched, split="matched_validation",
                            pos_label=1, threshold=None, strategy="youden",
                            target_recall=0.90, plot=True, out_dir_path=save_path)
        
        if hasattr(lit_model, 'per_step') and lit_model.per_step: #the trained model returns predictions per step, i can aggregate those
            print("#" * 50)
            print(f"Evaluating model on per_step predictions")
            results_df_per_step = get_per_step_results(outputs, train_df)
            plot_probability_trajectories(results_df_per_step, n_steps=20, ax=None,
                                  class_names=("negative", "positive"),
                                  min_count=1, save_path=save_path)
            #save the per_step results in a csv file
            results_df_per_step.to_csv(os.path.join(save_path, f"per_step_predictions.csv"), index=False)

        if exp_params['predict_on_train']:
            outputs = trainer.predict(lit_model, dataloaders=train_loader)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
            results_df_train, all_probs, all_preds, all_labels = get_result_df(outputs)
            analyze_results(all_preds, all_labels, results_df, split="train")

            #concatenate the result dataframes
            results_complete = pd.concat([results_df, results_df_train], ignore_index=True)
            results_df = results_complete.copy()
    
    store_results(csv_data, results_df, save_path,exp_params)

def override_val_exclusion(train_df, val_exclusion_set, exp_params):
    if exp_params['pre_training']:
        N=2000
        #reduce the number of samples in the validation set to 1000 for pre-training
        #identify the remaining ids after filtering for val_exclusion_set
        val_df = train_df[train_df['split']=='val']
        all_val_ids = set(val_df['unique_id'].unique())
        remaining_ids = set(val_df[~val_df['unique_id'].isin(val_exclusion_set)]['unique_id'].unique())
        #randomly sample N ids from the remaining ids
        sampled_ids = random.sample(list(remaining_ids), N)
        val_exclusion_set = all_val_ids - set(sampled_ids)
        print(f"Reduced the number of samples in the validation set from {len(all_val_ids)} to {N} for pre-training. Excluded {len(val_exclusion_set)} samples.", flush=True)
    return val_exclusion_set

def return_model_info(params):
    print(f"Model properties:")
    print(f"Model name -----> {params['model']}")
    if 'custom_pre_trained_weights_old' in params:
        print(f"Initial weights -----> {params['custom_pre_trained_weights_old']} (old run)")
    else:
        print(f"Initial weights -----> {params['custom_pre_trained_weights']}")
    print(f"Decoder structure -----> {params['model_structure']}")
    print(f"Fine tuning strategy -----> {params['layers_to_unfreeze']}")
    print(f"Data modality -----> {params['data_modality']}")
    print(f"Balancing strategy -----> factor={params['balancing_factor']}, balanced_data={params['balanced_data']}, weighted_loss={params['use_balanced_weights']}, balance_validation={params['balance_validation']}")
    print(f"Selected sequence -----> {params['censor_time']}")
    print('Training parameters:')
    print(f"  use_opt_groups: {params['use_opt_groups']}")
    print(f"  lr_backbone: {params['lr_backbone']}")
    print(f"  lr_classifier_head: {params['lr_classifier_head']}")
    print(f"  lr_scheduling: {params['lr_scheduling']}")
    #print(f"  batch_size: {params['batch_size']}")
    print(f"  num_epochs: {params['num_epochs']}")
    print(f"  patience: {params['patience']}")
    print(f"  eta_min_cosine: {params['eta_min_cosine']}")
    print(f"  weight_decay: {params['weight_decay']}")
    print(f"  warmup_fraction: {params['warmup_fraction']}")
    print(f"Head parameters: {params['model_parameters']}")

def prepare_balanced_validation(worker,prefetch_factor,exp_params, grid_dict, transform):
    exp_params_temp=exp_params.copy()
    exp_params_temp['balance_validation'] = True
    exp_params_temp['balancing_factor'] = 1.0

    train_df = pd.read_parquet(exp_params_temp['list_of_ids_paths'])
    exclusion_set, val_exclusion_set, counts = prepare_exclusion_sets_PD(exp_params_temp,verbose=VERBOSE,class_col=CLASS_COL, exclude_cases=True)

    _,val_loader,_,_= prepare_loaders_PD(worker,prefetch_factor,exp_params_temp,exclusion_set,val_exclusion_set, grid_dict, transform, 
                                                    SHARD_PATTERN_train=SHARD_PATTERN_train, SHARD_PATTERN_val=SHARD_PATTERN_val, train_df=train_df)
    return val_loader

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

@contextlib.contextmanager
def tee_stdout(path, mode="w"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        old = sys.stdout
        sys.stdout = _Tee(old, f)
        try:
            yield
        finally:
            sys.stdout = old

def get_per_step_results(outputs, train_df,class_names=None):
    """Per-timestep predictions from a per_step=True run.

    Returns one row per (subject, slot), sorted subject-major then by slot.
    Empty if the run wasn't per-step (nothing to assemble).
    """
    if not outputs or "tok_probs" not in outputs[0]:
        return pd.DataFrame()

    tok_probs = torch.cat([b["tok_probs"] for b in outputs]).cpu().numpy()
    tok_preds = torch.cat([b["tok_preds"] for b in outputs]).cpu().numpy()

    # seq_ids index into each batch's subject_ids list, so map them per-batch
    # before concatenating — a global cat would collide across batches.
    subj, label, slot = [], [], []
    for b in outputs:
        sids = b["subject_ids"]                            # list, len = B_batch
        labels_b = b["labels"].cpu().numpy()               # (B_batch,)
        seq_b  = b["tok_seq_ids"].cpu().numpy()            # (n_tok_batch,)
        slot_b = b["tok_slot_ids"].cpu().numpy()
        subj.extend(sids[i] for i in seq_b)
        label.extend(labels_b[i] for i in seq_b)
        slot.extend(slot_b.tolist())

    n_classes = tok_probs.shape[1]
    if class_names is None:
        class_names = list(range(n_classes))
    assert len(class_names) == n_classes

    df = pd.DataFrame({
        "unique_id":       subj,
        "slot":            slot,
        "true_label":      label,
        "predicted_label": tok_preds,
    })
    for i, name in enumerate(class_names):
        df[f"probability_{name}"] = tok_probs[:, i]
    
    def add_case_dt(result_df, train_df, n_slots=13):
        """Attach case_dt to result_df by matching (unique_id, slot=i) to
        train_df's case_dt_dateq{i+1}. Missing pairs get NaN."""
        cols = [f"case_dt_dateq{i}" for i in range(1, n_slots + 1)]

        long = (train_df[["unique_id", *cols]]
                .melt(id_vars="unique_id", value_vars=cols,
                    var_name="_col", value_name="case_dt"))
        long["slot"] = long["_col"].str.removeprefix("case_dt_dateq").astype(int) - 1

        return result_df.merge(long[["unique_id", "slot", "case_dt"]],
                            on=["unique_id", "slot"], how="left")
    
    result_df = add_case_dt(df, train_df)

    return result_df.sort_values(["unique_id", "slot"]).reset_index(drop=True)

def plot_probability_trajectories(df, n_steps=20, ax=None,
                                  class_names=("negative", "positive"),
                                  min_count=1, show_points=True,
                                  point_alpha=0.15, save_path=None,
                                  save_name="probability_trajectories", step_is_years=False):
    """Average probability_1 across subjects on a binned case_dt axis.
    Rows are binned to N equal-width steps between min and max case_dt.
    Each class (true_label 0 / 1) gets a mean curve with a ±SEM band.
    Bins with fewer than `min_count` subjects are dropped from that class.
    """
    d = df.dropna(subset=["case_dt", "probability_1", "true_label"]).copy()
    if d.empty:
        raise ValueError("no rows with case_dt, probability_1, and true_label")

    lo, hi = d["case_dt"].min(), d["case_dt"].max()
    if step_is_years:
        delta = hi - lo
        #determine the number of bins in which to divide the range of case_dt into equal-width bins of 1 year (as close as possible)
        #-> n_steps=2y with delta=6y -> n_steps=3, if delta=5y -> n_steps=ceil(5/2)=3
        n_steps = int(np.ceil(delta / n_steps))
        print(f"step_is_years=True, delta={delta}, n_steps={n_steps}")
    width_bin_years = (hi - lo) / n_steps
    print("Bin is {:.2f} years wide, from {:.2f} to {:.2f}".format(width_bin_years, lo, hi))

    edges = np.linspace(lo, hi, n_steps + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    d["_bin"] = np.clip(np.digitize(d["case_dt"], edges) - 1, 0, n_steps - 1)

    grp = d.groupby(["true_label", "_bin"])["probability_1"]
    stats = grp.agg(mean="mean", std="std", n="count").reset_index()
    stats["sem"] = stats["std"] / np.sqrt(stats["n"])
    stats = stats[stats["n"] >= min_count]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    palette = {0: "#3B8BD4", 1: "#D85A30"}
    for label, name in zip((0, 1), class_names):
        s = stats[stats["true_label"] == label].sort_values("_bin")
        if s.empty:
            continue
        x = centers[s["_bin"].to_numpy()]
        m = s["mean"].to_numpy()
        e = s["sem"].fillna(0).to_numpy()
        color = palette[label]

        # raw rows behind the curve — jittered horizontally within each bin
        if show_points:
            raw = d[d["true_label"] == label]
            if not raw.empty:
                bin_w = edges[1] - edges[0]
                rng = np.random.default_rng(label)
                jitter = rng.uniform(-0.3, 0.3, len(raw)) * bin_w
                ax.scatter(centers[raw["_bin"].to_numpy()] + jitter,
                           raw["probability_1"], s=8, color=color,
                           alpha=point_alpha, linewidths=0, zorder=1)

        # ±SEM band with a visible edge, then the mean line, then the markers
        ax.fill_between(x, m - e, m + e, color=color, alpha=0.25,
                        edgecolor=color, linewidth=1.2, zorder=2)
        ax.errorbar(x, m, yerr=e, fmt="o", color=color, ecolor=color,
                    elinewidth=1.4, capsize=3, markersize=6,
                    markeredgecolor="white", markeredgewidth=1,
                    label=f"true_label = {label} ({name})", zorder=3)
        ax.plot(x, m, color=color, lw=1.8, zorder=3)

    ax.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel(f"case_dt (binned), {n_steps} steps, {width_bin_years:.2f} years wide")
    ax.set_ylabel("probability_1")
    pad = 0.02 * (d["probability_1"].max() - d["probability_1"].min() or 1)
    ax.set_ylim(d["probability_1"].min() - pad,
                d["probability_1"].max() + pad)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="best", frameon=False)

    if save_path:
        fig_path = os.path.join(save_path, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved probability trajectories plot to {fig_path}")
    return stats

def plot_probability_trajectories_2(df, n_steps=20, ax=None,
                                  class_names=("negative", "positive"),
                                  min_count=1, show_points=True,
                                  point_alpha=0.15, save_path=None,
                                  save_name="probability_trajectories",
                                  step_is_years=False,
                                  show_bin_edges=True, show_bin_index=True,
                                  max_xticks=21):
    """Average probability_1 across subjects on a binned case_dt axis.
    Rows are binned to N equal-width steps between min and max case_dt.
    Each class (true_label 0 / 1) gets a mean curve with a ±SEM band.
    Raw points keep their true case_dt position; bin edges are drawn on the x axis.
    """
    d = df.dropna(subset=["case_dt", "probability_1", "true_label"]).copy()
    if d.empty:
        raise ValueError("no rows with case_dt, probability_1, and true_label")

    lo, hi = d["case_dt"].min(), d["case_dt"].max()
    if step_is_years:
        delta = hi - lo
        n_steps = int(np.ceil(delta / n_steps))
        print(f"step_is_years=True, delta={delta}, n_steps={n_steps}")
    width_bin_years = (hi - lo) / n_steps
    print("Bin is {:.2f} years wide, from {:.2f} to {:.2f}".format(width_bin_years, lo, hi))

    edges = np.linspace(lo, hi, n_steps + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    d["_bin"] = np.clip(np.digitize(d["case_dt"], edges) - 1, 0, n_steps - 1)

    grp = d.groupby(["true_label", "_bin"])["probability_1"]
    stats = grp.agg(mean="mean", std="std", n="count").reset_index()
    stats["sem"] = stats["std"] / np.sqrt(stats["n"])
    stats = stats[stats["n"] >= min_count]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    # bin boundaries drawn first so everything else sits on top
    if show_bin_edges:
        for e_ in edges:
            ax.axvline(e_, color="gray", lw=0.5, ls=":", alpha=0.35, zorder=0)

    palette = {0: "#3B8BD4", 1: "#D85A30"}
    y_lo, y_hi = d["probability_1"].min(), d["probability_1"].max()
    for label, name in zip((0, 1), class_names):
        s = stats[stats["true_label"] == label].sort_values("_bin")
        if s.empty:
            continue
        x = centers[s["_bin"].to_numpy()]
        m = s["mean"].to_numpy()
        e = s["sem"].fillna(0).to_numpy()
        color = palette[label]

        # raw rows behind the curve, at their actual case_dt (no binning, no jitter)
        if show_points:
            raw = d[d["true_label"] == label]
            if not raw.empty:
                ax.scatter(raw["case_dt"].to_numpy(), raw["probability_1"].to_numpy(),
                           s=8, color=color, alpha=point_alpha,
                           linewidths=0, zorder=1)

        ax.fill_between(x, m - e, m + e, color=color, alpha=0.25,
                        edgecolor=color, linewidth=1.2, zorder=2)
        ax.errorbar(x, m, yerr=e, fmt="o", color=color, ecolor=color,
                    elinewidth=1.4, capsize=3, markersize=6,
                    markeredgecolor="white", markeredgewidth=1,
                    label=f"true_label = {label} ({name})", zorder=3)
        ax.plot(x, m, color=color, lw=1.8, zorder=3)
        y_lo = min(y_lo, (m - e).min())
        y_hi = max(y_hi, (m + e).max())

    ax.axhline(0.5, color="gray", lw=0.7, ls="--", alpha=0.6)

    # major ticks on the bin edges, thinned out if there are too many
    step = max(1, int(np.ceil(len(edges) / max_xticks)))
    tick_edges = edges[::step]
    ax.set_xticks(tick_edges)
    ax.set_xticklabels([f"{t:.2f}" for t in tick_edges], rotation=45, ha="right")

    # minor ticks on the bin centres, labelled with the bin index
    if show_bin_index and n_steps <= 30:
        ax.set_xticks(centers, minor=True)
        ax.set_xticklabels([str(i) for i in range(n_steps)], minor=True)
        ax.tick_params(axis="x", which="minor", length=0, labelsize=7,
                       colors="gray", pad=2)

    span = (hi - lo) or 1
    ax.set_xlim(lo - 0.02 * span, hi + 0.02 * span)

    ax.set_xlabel(f"case_dt — {n_steps} bins of {width_bin_years:.2f} years "
                  f"(ticks = bin edges, small numbers = bin index)")
    ax.set_ylabel("probability_1")
    pad = 0.02 * ((y_hi - y_lo) or 1)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="best", frameon=False)

    if save_path:
        fig_path = os.path.join(save_path, f"{save_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved probability trajectories plot to {fig_path}")
    return stats

def get_result_df(outputs, class_names=None):
    all_probs = torch.cat([batch["probs"] for batch in outputs]).detach().cpu()
    all_preds = torch.cat([batch["preds"] for batch in outputs]).detach().cpu()
    all_labels = torch.cat([batch["labels"] for batch in outputs]).detach().cpu()
    all_subjects = [sid for batch in outputs for sid in batch["subject_ids"]]

    n_classes = all_probs.shape[1]
    if class_names is None:
        class_names = range(n_classes)
    assert len(class_names) == n_classes

    results_df = pd.DataFrame({
        "unique_id": all_subjects,
        "true_label": all_labels.numpy(),
        "predicted_label": all_preds.numpy(),
    })
    probs_np = all_probs.numpy()
    for i, name in enumerate(class_names):
        results_df[f"probability_{name}"] = probs_np[:, i]

    return results_df, all_probs, all_preds, all_labels

def analyze_results_old(all_preds, all_labels, results_df,split="validation"):
    # 3. Convert to numpy arrays for statistics calculation
    y_pred = all_preds.numpy()
    y_true = all_labels.numpy()
    
    print(f"\n================ {split.upper()} STATISTICS ================")
    print("\n--- Classification Report ---")
    # Adjust target_names to match your two classes if needed
    print(classification_report(y_true, y_pred, target_names=["healthy", "PD"]))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))
    print("=======================================================")    

    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results_df.head(10))

def store_results(csv_data, results_df, ckpt_path,params):
    merged_df = pd.merge(csv_data, results_df, on='unique_id', how='left')
    #check for duplicate rows
    if merged_df.duplicated(subset=['unique_id']).any():
        print("Warning: There are duplicate rows in the merged dataframe based on 'unique_id'.")
    else:
        print("No duplicate rows found in the merged dataframe based on 'unique_id'.")
    #save the merged dataframe in a csv file
    merged_df.to_csv(os.path.join(ckpt_path, f"predictions.csv"), index=False)
    #save params dict as predictions_metadata.pkl
    #save the exp_params dictionary to a pickle file in the checkpoint folder
    with open(os.path.join(ckpt_path, f"predictions_metadata.pkl"), 'wb') as f:
        pickle.dump(params, f)

def litmodel_initialization_from_checkpoint(model, ckpt_path, exp_params):
    if exp_params['grouped']:
        model_class=ModelPDGrouped
    else:
        model_class=ModelPDClassification
    lit_model = model_class.load_from_checkpoint(ckpt_path, write_log=False, model=model, strict=False)
    return lit_model
# ======================================================================
# score helpers
# ======================================================================
def _as_pos_scores(all_scores, pos_label=1):
    """Binary only: accept 1D P(pos) OR a 2D (N,2) array, return 1D P(pos)."""
    s = np.asarray(all_scores)
    if s.ndim == 2 and s.shape[1] == 2:
        s = s[:, pos_label]
    return s.ravel()


def _as_prob_matrix(all_scores, num_classes=None):
    """Return an (N, C) probability matrix from various score shapes.

    Accepts:
      - 1D array of P(pos)        -> treated as binary, expanded to (N, 2)
      - 2D (N, C) probabilities   -> used as-is
      - 2D (N, C) logits          -> softmaxed (detected when rows don't sum to 1)

    argmax is invariant to softmax, but the AUC metrics need proper
    probabilities, so we normalize logits here.
    """
    s = np.asarray(all_scores, dtype=float)
    if s.ndim == 1:
        s = np.column_stack([1.0 - s, s])          # binary P(pos) -> (N, 2)

    row_sums = s.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):  # looks like logits -> softmax
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        s = e / e.sum(axis=1, keepdims=True)

    if num_classes is not None and s.shape[1] != num_classes:
        raise ValueError(f"expected {num_classes} columns, got {s.shape[1]}")
    return s

# ======================================================================
# binary path (unchanged behavior) -- used when C == 2
# ======================================================================
def pick_threshold(y_true, y_scores, strategy="f1", target_recall=0.90, pos_label=1):
    """Choose a decision threshold for the positive class (binary only).
 
    strategy="f1"            -> threshold that maximizes F1 on the positive class
    strategy="youden"        -> threshold that maximizes Youden's J = TPR - FPR
                                (equivalently sensitivity + specificity - 1);
                                the point on the ROC curve furthest above the
                                chance diagonal. Weights both classes equally,
                                unlike F1, which ignores true negatives.
    strategy="target_recall" -> best-precision threshold that still hits
                                recall >= target_recall
    """
    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
        # thresholds[0] is +inf (sklearn >= 1.3) or max(score) + 1: the degenerate
        # "predict everything negative" point, where J == 0. Drop it.
        if len(thresholds) > 1:
            fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]
        j = tpr - fpr
        return float(thresholds[np.argmax(j)])
 
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_scores, pos_label=pos_label
    )
    precision, recall = precision[:-1], recall[:-1]
 
    if strategy == "f1":
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall + 1e-12),
            0.0,
        )
        return float(thresholds[np.argmax(f1)])
 
    if strategy == "target_recall":
        ok = recall >= target_recall
        if not ok.any():
            return float(thresholds[np.argmax(recall)])
        idx = np.where(ok)[0]
        best = idx[np.argmax(precision[idx])]
        return float(thresholds[best])
 
    raise ValueError(f"unknown strategy: {strategy}")


def _plot_roc_curve_binary(y_true, y_scores, pos_label=1, threshold=None,
                           path=None, name="model", data_path=None):
    """Plot a ROC curve and optionally dump the curve data to `data_path` (.npz)."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
    auc = float(roc_auc_score(y_true, y_scores))

    # Youden-optimal point on the curve
    if len(thresholds) > 1:
        j = (tpr - fpr)[1:]
        k = int(np.argmax(j)) + 1
        youden_fpr, youden_tpr = float(fpr[k]), float(tpr[k])
        youden_j, youden_thr = float(j[k - 1]), float(thresholds[k])
    else:
        youden_fpr = youden_tpr = youden_j = youden_thr = np.nan

    # the operating point actually used
    if threshold is not None:
        y_pred = (y_scores >= threshold).astype(int)
        sens = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        spec = float(recall_score(y_true, y_pred, pos_label=1 - pos_label, zero_division=0))
        thr_fpr, thr_tpr = 1.0 - spec, sens
    else:
        thr_fpr = thr_tpr = np.nan

    curve = {
        "name": name,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "threshold": np.nan if threshold is None else float(threshold),
        "threshold_fpr": thr_fpr,
        "threshold_tpr": thr_tpr,
        "youden_fpr": youden_fpr,
        "youden_tpr": youden_tpr,
        "youden_j": youden_j,
        "youden_threshold": youden_thr,
    }

    if data_path is not None:
        np.savez_compressed(data_path, **curve)

    if plt is None:
        print("matplotlib not available; skipping ROC plot")
        return curve

    _draw_roc_curves([curve], path=path)
    return curve


def _load_roc_curve(data_path):
    """Load a curve saved by _plot_roc_curve_binary."""
    with np.load(data_path, allow_pickle=False) as d:
        out = {"name": str(d["name"]), "fpr": d["fpr"], "tpr": d["tpr"]}
        for key in ("auc", "threshold", "threshold_fpr", "threshold_tpr",
                    "youden_fpr", "youden_tpr", "youden_j", "youden_threshold"):
            out[key] = float(d[key])
        return out


def _draw_roc_curves(curves, path=None, title=None, show_youden=True):
    """Draw one or more ROC curves (dicts from _plot_roc_curve_binary / _load_roc_curve)."""
    if plt is None:
        print("matplotlib not available; skipping ROC plot")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="random (AUC=0.500)")

    for c in curves:
        line, = ax.plot(c["fpr"], c["tpr"], label=f"{c['name']} (AUC={c['auc']:.3f})")
        color = line.get_color()
        if show_youden and not np.isnan(c["youden_fpr"]):
            ax.scatter([c["youden_fpr"]], [c["youden_tpr"]], facecolors="none",
                       edgecolors=color, s=90, zorder=4,
                       label=f"{c['name']} Youden J={c['youden_j']:.3f} "
                             f"@ {c['youden_threshold']:.3f}")
        if not np.isnan(c["threshold"]):
            ax.scatter([c["threshold_fpr"]], [c["threshold_tpr"]], color=color,
                       edgecolor="black", zorder=5,
                       label=f"{c['name']} thr={c['threshold']:.3f}")

    ax.set_xlabel("false positive rate (1 - specificity)")
    ax.set_ylabel("true positive rate (sensitivity)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    else:
        plt.show()


def _plot_roc_curve_with_previous(y_true, y_scores, prev_data_paths, pos_label=1,
                                  threshold=None, path=None, compare_path=None,
                                  name="exp2", data_path=None, show_youden=False):
    """Plot the current experiment alone, then superimposed with previous runs."""
    curve = _plot_roc_curve_binary(y_true, y_scores, pos_label=pos_label,
                                   threshold=threshold, path=path,
                                   name=name, data_path=data_path)

    if isinstance(prev_data_paths, (str, bytes)):
        prev_data_paths = [prev_data_paths]
    previous = [_load_roc_curve(p) for p in prev_data_paths]

    _draw_roc_curves(previous + [curve], path=compare_path,
                     title="ROC curves comparison", show_youden=show_youden)
    return curve

def _plot_roc_curve_mc(y_true, y_prob, class_names, path=None):
    if plt is None:
        print("matplotlib not available; skipping ROC plot")
        return
    classes = np.arange(len(class_names))
    Y = label_binarize(y_true, classes=classes)
 
    plt.figure(figsize=(6, 6))
    for c in classes:
        if Y[:, c].sum() == 0 or Y[:, c].sum() == len(y_true):
            continue                                  # AUC undefined for that class
        fpr, tpr, _ = roc_curve(Y[:, c], y_prob[:, c])
        auc = roc_auc_score(Y[:, c], y_prob[:, c])
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], ls="--", color="gray", label="random (AUC=0.500)")
 
    plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
    plt.title("One-vs-rest ROC curves")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

def _threshold_free_report_binary(y_true, y_scores, pos_label=1):
    prevalence = float(np.mean(y_true == pos_label))
    pr_auc = average_precision_score(y_true, y_scores, pos_label=pos_label)
    roc_auc = roc_auc_score(y_true, y_scores)

    print("\n--- Threshold-free metrics ---")
    print(f"PR-AUC (avg precision) : {pr_auc:.3f}   "
          f"[random baseline = prevalence = {prevalence:.3f}]")
    print(f"ROC-AUC                : {roc_auc:.3f}   [random baseline = 0.500]")
    return pr_auc, roc_auc, prevalence


def _baseline_table_binary(y_true, pos_label=1, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    prevalence = float(np.mean(y_true == pos_label))

    def row(name, y_hat):
        return {
            "model": name,
            "PD_precision": precision_score(y_true, y_hat, pos_label=pos_label, zero_division=0),
            "PD_recall":    recall_score(y_true, y_hat, pos_label=pos_label, zero_division=0),
            "PD_f1":        f1_score(y_true, y_hat, pos_label=pos_label, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y_true, y_hat),
        }

    neg = 1 - pos_label
    all_healthy = np.full(n, neg, dtype=int)
    all_pd      = np.full(n, pos_label, dtype=int)
    unif        = rng.integers(0, 2, size=n)
    strat       = (rng.random(n) < prevalence).astype(int)

    return pd.DataFrame([
        row("always healthy (majority)", all_healthy),
        row("always PD (minority)",      all_pd),
        row("uniform random 50/50",      unif),
        row("stratified random",         strat),
    ])


#--------------- pr curve ---------
def _plot_pr_curve_binary(y_true, y_scores, pos_label=1, threshold=None,
                          path=None, name="model", data_path=None):
    """Plot a PR curve and optionally dump the curve data to `data_path` (.npz)."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
    ap = float(average_precision_score(y_true, y_scores, pos_label=pos_label))
    prevalence = float(np.mean(y_true == pos_label))

    if threshold is not None:
        yp = (y_scores >= threshold).astype(int)
        thr_recall = float(recall_score(y_true, yp, pos_label=pos_label, zero_division=0))
        thr_precision = float(precision_score(y_true, yp, pos_label=pos_label, zero_division=0))
    else:
        thr_recall = thr_precision = np.nan

    curve = {
        "name": name,
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "prevalence": prevalence,
        "threshold": np.nan if threshold is None else float(threshold),
        "threshold_recall": thr_recall,
        "threshold_precision": thr_precision,
    }

    if data_path is not None:
        np.savez_compressed(data_path, **curve)

    if plt is None:
        print("matplotlib not available; skipping PR plot")
        return curve

    _draw_pr_curves([curve], path=path, prevalence=prevalence)
    return curve


def _load_pr_curve(data_path):
    """Load a curve saved by _plot_pr_curve_binary."""
    with np.load(data_path, allow_pickle=False) as d:
        return {
            "name": str(d["name"]),
            "precision": d["precision"],
            "recall": d["recall"],
            "ap": float(d["ap"]),
            "prevalence": float(d["prevalence"]),
            "threshold": float(d["threshold"]),
            "threshold_recall": float(d["threshold_recall"]),
            "threshold_precision": float(d["threshold_precision"]),
        }


def _draw_pr_curves(curves, path=None, prevalence=None, title=None):
    """Draw one or more curves (dicts from _plot_pr_curve_binary / _load_pr_curve)."""
    if plt is None:
        print("matplotlib not available; skipping PR plot")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    for c in curves:
        line, = ax.plot(c["recall"], c["precision"], label=f"{c['name']} (AP={c['ap']:.3f})")
        if not np.isnan(c["threshold"]):
            ax.scatter([c["threshold_recall"]], [c["threshold_precision"]],
                       color=line.get_color(), edgecolor="black", zorder=5,
                       label=f"{c['name']} thr={c['threshold']:.3f}")

    if prevalence is None:
        prevalence = curves[0]["prevalence"]
    ax.axhline(prevalence, ls="--", color="gray", label=f"random (AP={prevalence:.3f})")

    ax.set_xlabel("recall (PD)"); ax.set_ylabel("precision (PD)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    else:
        plt.show()


def _plot_pr_curve_with_previous(y_true, y_scores, prev_data_paths, pos_label=1,
                                 threshold=None, path=None, compare_path=None,
                                 name="exp2", data_path=None):
    """Plot the current experiment alone, then superimposed with previous runs."""
    curve = _plot_pr_curve_binary(y_true, y_scores, pos_label=pos_label,
                                  threshold=threshold, path=path,
                                  name=name, data_path=data_path)

    if isinstance(prev_data_paths, (str, bytes)):
        prev_data_paths = [prev_data_paths]
    previous = [_load_pr_curve(p) for p in prev_data_paths]

    _draw_pr_curves(previous + [curve], path=compare_path,
                    prevalence=curve["prevalence"], title="PR curves comparison")
    return curve


def _analyze_binary(y_true, y_prob, results_df, split, class_names, pos_label,
                    threshold, strategy, target_recall, plot, out_dir_path):
    y_scores = y_prob[:, pos_label]
 
    # 1. threshold-free view
    pr_auc, roc_auc, prevalence = _threshold_free_report_binary(y_true, y_scores, pos_label)
 
    # 2. pick / apply a threshold
    if threshold is None:
        threshold = pick_threshold(y_true, y_scores, strategy=strategy,
                                   target_recall=target_recall, pos_label=pos_label)
        extra = (", target_recall=%.2f" % target_recall) if strategy == "target_recall" else ""
        print(f"\nChosen threshold ({strategy}{extra}): {threshold:.3f}")
    else:
        print(f"\nUsing fixed threshold: {threshold:.3f}")
    y_pred = (y_scores >= threshold).astype(int)
 
    # 2b. operating point in sensitivity / specificity terms
    sens = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=1 - pos_label, zero_division=0)
    print(f"  sensitivity = {sens:.3f}   specificity = {spec:.3f}   "
          f"Youden's J = {sens + spec - 1:.3f}")
 
    # 3. report at that threshold
    print("\n--- Classification Report @ threshold ---")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("--- Confusion Matrix @ threshold ---")
    print(confusion_matrix(y_true, y_pred))
 
    # 4. baselines
    print("\n--- Baseline comparison (positive = %s) ---" % class_names[pos_label])
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(_baseline_table_binary(y_true, pos_label).to_string(index=False))
 
    dirname = os.path.basename(os.path.normpath(out_dir_path))
    parent_path = os.path.dirname(os.path.normpath(out_dir_path))
    if plot:
        _plot_pr_curve_binary(y_true, y_scores, pos_label, threshold,
                              path=os.path.join(out_dir_path, f"pr_curve_{split}.png"), data_path=os.path.join(out_dir_path, f"pr_curve_{split}.npz"))
        _plot_roc_curve_binary(y_true, y_scores, pos_label, threshold,
                               path=os.path.join(out_dir_path, f"roc_curve_{split}.png"), data_path=os.path.join(out_dir_path, f"roc_curve_{split}.npz"))
        if dirname == "pre_diagnosis":
            _plot_pr_curve_with_previous(y_true, y_scores, prev_data_paths=[os.path.join(parent_path, f"pr_curve_{split}.npz")], pos_label=pos_label,
                                         threshold=threshold, path=os.path.join(out_dir_path, f"pr_curve_{split}.png"), compare_path=os.path.join(out_dir_path, f"pr_curve_{split}_comparison.png"),
                                         name=f"{split} (this run)", data_path=os.path.join(out_dir_path, f"pr_curve_{split}.npz"))
            _plot_roc_curve_with_previous(y_true, y_scores, prev_data_paths=[os.path.join(parent_path, f"roc_curve_{split}.npz")], pos_label=pos_label,
                                         threshold=threshold, path=os.path.join(out_dir_path, f"roc_curve_{split}.png"), compare_path=os.path.join(out_dir_path, f"roc_curve_{split}_comparison.png"),
                                         name=f"{split} (this run)", data_path=os.path.join(out_dir_path, f"roc_curve_{split}.npz"))
    #get the balanced accuracy score from the classification report
    balanced_acc = report["macro avg"]["recall"]
    f1_positive  = report["PD"]["f1-score"]
    precision_positive = report["PD"]["precision"]
    accuracy = report["accuracy"]
    return {"threshold": threshold, "y_pred": y_pred,
            "sensitivity": sens, "specificity": spec, "youden_j": sens + spec - 1,
            "pr_auc": pr_auc, "roc_auc": roc_auc, "prevalence": prevalence, "balanced_acc": balanced_acc, "f1_positive": f1_positive,
            "precision_positive": precision_positive, "accuracy": accuracy}
# ======================================================================
# multiclass path -- used when C > 2
# ======================================================================
def _threshold_free_report_mc(y_true, y_prob, class_names):
    C = y_prob.shape[1]
    classes = np.arange(C)
    prevalence = np.array([(y_true == c).mean() for c in classes])
    Y = label_binarize(y_true, classes=classes)   # (N, C) one-hot

    # per-class one-vs-rest average precision; macro = simple mean
    ap = np.full(C, np.nan)
    for c in classes:
        if Y[:, c].sum() > 0:                      # class present in y_true
            ap[c] = average_precision_score(Y[:, c], y_prob[:, c])
    macro_ap = np.nanmean(ap)

    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                average="macro", labels=classes)
    except ValueError:
        roc_auc = float("nan")                     # a class missing from y_true

    print("\n--- Threshold-free metrics (macro, one-vs-rest) ---")
    print(f"macro PR-AUC : {macro_ap:.3f}   "
          f"[random baseline = mean prevalence = {prevalence.mean():.3f}]")
    print(f"macro ROC-AUC: {roc_auc:.3f}   [random baseline = 0.500]")
    print("  per-class AP:")
    for c in classes:
        print(f"    {class_names[c]:>15s}: AP={ap[c]:.3f}   "
              f"[prevalence={prevalence[c]:.3f}]")
    return macro_ap, roc_auc, prevalence


def _baseline_table_mc(y_true, class_names, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    classes = np.arange(len(class_names))
    prevalence = np.array([(y_true == c).mean() for c in classes])
    majority = int(np.argmax(prevalence))

    def row(name, y_hat):
        return {
            "model": name,
            "macro_precision": precision_score(y_true, y_hat, average="macro", zero_division=0),
            "macro_recall":    recall_score(y_true, y_hat, average="macro", zero_division=0),
            "macro_f1":        f1_score(y_true, y_hat, average="macro", zero_division=0),
            "balanced_acc":    balanced_accuracy_score(y_true, y_hat),
        }

    always_majority = np.full(n, majority, dtype=int)
    unif            = rng.integers(0, len(classes), size=n)
    strat           = rng.choice(classes, size=n, p=prevalence)

    return pd.DataFrame([
        row(f"always majority ({class_names[majority]})", always_majority),
        row("uniform random",    unif),
        row("stratified random", strat),
    ])


def _plot_pr_curve_mc(y_true, y_prob, class_names, path=None):
    if plt is None:
        print("matplotlib not available; skipping PR plot")
        return
    classes = np.arange(len(class_names))
    Y = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(6, 6))
    for c in classes:
        if Y[:, c].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(Y[:, c], y_prob[:, c])
        ap = average_precision_score(Y[:, c], y_prob[:, c])
        plt.plot(recall, precision, label=f"{class_names[c]} (AP={ap:.3f})")
    plt.xlabel("recall"); plt.ylabel("precision")
    plt.title("One-vs-rest PR curves")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
    plt.savefig(path) if path else plt.show()


def _analyze_multiclass(y_true, y_prob, results_df, split, class_names,
                        plot, out_dir_path):
    n_classes = y_prob.shape[1]
    labels = np.arange(n_classes)
 
    # 1. threshold-free view (macro OvR)
    macro_ap, roc_auc, prevalence = _threshold_free_report_mc(y_true, y_prob, class_names)
 
    # 2. predictions via argmax (no threshold in multiclass)
    y_pred = np.argmax(y_prob, axis=1)
 
    # 3. report
    print("\n--- Classification Report (argmax) ---")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=class_names, zero_division=0))
    print("--- Confusion Matrix (rows=true, cols=pred) ---")
    print(confusion_matrix(y_true, y_pred, labels=labels))
 
    # 4. baselines
    print("\n--- Baseline comparison (macro-averaged) ---")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(_baseline_table_mc(y_true, class_names).to_string(index=False))
 
    if plot:
        _plot_pr_curve_mc(y_true, y_prob, class_names,
                          path=os.path.join(out_dir_path, f"pr_curve_{split}.png"))
        _plot_roc_curve_mc(y_true, y_prob, class_names,
                           path=os.path.join(out_dir_path, f"roc_curve_{split}.png"))
    return {"threshold": None, "y_pred": y_pred,
            "pr_auc": macro_ap, "roc_auc": roc_auc}


# ======================================================================
# dispatcher
# ======================================================================
def analyze_results(all_scores, all_labels, results_df, split="validation",
                    class_names=None, pos_label=1, threshold=None, strategy="f1",
                    target_recall=0.90, plot=False, out_dir_path='.'):
    """
    all_scores : per-sample class probabilities.
                 - binary: 1D P(pos) or 2D (N, 2)
                 - multiclass: 2D (N, C) probabilities or logits
                 (NOT hard predictions -- PR-AUC/thresholding need scores.)
    class_names: list of length C. Defaults to ["healthy","PD"] when C==2,
                 else ["class 0", ...].
    strategy   : "f1" | "youden" | "target_recall"  (binary only).
    Binary (C==2): threshold-based analysis, tuned via `strategy`/`threshold`.
                   With plot=True writes pr_curve_{split}.png and roc_curve_{split}.png.
    Multiclass (C>2): argmax-based analysis with macro / per-class metrics;
                      `pos_label`, `threshold`, `strategy` are ignored.
                      With plot=True writes one-vs-rest PR and ROC figures.
    Assumes labels are integers in [0, C-1].
    """
    y_true = all_labels.numpy() if hasattr(all_labels, "numpy") else np.asarray(all_labels)
    y_true = y_true.astype(int).ravel()
    scores = all_scores.numpy() if hasattr(all_scores, "numpy") else all_scores
    y_prob = _as_prob_matrix(scores)
    n_classes = y_prob.shape[1]
 
    if class_names is None:
        class_names = (["healthy", "PD"] if n_classes == 2
                       else [f"class {i}" for i in range(n_classes)])
    if len(class_names) != n_classes:
        raise ValueError(f"class_names has {len(class_names)} entries "
                         f"but scores imply {n_classes} classes")
 
    print(f"\n================ {split.upper()} STATISTICS ================")
 
    if n_classes == 2:
        result = _analyze_binary(y_true, y_prob, results_df, split, class_names,
                                 pos_label, threshold, strategy, target_recall,
                                 plot, out_dir_path)
    else:
        result = _analyze_multiclass(y_true, y_prob, results_df, split,
                                     class_names, plot, out_dir_path)
 
    print("=======================================================")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(results_df.head(10))
    return result

if __name__ == "__main__":
    main(exp_params)
    