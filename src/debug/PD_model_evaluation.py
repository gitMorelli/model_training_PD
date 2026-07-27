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

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/PD/"
CHECKPOINT_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/PD/resnet50_model_results/checkpoints"
version='9'
params_path = os.path.join(CHECKPOINT_PATH,f"v_{version}", "exp_params.pkl")
checkpoint_to_load=f"v_{version}/best-29-0.2294.ckpt"
#open and save as exp_params dict
with open(params_path, 'rb') as f:
    exp_params = pd.read_pickle(f)

#exp_params['filter_missing']='all'
#exp_params['censor_time']='all'

exp_params['predict_on_train'] = False
exp_params['balance_validation'] = False
exp_params['batch_size'] = 1
#'precision': "16-mixed",

exp_params['matched_validation'] = False

#PATHS
SOURCE_PATTERN = os.path.join(SOURCE_PATH,exp_params['data_folder'])
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")
SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")

VERBOSE = False
CLASS_COL = exp_params['class_col'] 


#overrides
#print(exp_params['custom_pre_trained_weights'],flush=True)
#assert 1==0
'''
exp_params['list_of_ids_paths'], exp_params['data_folder'], exp_params['grid_dict_path'] = return_file_paths('PD', False, False)
print(exp_params['custom_pre_trained_weights'],flush=True)
exp_params['custom_pre_trained_weights'] = None'''

def main(exp_params):
    args = get_args()
    worker = args.num_workers
    prefetch_factor = 2 if worker > 0 else None

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
    train_loader,val_loader,_,_= prepare_loaders_PD(worker,prefetch_factor,exp_params,exclusion_set,val_exclusion_set, grid_dict, transform, 
                                                    SHARD_PATTERN_train=SHARD_PATTERN_train, SHARD_PATTERN_val=SHARD_PATTERN_val, train_df=train_df)
    
    if exp_params['matched_validation']:
        matched_val_loader = prepare_balanced_validation(worker,prefetch_factor,exp_params, grid_dict, transform)
    
    # 1. Gather predictions using the best checkpoint saved during training
    # Setting ckpt_path="best" tells Lightning to automatically find your top model
    ckpt_path=os.path.join(CHECKPOINT_PATH,checkpoint_to_load) 
    lit_model = litmodel_initialization_from_checkpoint(model, ckpt_path, exp_params)

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

    log_path = os.path.join(os.path.dirname(ckpt_path), f"stats.txt") #copy prints also to a log file in the checkpoint folder
    with tee_stdout(log_path):
        analyze_results(all_probs, all_labels, results_df, split="validation",
                        pos_label=1, threshold=None, strategy="youden",
                        target_recall=0.90, plot=True, out_dir_path=os.path.dirname(ckpt_path))
        
        if exp_params['matched_validation']:
            print("#" * 50)
            print(f"Evaluating model on matched validation set using checkpoint: {ckpt_path}")
            outputs = trainer.predict(lit_model, dataloaders=matched_val_loader)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
            results_df_matched, all_probs_matched, all_preds_matched, all_labels_matched = get_result_df(outputs)
            analyze_results(all_probs_matched, all_labels_matched, results_df_matched, split="matched_validation",
                            pos_label=1, threshold=None, strategy="youden",
                            target_recall=0.90, plot=True, out_dir_path=os.path.dirname(ckpt_path))

        if exp_params['predict_on_train']:
            outputs = trainer.predict(lit_model, dataloaders=train_loader)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
            results_df_train, all_probs, all_preds, all_labels = get_result_df(outputs)
            analyze_results(all_preds, all_labels, results_df, split="train")

            #concatenate the result dataframes
            results_complete = pd.concat([results_df, results_df_train], ignore_index=True)
            results_df = results_complete.copy()
    
    store_results(csv_data, results_df, ckpt_path,exp_params)

def prepare_balanced_validation(worker,prefetch_factor,exp_params, grid_dict, transform):
    exp_params_temp=exp_params.copy()
    exp_params_temp['balance_validation'] = True
    exp_params_temp['balancing_factor'] = 1.0

    train_df = pd.read_parquet(exp_params_temp['list_of_ids_paths'])
    exclusion_set, val_exclusion_set, counts = prepare_exclusion_sets_PD(exp_params_temp,verbose=VERBOSE,class_col=CLASS_COL)

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
    merged_df.to_csv(os.path.join(os.path.dirname(ckpt_path), f"predictions.csv"), index=False)
    #save params dict as predictions_metadata.pkl
    #save the exp_params dictionary to a pickle file in the checkpoint folder
    with open(os.path.join(os.path.dirname(ckpt_path), f"predictions_metadata.pkl"), 'wb') as f:
        pickle.dump(params, f)

def litmodel_initialization_from_checkpoint(model, ckpt_path, exp_params):
    if exp_params['grouped']:
        model_class=ModelPDGrouped
    else:
        model_class=ModelPDClassification
    lit_model = model_class.load_from_checkpoint(ckpt_path, write_log=False, model=model)
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

def _plot_roc_curve_binary(y_true, y_scores, pos_label=1, threshold=None, path=None):
    if plt is None:
        print("matplotlib not available; skipping ROC plot")
        return
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
    auc = roc_auc_score(y_true, y_scores)
 
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"model (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], ls="--", color="gray", label="random (AUC=0.500)")
 
    # Youden-optimal point on the curve, shown regardless of which strategy was used
    if len(thresholds) > 1:
        j = (tpr - fpr)[1:]
        k = int(np.argmax(j)) + 1
        plt.scatter([fpr[k]], [tpr[k]], facecolors="none", edgecolors="green",
                    s=90, zorder=4,
                    label=f"max Youden J={j[k - 1]:.3f} @ {thresholds[k]:.3f}")
 
    # the operating point actually used
    if threshold is not None:
        y_pred = (y_scores >= threshold).astype(int)
        sens = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        spec = recall_score(y_true, y_pred, pos_label=1 - pos_label, zero_division=0)
        plt.scatter([1 - spec], [sens], color="red", zorder=5,
                    label=f"threshold={threshold:.3f}")
 
    plt.xlabel("false positive rate (1 - specificity)")
    plt.ylabel("true positive rate (sensitivity)")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

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


def _plot_pr_curve_binary(y_true, y_scores, pos_label=1, threshold=None, path=None):
    if plt is None:
        print("matplotlib not available; skipping PR plot")
        return
    precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
    ap = average_precision_score(y_true, y_scores, pos_label=pos_label)
    prevalence = float(np.mean(y_true == pos_label))

    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"model (AP={ap:.3f})")
    plt.axhline(prevalence, ls="--", color="gray", label=f"random (AP={prevalence:.3f})")
    if threshold is not None:
        yp = (y_scores >= threshold).astype(int)
        plt.scatter([recall_score(y_true, yp, pos_label=pos_label, zero_division=0)],
                    [precision_score(y_true, yp, pos_label=pos_label, zero_division=0)],
                    color="red", zorder=5, label=f"threshold={threshold:.3f}")
    plt.xlabel("recall (PD)"); plt.ylabel("precision (PD)")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
    plt.savefig(path) if path else plt.show()


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
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("--- Confusion Matrix @ threshold ---")
    print(confusion_matrix(y_true, y_pred))
 
    # 4. baselines
    print("\n--- Baseline comparison (positive = %s) ---" % class_names[pos_label])
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(_baseline_table_binary(y_true, pos_label).to_string(index=False))
 
    if plot:
        _plot_pr_curve_binary(y_true, y_scores, pos_label, threshold,
                              path=os.path.join(out_dir_path, f"pr_curve_{split}.png"))
        _plot_roc_curve_binary(y_true, y_scores, pos_label, threshold,
                               path=os.path.join(out_dir_path, f"roc_curve_{split}.png"))
    return {"threshold": threshold, "y_pred": y_pred,
            "sensitivity": sens, "specificity": spec, "youden_j": sens + spec - 1,
            "pr_auc": pr_auc, "roc_auc": roc_auc, "prevalence": prevalence}
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
    