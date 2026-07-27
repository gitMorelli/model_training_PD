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
from src.debug.PD_model_evaluation import analyze_results, tee_stdout

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/PD/cross_val"
CHECKPOINT_PATH = os.path.join(SOURCE_PATH, "resnet18_model_results/checkpoints")
version='1'
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

    log_path = os.path.join(load_folder, f"stats.txt") #copy prints also to a log file in the checkpoint folder
    with tee_stdout(log_path):
        print("############## Pooled CV analysis ##############")
        analyze_cv_pooled(results, results_df=None,
                        pos_label=1, threshold=None, strategy="youden",
                        target_recall=0.90, plot=True, out_dir_path=load_folder)

        print("\n############## Per-fold CV analysis ##############")
        analyze_cv_per_fold(results, metrics=("pr_auc", "roc_auc", "youden_j"), pos_label=1, threshold=None, strategy="youden",
                            target_recall=0.90, out_dir_path=load_folder)

def _to_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

def analyze_cv_pooled(results, results_df=None, **kwargs):
    # results: dict {fold: DataFrame}  (from your earlier loader)
    frames = [results[f] for f in sorted(results)]
    pooled = pd.concat(frames, ignore_index=True)

    scores, labels = _scores_labels_from_df(pooled)
    return analyze_results(scores, labels, results_df=pooled,
                           split="cv_oof", **kwargs)

def analyze_cv_per_fold(results, metrics=("pr_auc", "roc_auc", "youden_j"), **kwargs):
    per_fold = {}
    for f in sorted(results):
        print(f"\n########## FOLD {f} ##########")
        scores, labels = _scores_labels_from_df(results[f])
        per_fold[f] = analyze_results(scores, labels, results_df=results[f],
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
