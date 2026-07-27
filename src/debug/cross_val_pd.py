import gc
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
import pickle
from pympler import asizeof
import numpy as np
import psutil
from sklearn.model_selection import KFold, StratifiedKFold
from pathlib import Path

from src.utils.data_loading_utils import prepare_loaders_PD, load_grid_dict, synthetic_data_override
from src.utils.data_loading_utils import prepare_PD_dataset, prepare_exclusion_sets_PD, return_file_paths
from src.utils.model_utils import SequenceQuestionnaireModel, SetQuestionnaireModel, load_ln_checkpoint
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_PD, debug_print_batch_meta
from src.utils.image_processing import ResizeLongestSide, PadToSquare, get_augmentation_transform, get_transforms,get_mu_std, ALL_SYNTHETIC_TRANSFORMS
from src.utils.training_utils import BestMetricTracker, ModelPDGrouped, ModelPDClassification, ClearCache, TimeLoader, get_optimization_groups
from src.utils.training_utils import set_automatic_hyperparameters, MemMonitor, BatchTimer, ThroughputMonitor
from src.scripts.train_PD_model import *
from src.debug.PD_model_evaluation import get_result_df, litmodel_initialization_from_checkpoint


SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/PD/"
CHECKPOINT_PATH_SOURCE = "/mnt/beegfs02/scratch/a_morelli/model_training/PD/resnet18_model_results/checkpoints"
version='22'
params_path = os.path.join(CHECKPOINT_PATH_SOURCE,f"v_{version}", "exp_params.pkl")
with open(params_path, 'rb') as f:
    exp_params = pd.read_pickle(f)

exp_params['n_folds'] = 2
exp_params['short_test'] = True
if 'stopping_metric' not in exp_params:
    exp_params['stopping_metric'] = 'val/loss'

def get_source_path():
    if exp_params['problem'] == 'PD' and not exp_params['pre_training']:
        return "/mnt/beegfs02/scratch/a_morelli/model_training/PD/cross_val"
    elif exp_params['problem'] == 'PD' and exp_params['pre_training']:
        return "/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/E3N"
    else:
        raise ValueError(f"Unknown problem type: {exp_params['problem']}")
SOURCE_PATH = get_source_path()
#"/mnt/beegfs02/scratch/a_morelli/model_training/PD/"

# Authomatic settings
#exp_params['list_of_ids_paths'], exp_params['data_folder'], exp_params['grid_dict_path'] = return_file_paths(exp_params['problem'], exp_params['grouped'], exp_params['pre_training'])

SHARD_PATTERN_train = os.path.join(exp_params['data_folder'],"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(exp_params['data_folder'],"val/worker*_shard-*.tar")
SAVE_DEBUG_PATH = "/home/a_morelli/vscode_projects/model_training/data/debug_training"

define_optimization_groups = get_optimization_groups(model_name=exp_params['model'],exp_params=exp_params)
exp_params['optimization_groups'] = define_optimization_groups

OUTPUT_PATH = os.path.join(SOURCE_PATH,f"{exp_params['model']}_model_results")
CHECKPOINT_PATH = os.path.join(OUTPUT_PATH, "checkpoints")
exp_params['CHECKPOINT_PATH'] = CHECKPOINT_PATH
exp_params['OUTPUT_PATH'] = OUTPUT_PATH
exp_params['SOURCE_PATH'] = SOURCE_PATH

DEBUG_IMGS = False
VERBOSE = True

print(exp_params['model'])
    
def main(exp_params):
    args = get_args()
    worker = args.num_workers
    prefetch_factor = exp_params['prefetch_factor'] if worker > 0 else None

    print(f"baseline: {rss_gb():.2f}") #0.81

    validity_checks(exp_params)

    #set seed with lightning for reproducibility
    L.seed_everything(exp_params['seed'], workers=True)

    exp_params['norm_mu'],exp_params['norm_std'] = get_mu_std(exp_params, verbose=VERBOSE)

    exp_params=synthetic_data_override(exp_params, verbose=VERBOSE)
    
    #load grid_files for selecting chunks from the images during the dataloading
    grid_dict = load_grid_dict(exp_params)
    
    print(f"after grid_dict : {rss_gb():.2f}", flush=True) #8.10

    write_log, current_version = logging_initialization_CV() 

    #exclude controls from the training if i want to reduce the asimmetry of the dataset (for example if i want to have a 1:1 ratio between cases and controls)
    exclusion_set_global, val_exclusion_set_global, counts_global = prepare_exclusion_sets_PD(exp_params,verbose=VERBOSE,class_col=exp_params['class_col'])
    train_df_global = pd.read_parquet(exp_params['list_of_ids_paths'])
    exclusion_set_global = set(exclusion_set_global) | set(val_exclusion_set_global)
    del val_exclusion_set_global

    kfold = StratifiedKFold(n_splits=exp_params['n_folds'], shuffle=True, random_state=42)
    #exclude the rows with unique_id in the val_exclusion_set_global and exclusion_set_global from the training dataframe
    train_df_filtered = train_df_global[~train_df_global['unique_id'].isin(exclusion_set_global)]

    if exp_params['short_test']:
        train_df_filtered = train_df_filtered.sample(n=100, random_state=42)
        write_log(f"Short test mode: Using only 20 samples for training and validation.")
        exp_params['num_epochs'] = 1
        exp_params['patience'] = 1
        exp_params['filter_missing'] = 'all'
        print('Censor time was ', exp_params['censor_time'], 'but is now set to all for short test mode.')
        exp_params['censor_time'] = 'all'

    dataset = train_df_filtered['unique_id'].values 
    labels = train_df_filtered[exp_params['class_col']].values
    
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset, labels)):  # labels should be your target variable
        print(f"--- Fold {fold + 1}/{exp_params['n_folds']} ---")
        write_log(f"--- Fold {fold + 1}/{exp_params['n_folds']} ---")

        if fold > 0:
            initialize_fold(os.path.join(CHECKPOINT_PATH,f'v_{current_version}'))

        #get train and validation unique_ids for this fold
        train_ids = dataset[train_idx]
        val_ids = dataset[val_idx]

        #create exclusion sets for this fold
        exclusion_set = set(train_df_global[~train_df_global['unique_id'].isin(train_ids)]['unique_id'].values)
        val_exclusion_set = set(train_df_global[~train_df_global['unique_id'].isin(val_ids)]['unique_id'].values)

        #update train counts for this fold
        train_df_fold = train_df_global[train_df_global['unique_id'].isin(train_ids)]
        counts = (
            train_df_fold[exp_params['class_col']]
            .value_counts()
            .reindex(range(train_df_fold[exp_params['class_col']].max() + 1), fill_value=0)
            .sort_index()
        )

        if fold == 0:
            local_verbose = VERBOSE
        else:
            local_verbose = False  # Only verbose for the first fold
        model, transform = model_initialization(write_log,exp_params,verbose=local_verbose, **exp_params['model_parameters'])

        #get_memory_usage(grid_dict, train_df, exclusion_set)

        train_loader,val_loader,_,_= prepare_loaders_PD(worker,prefetch_factor,exp_params,exclusion_set,val_exclusion_set, 
                                                                        grid_dict, transform, 
                                                                        SHARD_PATTERN_train=SHARD_PATTERN_train, SHARD_PATTERN_val=SHARD_PATTERN_val,
                                                                        train_df=train_df_global,cross_val=True)
        
        if DEBUG_IMGS:
            debug(train_loader,val_loader,exp_params)
        
        lit_model = litmodel_initialization(model,counts,write_log,define_optimization_groups,exp_params, exclusion_set, VERBOSE)

        trainer, metrics_tracker = trainer_definition(current_version, exp_params, cv=fold)
        
        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        results_df = evaluate_best_model_on_fold(current_version, exp_params, val_loader, model)
        results_df.to_csv(os.path.join(CHECKPOINT_PATH,f'v_{current_version}', f'fold_{fold}_results.csv'), index=False)

        if fold == exp_params['n_folds'] - 1:
            exp_params['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            exp_params['current_version'] = current_version
            base_name = os.path.basename(exp_params['list_of_ids_paths'])
            copy_path = os.path.join(CHECKPOINT_PATH,f'v_{current_version}', base_name)
            shutil.copy(exp_params['list_of_ids_paths'], copy_path)
            exp_params['list_of_ids_paths'] = copy_path
    
    # Save normal log
    save_experiment_logs(exp_params)
        
    #save the exp_params dictionary to a pickle file in the checkpoint folder
    with open(os.path.join(exp_params['CHECKPOINT_PATH'],f'v_{current_version}', 'exp_params.pkl'), 'wb') as f:
        pickle.dump(exp_params, f)


def evaluate_best_model_on_fold(current_version, exp_params, val_loader, model):
    folder = Path(os.path.join(CHECKPOINT_PATH,f'v_{current_version}'))
    best_ckpt = next(folder.glob("best*.ckpt"), None)
    lit_model = litmodel_initialization_from_checkpoint(model, best_ckpt, exp_params)

    tb_logger=False
    # 4. Initialize Trainer and Fit
    trainer = L.Trainer(
        max_epochs=1,
        logger = tb_logger,
        accelerator="auto"                # Automatically selects GPU/CPU/MPu
    ) 

    outputs = trainer.predict(lit_model, dataloaders=val_loader)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
    results_df, all_probs, all_preds, all_labels = get_result_df(outputs)
    return results_df

def initialize_fold(dirpath):
    #clean memory
    torch.cuda.empty_cache()
    gc.collect()
    
    d = Path(dirpath)
    if d.exists():
        for ckpt in d.glob("*.ckpt"):
            ckpt.unlink()
     
def logging_initialization_CV():
    #read the current version number (starts from 1)
    current_version=1
    #if it exists open the log file in CHECKPOINT_PATH, else create it 
    log_path = os.path.join(CHECKPOINT_PATH,"experiments_log.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        #get the maximum version number in the log file and add 1 to it for the current experiment
        current_version = df['current_version'].max() + 1
    print(f"Current experiment version: {current_version}")

    #Create the file to log all relevant information
    log_folder = os.path.join(CHECKPOINT_PATH,f"v_{current_version}")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_file = os.path.join(log_folder,f"version_{current_version}_training_log.txt")
    with open(log_file,'w') as f:
        f.write("Log file for experiment version: " + str(current_version) + "\n")
    #define a writign function that creates the file if it doesn't exist and append if it exists and write only if TRAIN==True
    def write_log(message):
        with open(log_file, 'a') as f:
            f.write(message + "\n")

    # Automatically use GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_path = os.environ["HF_HOME"]
    write_log(f"Hugging Face cache directory: {hf_path}")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu = torch.cuda.get_device_name(device_id)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
        print(f"--- GPU DIAGNOSTICS ---")
        print(f"Active GPU: {gpu}")
        print(f"CUDA_VISIBLE_DEVICES: {visible}")
        write_log(f"GPU detected: {gpu} (Device ID: {device_id})")
    return write_log, current_version


if __name__ == "__main__":
    main(exp_params)