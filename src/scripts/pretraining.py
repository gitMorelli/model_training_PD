from __future__ import annotations
import tarfile
import time
import io
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
from pathlib import Path
import yaml
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
import math
from peft import LoraConfig, get_peft_model

from src.utils.data_loading import prepare_loaders_PD, load_grid_dict, synthetic_data_override
from src.utils.data_loading import prepare_PD_dataset, prepare_exclusion_sets_PD, return_file_paths
from src.utils.models import SequenceQuestionnaireModel, SetQuestionnaireModel, load_ln_checkpoint, FlexibleSequenceQuestionnaireModel, ModularModel
from src.utils.models import get_model, test_output, unfreeze_layers
from src.utils.visualization import debug_images_PD, debug_print_batch_meta
from src.utils.image_processing import ResizeLongestSide, PadToSquare, get_augmentation_transform, get_transforms,get_mu_std, ALL_SYNTHETIC_TRANSFORMS
from src.utils.image_processing import ALL,NO_AUG,PHOTO_ONLY,GEOM_ONLY
from src.utils.tabular_data_utils import preprocess_PD_csv

from src.scripts.train_PD_model import get_input_modality, _resolve_precision, _selected_properties_get_dims, override_val_exclusion 
from src.utils.pd_pretrain import load_config, instantiate
from src.utils.training_utils import set_automatic_hyperparameters



def main():
    #load the config file
    cfg = load_config()  # plain nested dict: cfg["model"]["init_args"]["lr"], cfg["trainer"]["max_epochs"], ...
    exp_params = cfg.get("experiment", {})
    exp_params = _set_fixed_exp_parameters(exp_params)

    RESTORE = False
    VERBOSE = True
    version_override =  7
    source_path = "/home/a_morelli/models/model_training_logs/pre_trained_models/unsupervised"
    exp_params['list_of_ids_paths'], exp_params['data_folder'], exp_params['grid_dict_path'] = return_file_paths(exp_params['problem'], exp_params['grouped'], exp_params['pre_training'])
    SHARD_PATTERN_train = os.path.join(exp_params['data_folder'],"train/worker*_shard-*.tar")
    SHARD_PATTERN_val = os.path.join(exp_params['data_folder'],"val/worker*_shard-*.tar")
    SAVE_DEBUG_PATH = "/home/a_morelli/vscode_projects/model_training/data/debug_training"

    #read the cfg file
    exp_name = exp_params.get("name", "default")
    num_workers = cfg.get("num_workers", 4)
    prefetch_factor = exp_params['prefetch_factor'] if num_workers > 0 else None
    model_name = cfg["model"]["class_path"].rsplit(".", 1)[-1]
    image_size = cfg["model"]["init_args"].get("image_size", exp_params['input_size']) #if image_size is specfied in the yaml it should match the one in exp_params['input_size'] otherwise it will be set to the one in exp_params
    exp_params['data_modality'] = get_input_modality(exp_params['data_modality_name'])

    exp_params = set_automatic_hyperparameters(exp_params)

    #set variables
    root_path = os.path.join(source_path, exp_name)
    torch.set_float32_matmul_precision("high")
    L.seed_everything(exp_params['seed'], workers=True)
    exp_params['norm_mu'],exp_params['norm_std'] = get_mu_std(exp_params, verbose=VERBOSE)
    MODALITIES = _get_unique_modalities(exp_params['data_modality'])
    exp_params['mod_to_id'] = {name: i for i, name in enumerate(MODALITIES)}
    cfg["model"]["init_args"]["modalities"] = list(MODALITIES)
    cfg["trainer"]["precision"] = _resolve_precision(cfg["trainer"]["precision"])
    

    #prepare variables
    grid_dict = load_grid_dict(exp_params)
    train_df = pd.read_parquet(exp_params['list_of_ids_paths'])
    train_df, *_ = preprocess_PD_csv(train_df,exp_params['selected_properties'], split_col='split')
    n_local,n_global = _selected_properties_get_dims(exp_params, train_df)
    print(f"[DEBUG] n_local: {n_local}, n_global: {n_global}", flush=True)
    exclusion_set, val_exclusion_set, counts = prepare_exclusion_sets_PD(exp_params,verbose=VERBOSE,class_col=exp_params['class_col'])
    cfg["trainer"]['val_check_interval'],cfg["trainer"]['max_steps'] =_get_step_counts(counts, exp_params)
    val_exclusion_set = override_val_exclusion(train_df, val_exclusion_set, exp_params)
    exp_params['batch_size'] = exp_params['subj_per_batch'] #batch size is used in the collate function -> it is the subj_per_batch 

    n_im_per_subj = exp_params['unsupervised_kwargs']['n_im_per_subj']
    print(f"[DEBUG] Batch size is obtained from subj_per_batch ({exp_params['subj_per_batch']}) * n_im_per_subj ({n_im_per_subj}) -> {exp_params['subj_per_batch']*n_im_per_subj}", flush=True)
    assert image_size == exp_params['input_size'], f"image_size in config ({image_size}) does not match input_size in exp_params ({exp_params['input_size']})"

    transform = get_transforms(exp_params, None)
    train_loader,val_loader,_,_= prepare_loaders_PD(num_workers,prefetch_factor,exp_params,exclusion_set,val_exclusion_set, 
                                                                           grid_dict, transform, 
                                                                           SHARD_PATTERN_train=SHARD_PATTERN_train, SHARD_PATTERN_val=SHARD_PATTERN_val,
                                                                           train_df=train_df, merge_shards=exp_params['merge_shards'])
    
    #define model
    model = instantiate(cfg["model"])

    #define callbacks
    trainer_cfg = dict(cfg.get("trainer", {}))
    callbacks = [instantiate(c) for c in trainer_cfg.pop("callbacks", [])]

    #define logger
    logger = TensorBoardLogger(save_dir=root_path, name=f"{model_name}_{image_size}px")

    #define trainer
    trainer = L.Trainer(**trainer_cfg, logger=logger, callbacks=callbacks, default_root_dir=root_path, 
                        enable_progress_bar=False, fast_dev_run=exp_params['fast_dev_run'])
    # Save the config and experiment parameters to the log directory for reproducibility
    if trainer.is_global_zero and trainer.log_dir is not None:
        log_dir = Path(trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        with open(log_dir / "exp_params.pkl", "wb") as f:
            pickle.dump(exp_params, f)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.get("ckpt_path"))


    return

def _get_step_counts(counts, exp_params):
    n_subjects = sum(counts)
    n_batches = math.ceil(n_subjects/(exp_params['subj_per_batch']))
    n_steps_per_epoch = math.ceil(n_batches/(exp_params['accumulate_grad_batches']))
    steps_per_check_interval = math.ceil(n_steps_per_epoch * exp_params['val_check_interval']) 
    total_steps = n_steps_per_epoch * exp_params['epochs']
    return steps_per_check_interval,total_steps

def _get_unique_modalities(modalities):
    unique_modalities = []
    for m in modalities:
        if m not in unique_modalities:
            unique_modalities.append(m)
    return unique_modalities

def _set_fixed_exp_parameters(exp_params):
    fixed_params = {
        'problem': 'unsupervised_pretraining', #handedness, PD, 
        'unsupervised_pretrain':True,
        'class_col': 'no_label', #not in the column names -> follow the name not in .columns path

        'pass_example_input_array': False,
        'grouped': False, #if true i have all elements from the same case-control group in the batch and train to distinguish the case from the controls
        'pre_training': False, #if true the model is pre-trained on a large dataset of images and the classification head is trained on the PD dataset, if false the model is trained from scratch on the PD dataset
        'bce_aux_weight': 0.3, #weight for the BCE loss on the auxiliary output (the one that predicts the case-control group)
        'synthetic': None , # ALL_SYNTHETIC_TRANSFORMS or None
        'synthetic_proportions': [1/len(ALL_SYNTHETIC_TRANSFORMS) for _ in range(len(ALL_SYNTHETIC_TRANSFORMS))], #if synthetic is not None, the proportions of each synthetic class in the training set (must sum to 1)
        
        'merge_shards': False, #automatically set to true if you use handedness as label
        'num_tiles': 3,
        'use_grid': True,

        'custom_pre_trained_weights': None, #None, 'pre_trained_E3N_convnext_tiny_window_1','pre_trained_E3N_resnet18' or 'pre_trained_E3N_resnet50' or 'pre_trained_E3N_custom_window'
        'load_full_model': False, #if true loads backbone+head, if false only the backbone (the head is initialized randomly)
        'pretrained': True, #True, False, e.g. for resnet if True loads the imagenet weights for the backbone, if False loads the backbone with random weights

        'balanced_data': False,
        'balance_validation': False,    
    }
    exp_params.update(fixed_params)
    return exp_params

if __name__ == "__main__":
    main()
    