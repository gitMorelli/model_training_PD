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
    exp_params = {
        'seed': 42,

        'debug': False,
    
        #debugging
        'debugging_callbacks':True,
        'fast_dev_run': False, #can be False, None or True, False and None have same behavior
    
        #training modality
        'selected_properties': [],#['etudegp', 'profq2', 'lateralite', 'relative_age', 'case_dt_dateq*'], #None or [] means no conditioning on covariates
        #['etudegp', 'profq2', 'lateralite', 'relative_age', 'case_dt_dateq*']
        'global_fusion':'late', #early, late
    
        #experiment parameters
        'data_modality': get_input_modality('window_3x_2d_1t'),#'window_3x_2d_1t'), #mixed_view, window_view, window_view_minimal
        
        'filter_missing': 'all', #'all', 'last_q' #if all remove only ids with grid_pattern=0000..00 13 times, 
        #if 'last_q' with the first last_q equal to 0
        'censor_time': 'pre_diagnosis',#'pre_diagnosis', #'all_matched',#'first_and_last',#'successive','last_successive_and_previous',#'last_and_successive', #'all', 'pre_diagnosis', 'pre_diagnosis_1y', 'last_and_previous','last_and_successive'
        #censor time is set authomatically to all for pre_training==true and problem in [handedness]
        'filter_modality' : 'digit', 
    
        #model definition
        'model': "convnext_tiny",#'vit_b_16',#'swin_v2_t', #'efficientnet_v2_s',#"convnext_tiny" "FiveStageResidualStridedConvNet", #'swin_s' #'resnet18', 'custom_cnn', 'resnet34_layer1','resnet34_layer2','resnet34_layer3', 'resnet34', 'resnet50'
        #clip-vit-large-patch14, clip-vit-large-patch14-inter

        #Transforms definitions
        'norm_mu': 'imagenet', 
        'norm_std': 'imagenet',
        'custom_transform': None, #'2xcrop_resize_normalize',#'crop_normalize', ##'pad_resize_normalize', #None, #if not None overrides the transform defined for the model with ta custom one
        'input_size': 224, #224,32
        'apply_augmentation': None, #None, 'random_crop_half' ; if data_modality is a list the transform for each view mode will be determined
        #in the code based on the view name
        'invert_color':True,
        'to_grayscale': True, #if True converts the images to grayscale (1 channel) before feeding them to the model
        'image_pixel_space_augmentations': None,#None, #None, ALL,NO_AUG,PHOTO_ONLY,GEOM_ONLY ; None and NO_AUG are equivalent but None skips the function call
    
        
        #Training params definition
        'max_val_samples': 2000, #reduce the number of samples in the validation set to 1000 for pre-training
        'batch_size': 16,
        'num_epochs': 60,
        'max_steps': -1, #N or -1
        'patience': 20, #always in epochs (even if you take fractional validation steps -> real patience will be 1/val_check_interval * patience)
        'lr_backbone': 5e-5,
        'lr_classifier_head': 5e-4,
        'lr_scheduling': "two_stage", #'cosine' # "two_stage", 'step', None
        'scale_lr_with_batch_size': False, #if True scales the learning rate with the batch size, if False uses the learning rate defined in lr_backbone and lr_classifier_head
        'stopping_metric': 'val/pr_auc',#'val/pr_auc', #'val/loss', #the metric to monitor for early stopping, can be 'val/pr_auc', 'val/loss' or 'val/roc_auc' or 'val/f1' or 'val/mcc' or 'val/accuracy'
        'eta_min_cosine': 1e-6, #the timm-style convention (base/100)
        'weight_decay': 0.01, #1e-5 - 1e-8 (swin fine-tuning) #1e-2 (resnet for fine-tuning), 0.05 (resnet for training from scratch)
        'warmup_fraction': 0.05,   # ~5% of total steps as warmup
        'accumulate_grad_batches': 4,#8,   # effective batch = batch_size * accumulate_grad_batches or None
        'precision': "bf16-mixed",#"16-mixed", #None, #"16-mixed","bf16-mixed"        # AMP: autocast + GradScaler handled for you or None
        #bf16 needs Ampere or newer (A100, H100, RTX 30xx/40xx
        'gradient_clip_val': 1.0, #1.0, None
    
        'prefetch_factor': 4,
    }
    RESTORE = False
    VERBOSE = True
    version_override =  7
    source_path = "home/a_morelli/models/model_training_logs/pre_trained_models/unsupervised"
    exp_params['list_of_ids_paths'], exp_params['data_folder'], exp_params['grid_dict_path'] = return_file_paths(exp_params['problem'], exp_params['grouped'], exp_params['pre_training'])
    exp_params = set_automatic_hyperparameters(exp_params)
    SHARD_PATTERN_train = os.path.join(exp_params['data_folder'],"train/worker*_shard-*.tar")
    SHARD_PATTERN_val = os.path.join(exp_params['data_folder'],"val/worker*_shard-*.tar")
    SAVE_DEBUG_PATH = "/home/a_morelli/vscode_projects/model_training/data/debug_training"

    exp_params = _set_fixed_exp_parameters(exp_params)

    #read the cfg file
    cfg = load_config()  # plain nested dict: cfg["model"]["init_args"]["lr"], cfg["trainer"]["max_epochs"], ...
    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "default")
    num_workers = cfg.get("num_workers", 4)
    prefetch_factor = exp_params['prefetch_factor'] if num_workers > 0 else None
    exp_params['precision'] = _resolve_precision(exp_params['precision'])

    #set variables
    root_path = os.path.join(source_path, exp_name)
    torch.set_float32_matmul_precision("high")
    L.seed_everything(exp_params['seed'], workers=True)
    exp_params['norm_mu'],exp_params['norm_std'] = get_mu_std(exp_params, verbose=VERBOSE)
    

    #prepare variables
    grid_dict = load_grid_dict(exp_params)
    train_df = pd.read_parquet(exp_params['list_of_ids_paths'])
    train_df, *_ = preprocess_PD_csv(train_df,exp_params['selected_properties'], split_col='split')
    n_local,n_global = _selected_properties_get_dims(exp_params, train_df)
    print(f"n_local: {n_local}, n_global: {n_global}", flush=True)
    exclusion_set, val_exclusion_set, counts = prepare_exclusion_sets_PD(exp_params,verbose=VERBOSE,class_col=exp_params['class_col'])
    val_exclusion_set = override_val_exclusion(train_df, val_exclusion_set, exp_params)

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
    model_name = cfg["model"]["class_path"].rsplit(".", 1)[-1]
    image_size = cfg["data"]["init_args"].get("image_size")
    logger = TensorBoardLogger(save_dir=root_path, name=f"{model_name}_{image_size}px")

    #define trainer
    trainer = L.Trainer(**trainer_cfg, logger=logger, callbacks=callbacks, default_root_dir=root_path)

    # Save the config and experiment parameters to the log directory for reproducibility
    if trainer.is_global_zero:
        log_dir = Path(trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        with open(log_dir / "exp_params.pkl", "wb") as f:
            pickle.dump(exp_params, f)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.get("ckpt_path"))


    return



def _set_fixed_exp_parameters(exp_params):
    fixed_params = {
        'problem': 'unsupervised_pretraining', #handedness, PD, 

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
    