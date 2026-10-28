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

from src.scripts.train_PD_model import get_input_modality



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
        
        #loss options
        
        'filter_missing': 'all', #'all', 'last_q' #if all remove only ids with grid_pattern=0000..00 13 times, 
        #if 'last_q' with the first last_q equal to 0
        'censor_time': 'pre_diagnosis',#'pre_diagnosis', #'all_matched',#'first_and_last',#'successive','last_successive_and_previous',#'last_and_successive', #'all', 'pre_diagnosis', 'pre_diagnosis_1y', 'last_and_previous','last_and_successive'
        #censor time is set authomatically to all for pre_training==true and problem in [handedness]
        'filter_modality' : 'digit', 
    
        #model definition
        'model': "convnext_tiny",#'vit_b_16',#'swin_v2_t', #'efficientnet_v2_s',#"convnext_tiny" "FiveStageResidualStridedConvNet", #'swin_s' #'resnet18', 'custom_cnn', 'resnet34_layer1','resnet34_layer2','resnet34_layer3', 'resnet34', 'resnet50'
        #clip-vit-large-patch14, clip-vit-large-patch14-inter
    
        'val_check_interval': None, #None or float between 0 and 1, if None validation is done at the end of each epoch, if float validation is done every val_check_interval fraction of an epoch
        'align_train_metrics_to_val': True,  
        #val and align are set authomatically: to none if pre_training is not active and PD; to 0.3 and True if problem==handedness
        'min_window_steps': 50,

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
    version_override =  7

    exp_params = _set_fixed_exp_parameters(exp_params)
    

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
    }
    exp_params.update(fixed_params)
    return exp_params

if __name__ == "__main__":
    main()
    