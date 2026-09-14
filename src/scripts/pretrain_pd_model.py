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

from src.utils.data_loading_utils import prepare_loaders_PD, load_grid_dict, synthetic_data_override
from src.utils.data_loading_utils import prepare_PD_dataset, prepare_exclusion_sets_PD, return_file_paths
from src.utils.model_utils import SequenceQuestionnaireModel, SetQuestionnaireModel, load_ln_checkpoint, FlexibleSequenceQuestionnaireModel
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_PD, debug_print_batch_meta
from src.utils.image_processing import ResizeLongestSide, PadToSquare, get_augmentation_transform, get_transforms,get_mu_std, ALL_SYNTHETIC_TRANSFORMS
from src.utils.training_utils import BestMetricTracker, ModelPDGrouped, ModelPDClassification, ClearCache, TimeLoader, get_optimization_groups
from src.utils.training_utils import set_automatic_hyperparameters, MemMonitor, BatchTimer, ThroughputMonitor, WriteProbe
from src.scripts.train_PD_model import get_input_modality

RESTORE = False
version_override =  11

exp_params = {
    'problem': 'PD', #handedness, PD, 
    'class_col': 'diag_park_final1_quest',
    'debug': False,

    #debugging
    'debugging_callbacks':True,
    'fast_dev_run':False, #can be False, None or True, False and None have same behavior

    #training modality
    'grouped': False, #if true i have all elements from the same case-control group in the batch and train to distinguish the case from the controls
    'pre_training': False,
    'bce_aux_weight': 0.3, #weight for the BCE loss on the auxiliary output (the one that predicts the case-control group)
    'synthetic': None, # ALL_SYNTHETIC_TRANSFORMS or None
    'synthetic_proportions': [1/len(ALL_SYNTHETIC_TRANSFORMS) for _ in range(len(ALL_SYNTHETIC_TRANSFORMS))], #if synthetic is not None, the proportions of each synthetic class in the training set (must sum to 1)


    #experiment parameters
    'data_modality': get_input_modality('window_view'), #mixed_view, window_view
    'num_tiles': 3,
    'use_grid': True,
    'use_balanced_weights': False,
    'balancing_factor': 3, #even if float is converted to int with int(balancing_factor), balancing_factor controls for each case-control group are kept 
    'balanced_data': False, #note that this and balace_validation are independent
    'balance_validation': False, #if True the validation set is balanced, if False it is not balanced
    'majority_class_id': 0, 
    'threshold_num': 1,
    'num_classes': 1, #1 for BCE loss, 2 for crossentropy
    'filter_missing': 'all', #'all', 'last_q' #if all remove only ids with grid_pattern=0000..00 13 times, 
    #if 'last_q' with the first last_q equal to 0
    'censor_time': 'all_matched',#'pre_diagnosis', #'all_matched',#'first_and_last',#'successive','last_successive_and_previous',#'last_and_successive', #'all', 'pre_diagnosis', 'pre_diagnosis_1y', 'last_and_previous','last_and_successive'
    'filter_modality' : 'digit', 

    #model definition
    'model': "convnext_tiny",#'swin_v2_t', #'efficientnet_v2_s',#"convnext_tiny" "FiveStageResidualStridedConvNet", #'swin_s' #'resnet18', 'custom_cnn', 'resnet34_layer1','resnet34_layer2','resnet34_layer3', 'resnet34', 'resnet50'
#clip-vit-large-patch14, clip-vit-large-patch14-inter
    'custom_pre_trained_weights': pre_trained_weights('pre_trained_E3N_convnext_tiny_window_1'), #None, 'pre_trained_E3N_resnet18' or 'pre_trained_E3N_resnet50' or 'pre_trained_E3N_custom_window'
    'pretrained': True, #True, False, e.g. for resnet if True loads the imagenet weights for the backbone, if False loads the backbone with random weights
    'norm_mu': 'PD_window', #imagenet,handedness,mnist,PD_window
    'norm_std': 'PD_window',
    'model_structure': 'FlexibleSequenceQuestionnaireModel', #'SetQuestionnaireModel',#'SequenceQuestionnaireModel',
    'val_check_interval': None, #None or float between 0 and 1, if None validation is done at the end of each epoch, if float validation is done every val_check_interval fraction of an epoch
    'align_train_metrics_to_val': False,  
    'min_window_steps': 50,
    'model_parameters': {
        'd_model': 128, 
        'n_heads': 4, # -> for each head 128/4 = 32 is the hidden dimension during the attention computation 
        'n_layers':1,
        'ff_mult':2, # the hidden dimension of the feedforward layer is ff_mult*d_model
        'dropout': 0.4,
        'count_norm': 2,
        'use_spread': False, #add a variance feature of dimension d_model to the average d_model feature
        'use_count_feature': False,
        'use_attention_pool': False, #if true overrides use_spread and use_count_feature 
        'seq_model':'gru', #causal_gru (bidirectional has no impact, predictions are returned per_step)
        'bidirectional':True,
    },

    #Transforms definitions
    'custom_transform': 'pad_resize_normalize',#'pad_resize_normalize', #None, #if not None overrides the transform defined for the model with ta custom one
    'apply_augmentation': None, #None, 'random_crop_half' ; if data_modality is a list the transform for each view mode will be determined
    #in the code based on the view name
    'invert_color':True,
    'to_grayscale': True, #if True converts the images to grayscale (1 channel) before feeding them to the model
    
    #Training params definition
    'lora_tuning': False, #if True uses LoRA tuning for the model, if False uses standard fine-tuning
    'use_opt_groups': True,
    'lr_decay': 0.75, #decay factor for the learning rate of the backbone layers, if use_opt_groups is True
    'lr_backbone': 1e-4,
    'lr_classifier_head': 1e-3,
    'lr_scheduling': 'cosine', #'cosine' # 'cosine', 'step', None
    'batch_size': 4,
    'scale_lr_with_batch_size': True, #if True scales the learning rate with the batch size, if False uses the learning rate defined in lr_backbone and lr_classifier_head
    'num_epochs': 50,
    'max_steps': -1, #N or -1
    'patience': 10, #always in epochs (even if you take fractional validation steps -> real patience will be 1/val_check_interval * patience)
    'stopping_metric': 'val/pr_auc',#'val/pr_auc', #'val/loss', #the metric to monitor for early stopping, can be 'val/pr_auc', 'val/loss' or 'val/roc_auc' or 'val/f1' or 'val/mcc' or 'val/accuracy'
    'eta_min_cosine': 1e-6, #the timm-style convention (base/100)
    'weight_decay': 0.05, #1e-5 - 1e-8 (swin fine-tuning) #1e-2 (resnet for fine-tuning), 0.05 (resnet for training from scratch)
    'warmup_fraction': 0.05,   # ~5% of total steps as warmup
    'input_size': 224,
    'layers_to_unfreeze': ['all'],
    #['classifier','vision_model.features.6','vision_model.features.7','vision_model.final_norm'], #['all'],#['classifier','layer4'],#['all','classifier'], #Update it for every model
    #['stages.3', 'stages.4', 'head', 'projector', 'classifier']
    'seed': 42,
    'accumulate_grad_batches': 16,#8,   # effective batch = batch_size * accumulate_grad_batches or None
    'precision': "16-mixed", #None, #"16-mixed","bf16-mixed"        # AMP: autocast + GradScaler handled for you or None
    #bf16 needs Ampere or newer (A100, H100, RTX 30xx/40xx
    'gradient_clip_val': 1.0, #1.0, None

    'prefetch_factor': 4,
}

def get_source_path():
    return "/home/a_morelli/models/model_training_logs/pre_trained_models/E3N"
SOURCE_PATH = get_source_path()

# Authomatic settings
exp_params['list_of_ids_paths'], exp_params['data_folder'], exp_params['grid_dict_path'] = return_file_paths(exp_params['problem'], 
                                                                                                             exp_params['grouped'], 
                                                                                                             exp_params['pre_training'])
exp_params = set_automatic_hyperparameters(exp_params)


SHARD_PATTERN_train = os.path.join(exp_params['data_folder'],"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(exp_params['data_folder'],"val/worker*_shard-*.tar")
SAVE_DEBUG_PATH = "/home/a_morelli/vscode_projects/model_training/data/debug_training"

define_optimization_groups = get_optimization_groups(model_name=exp_params['model'],exp_params=exp_params)

OUTPUT_PATH = os.path.join(SOURCE_PATH,f"{exp_params['model']}_model_results")
CHECKPOINT_PATH = os.path.join(OUTPUT_PATH, "checkpoints")
exp_params['CHECKPOINT_PATH'] = CHECKPOINT_PATH
exp_params['OUTPUT_PATH'] = OUTPUT_PATH
exp_params['SOURCE_PATH'] = SOURCE_PATH

DEBUG_IMGS = True
VERBOSE = True