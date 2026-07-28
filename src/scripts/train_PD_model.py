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

from src.utils.data_loading_utils import prepare_loaders_PD, load_grid_dict, synthetic_data_override
from src.utils.data_loading_utils import prepare_PD_dataset, prepare_exclusion_sets_PD, return_file_paths
from src.utils.model_utils import SequenceQuestionnaireModel, SetQuestionnaireModel, load_ln_checkpoint
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_PD, debug_print_batch_meta
from src.utils.image_processing import ResizeLongestSide, PadToSquare, get_augmentation_transform, get_transforms,get_mu_std, ALL_SYNTHETIC_TRANSFORMS
from src.utils.training_utils import BestMetricTracker, ModelPDGrouped, ModelPDClassification, ClearCache, TimeLoader, get_optimization_groups
from src.utils.training_utils import set_automatic_hyperparameters, MemMonitor, BatchTimer, ThroughputMonitor
#set variables
#torch.set_num_threads(2)  # Set the number of threads cores for the main process (eg 2+workers=num physical cores)


def pre_trained_weights(name):
    if name is None:
        return None
    #options for custom_pre_trained_weights:
    if name == 'pre_trained_E3N_resnet18':
        out_path =os.path.join('/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/E3N/resnet18_model_results/checkpoints/v_2',
                    'best-epoch=01-val_loss=0.23.ckpt')
    elif name == 'pre_trained_E3N_resnet18_window':
        out_path = os.path.join('/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/E3N/resnet18_model_results/checkpoints/v_4',
                    'best-02-0.2222.ckpt')
    elif name == 'pre_trained_E3N_resnet50':
        out_path = os.path.join('/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/E3N/resnet50_model_results/checkpoints/v_1',
                    'best-epoch=00-val_loss=0.30.ckpt')
    elif name == 'pre_trained_E3N_resnet50_window':
        out_path = os.path.join('/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/E3N/resnet50_model_results/checkpoints/v_2',
                    'best-02-0.2264.ckpt')
    elif name == 'pre_trained_E3N_custom_window':
        out_path = os.path.join('/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/E3N/FiveStageResidualStridedConvNet_model_results/checkpoints/v_1',
                    'best-00-0.8068.ckpt')
    '''os.path.join(
        '/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/mnist',
        'resnet50/checkpoints/best-resnet18-mnist-epoch=28-val_loss=0.0197.ckpt'
    ),'''
    #resnet50/checkpoints/best-resnet18-mnist-epoch=28-val_loss=0.0197.ckpt
    #resnet18/checkpoints/best-resnet18-mnist-epoch=05-val_loss=0.0181.ckpt
    return out_path

def get_input_modality(name):
    if name=='mixed_view':
        return ['X_crop', 'digit_full', 'digit_crop', 'digit', 'digit', 'digit', 'text_full', 'text_crop', 'text', 'text', 'text']
    elif name=='window_view':
        return ['X_window' for _ in range(3)]+['text_window' for _ in range(3)]+ ['digit_window' for _ in range(3)]

exp_params = {
    'problem': 'PD', #handedness, PD, pre_training
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
    'use_balanced_weights': True,
    'balancing_factor': 3, #even if float is converted to int with int(balancing_factor), balancing_factor controls for each case-control group are kept 
    'balanced_data': True, #note that this and balace_validation are independent
    'balance_validation': False, #if True the validation set is balanced, if False it is not balanced
    'majority_class_id': 0, 
    'threshold_num': 1,
    'num_classes': 1, #1 for BCE loss, 2 for crossentropy
    'filter_missing': 'all', #'all', 'last_q' #if all remove only ids with grid_pattern=0000..00 13 times, 
    #if 'last_q' with the first last_q equal to 0
    'censor_time': 'all_matched',#'first_and_last',#'successive','last_successive_and_previous',#'last_and_successive', #'all', 'pre_diagnosis', 'pre_diagnosis_1y', 'last_and_previous','last_and_successive'
    'filter_modality' : 'digit', 

    #model definition
    'model':"resnet18",#"FiveStageResidualStridedConvNet", #'swin_s' #'resnet18', 'custom_cnn', 'resnet34_layer1','resnet34_layer2','resnet34_layer3', 'resnet34', 'resnet50'
#clip-vit-large-patch14, clip-vit-large-patch14-inter
    'custom_pre_trained_weights': pre_trained_weights('pre_trained_E3N_resnet18_window'), #None, 'pre_trained_E3N_resnet18' or 'pre_trained_E3N_resnet50' or 'pre_trained_E3N_custom_window'
    #or
    'norm_mu': 'PD_window', #imagenet,handedness,mnist,PD_window
    'norm_std': 'PD_window',
    'model_structure': 'SequenceQuestionnaireModel', #'SetQuestionnaireModel',#'SequenceQuestionnaireModel',
    'model_parameters': {
        'd_model': 128, 
        'n_heads': 4, # -> for each head 128/4 = 32 is the hidden dimension during the attention computation 
        'n_layers':1,
        'ff_mult':2, # the hidden dimension of the feedforward layer is ff_mult*d_model
        'dropout': 0.4,
        'count_norm': 2,
        'use_spread': True, #add a variance feature of dimension d_model to the average d_model feature
        'use_count_feature': True,
        'use_attention_pool': False #if true overrides use_spread and use_count_feature 
    },

    #Transforms definitions
    'custom_transform': 'pad_resize_normalize', #None, #if not None overrides the transform defined for the model with ta custom one
    'apply_augmentation': None, #None, 'random_crop_half' ; if data_modality is a list the transform for each view mode will be determined
    #in the code based on the view name
    'invert_color':True,
    'to_grayscale': True, #if True converts the images to grayscale (1 channel) before feeding them to the model
    
    #Training params definition
    'use_opt_groups': False,
    'lr_backbone': 1e-6,
    'lr_classifier_head': 1e-4,
    'lr_scheduling': 'cosine', #'cosine' # 'cosine', 'step', None
    'batch_size': 4,
    'num_epochs': 60,
    'patience': 10,
    'stopping_metric': 'val/pr_auc',#'val/pr_auc', #'val/loss', #the metric to monitor for early stopping, can be 'val/pr_auc', 'val/loss' or 'val/roc_auc' or 'val/f1' or 'val/mcc' or 'val/accuracy'
    'val_check_interval': 1.0, #if integers the number of steps after which 
    #to perform validation, if float the fraction of the epoch after which to perform validation, 1.0 is the default value for doing at epoch end
    'eta_min_cosine': 1e-7,
    'weight_decay': 1e-2, #0.05 (swi) #1e-2 (resnet)
    'warmup_fraction': 0.05,   # ~5% of total steps as warmup
    'input_size': 224,
    'layers_to_unfreeze': ['classifier','layer4'], #['all'],#['classifier','layer4'],#['all','classifier'], #Update it for every model
    #['stages.3', 'stages.4', 'head', 'projector', 'classifier']
    'seed': 42,
    'accumulate_grad_batches': 8,#8,   # effective batch = batch_size * accumulate_grad_batches or None
    'precision': "16-mixed", #None, #"16-mixed",        # AMP: autocast + GradScaler handled for you or None
    'gradient_clip_val': 1.0, #1.0, None

    'prefetch_factor': 2,
}

def get_source_path():
    if exp_params['problem'] == 'PD' and not exp_params['pre_training']:
        return "/mnt/beegfs02/scratch/a_morelli/model_training/PD/"
    elif exp_params['problem'] == 'PD' and exp_params['pre_training']:
        return "/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/E3N"
    else:
        raise ValueError(f"Unknown problem type: {exp_params['problem']}")
SOURCE_PATH = get_source_path()
#"/mnt/beegfs02/scratch/a_morelli/model_training/PD/"

# Authomatic settings
exp_params['list_of_ids_paths'], exp_params['data_folder'], exp_params['grid_dict_path'] = return_file_paths(exp_params['problem'], exp_params['grouped'], exp_params['pre_training'])
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

#if you need the following variables load them from the csv_data ->
#last available questionnaire for the subject, last_q = id_row['last_avail_q']
#variables_to_add = ['etudegp', 'profq2', 'lateralite', 'relative_age', 'birth_date', 'follow_up_time']
#filter these first when rebalancing: warning==1 if the control was selected with at least -> "at_least_warning": at_least_warning,
    
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

    #exclude controls from the training if i want to reduce the asimmetry of the dataset (for example if i want to have a 1:1 ratio between cases and controls)
    exclusion_set, val_exclusion_set, counts = prepare_exclusion_sets_PD(exp_params,verbose=VERBOSE,class_col=exp_params['class_col'])

    write_log, current_version = logging_initialization() 

    model, transform = model_initialization(write_log,exp_params,verbose=VERBOSE, **exp_params['model_parameters'])

    train_df = pd.read_parquet(exp_params['list_of_ids_paths'])

    #get_memory_usage(grid_dict, train_df, exclusion_set)

    train_loader,val_loader,_,_= prepare_loaders_PD(worker,prefetch_factor,exp_params,exclusion_set,val_exclusion_set, 
                                                                       grid_dict, transform, 
                                                                       SHARD_PATTERN_train=SHARD_PATTERN_train, SHARD_PATTERN_val=SHARD_PATTERN_val,
                                                                       train_df=train_df)
    
    
    '''p = psutil.Process(os.getpid())
    for i, batch in enumerate(train_loader):
        del batch
        if i % 200 == 0:
            print(f"batch {i}: rss={p.memory_info().rss/1e9:.2f} GB", flush=True)
        if i == 300: break'''
    

    
    if DEBUG_IMGS:
        debug(train_loader,val_loader,exp_params)
    
    
    lit_model = litmodel_initialization(model,counts,write_log,define_optimization_groups,exp_params, exclusion_set, VERBOSE)

    trainer, metrics_tracker = trainer_definition(current_version, exp_params)
    
    exp_params['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    exp_params['best_epoch'] = "N/A (Cancelled)"
    exp_params['best_val_loss'] = "N/A (Cancelled)"
    exp_params['best_val_acc'] = "N/A (Cancelled)"
    exp_params['best_train_acc'] = "N/A (Cancelled)"
    exp_params['current_version'] = current_version

    '''it = iter(train_loader)
    for _ in range(60): next(it)
    t = time.time()
    for _ in range(300): next(it)
    print("pre-fit time per dataloader:", (time.time() - t) / 300)'''

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # If it finishes successfully, update the metrics dictionary with the real values
    exp_params["best_epoch"] = metrics_tracker.best_epoch
    exp_params["best_val_loss"] = metrics_tracker.best_val_loss
    exp_params["best_val_acc"] = metrics_tracker.best_val_acc
    exp_params["best_train_acc"] = metrics_tracker.best_train_acc
    #copy the list_ids parquet file to the checkpoint folder with the name 
    base_name = os.path.basename(exp_params['list_of_ids_paths'])
    copy_path = os.path.join(CHECKPOINT_PATH,f'v_{current_version}', base_name)
    shutil.copy(exp_params['list_of_ids_paths'], copy_path)
    exp_params['list_of_ids_paths'] = copy_path
    exp_params['optimization_groups'] = define_optimization_groups
    
    # Save normal log
    save_experiment_logs(exp_params)
        
    #save the exp_params dictionary to a pickle file in the checkpoint folder
    with open(os.path.join(CHECKPOINT_PATH,f'v_{current_version}', 'exp_params.pkl'), 'wb') as f:
        pickle.dump(exp_params, f)



#### HELPER FUCNTIONS #### 
def get_memory_usage(grid_dict, train_df, exclusion_set):
    def sizeof_grid_dict(d):
        """Sum numpy buffer sizes + rough python overhead for a (possibly nested) dict."""
        total = 0
        n_objects = 0
        stack = [d]
        while stack:
            obj = stack.pop()
            n_objects += 1
            if isinstance(obj, np.ndarray):
                total += obj.nbytes
            elif isinstance(obj, dict):
                total += sys.getsizeof(obj)
                stack.extend(obj.keys())
                stack.extend(obj.values())
            elif isinstance(obj, (list, tuple, set)):
                total += sys.getsizeof(obj)
                stack.extend(obj)
            else:
                total += sys.getsizeof(obj)
        return total, n_objects

    nbytes, nobjs = sizeof_grid_dict(grid_dict)
    print(f"grid_dict: {nbytes/1e9:.2f} GB across {nobjs:,} python objects")

    print(f"train_df: {train_df.memory_usage(deep=True).sum()/1e9:.2f} GB, "
        f"dtypes:\n{train_df.dtypes.value_counts()}")

    print(f"exclusion_set: {sys.getsizeof(exclusion_set)/1e9:.3f} GB, "
        f"{len(exclusion_set):,} items", flush=True)

def rss_gb(): return psutil.Process(os.getpid()).memory_info().rss / 1e9

def validity_checks(exp_params):
    if exp_params['grouped']==False and 'group' in exp_params['data_folder']:
        raise ValueError("Error: you have selected grouped=False but the data_folder contains 'group' in its name. Please check your settings.")
    
def trainer_definition(current_version, exp_params, cv=None):

    stopping_metric = exp_params['stopping_metric']  # e.g., 'val/pr_auc'
    if stopping_metric in ['val/pr_auc', 'val/roc_auc', 'val/f1', 'val/mcc', 'val/accuracy']:
        mode = 'max'  # We want to maximize these metrics
    elif stopping_metric in ['val/loss',"val/loss_unweighted"]:
        mode = 'min'  # We want to minimize loss

    # 2. Setup Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor=stopping_metric,
        dirpath=os.path.join(exp_params['CHECKPOINT_PATH'], f'v_{current_version}'),
        # auto_insert_metric_name=False so the '/' in 'val/pr_auc' is NOT turned
        # into a subdirectory — the value is substituted, not the key.
        filename=f"best-{{epoch:02d}}-{{{stopping_metric}:.4f}}",
        auto_insert_metric_name=False,
        save_top_k=1,
        mode=mode,
        save_last=False,
    )

    periodic_ckpt = ModelCheckpoint(
        dirpath=os.path.join(exp_params['CHECKPOINT_PATH'], f'v_{current_version}'),
        filename=f"latest-{{epoch:02d}}",
        auto_insert_metric_name=False, #auto_insert_metric_name=False with {val/pr_auc:.4f} produces best-04-0.7321.ckpt
        save_top_k=1,
        every_n_epochs=1,
        monitor=None,          # no metric -> saves the current/last state each time
    )

    # 3. Setup Early Stopping
    early_stop_callback = EarlyStopping(
        monitor=stopping_metric,
        patience=exp_params['patience'],
        mode=mode,
        verbose=True,
    )

    metrics_tracker = BestMetricTracker()
    
    # 1. Construct clean paths (removed trailing slash, kept version as a string/int)
    if cv is not None:
        log_root = os.path.join(exp_params['SOURCE_PATH'], 'tensor_board_logging',exp_params['model'],str(current_version))
        version_dir = os.path.join(log_root, 'run_'+str(cv))
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
        # 3. Initialize the logger
        tb_logger = TensorBoardLogger(
            save_dir=log_root,
            name='',
            log_graph=False, 
            version='run_'+str(cv)  
        )
    else:
        log_root = os.path.join(exp_params['SOURCE_PATH'], 'tensor_board_logging')
        version_dir = os.path.join(log_root, exp_params['model'], 'version_'+str(current_version)) #tensorboard automatically adds 'version_' prefix, 
        #so we match that format here.
        # 2. Wipe the old folder if it exists
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
        # 3. Initialize the logger
        tb_logger = TensorBoardLogger(
            save_dir=log_root,
            name=exp_params['model'],
            log_graph=False, 
            version=current_version  # Works perfectly as an integer or string
        )

    optional = ['precision', 'accumulate_grad_batches', 'gradient_clip_val', 'fast_dev_run']
    extra_kwargs = {k: exp_params[k] for k in optional if exp_params.get(k) is not None}

    # 4. Initialize Trainer and Fit
    extra_callbacks = [ClearCache(),ThroughputMonitor(), MemMonitor()] if exp_params['debugging_callbacks'] else []
    trainer = L.Trainer(
        #limit_train_batches=500, limit_val_batches=0,
        max_epochs=exp_params['num_epochs'],
        logger = tb_logger,
        accelerator="auto",                # Automatically selects GPU/CPU/MPS
        callbacks=[checkpoint_callback, early_stop_callback, metrics_tracker, periodic_ckpt]+extra_callbacks,
        profiler="simple",  # Add this line to get a performance summary
        enable_progress_bar=False,  # Remove this CPU overhead
        **extra_kwargs
    )
    # if you want you can set anothr kind of logger (not tensorboard but csv ..)
    return trainer, metrics_tracker

def debug(train_dataloader,val_dataloader,exp_params):
    batch = next(iter(train_dataloader))
    debug_print_batch_meta(batch)

    debug_images_PD(mean=exp_params['norm_mu'], std=exp_params['norm_std'], loader=train_dataloader, out_dir=os.path.join(SAVE_DEBUG_PATH,'train'))
    debug_images_PD(mean=exp_params['norm_mu'], std=exp_params['norm_std'], loader=val_dataloader, out_dir=os.path.join(SAVE_DEBUG_PATH,'val'))

def litmodel_initialization(model, counts,write_log, define_optimization_groups,exp_params, exclusion_set, verbose):
    additional_kwargs={
    }
    if exp_params['grouped']:
        model_class=ModelPDGrouped
        additional_kwargs['bce_aux_weight'] = exp_params['bce_aux_weight']
        total_units = compute_unique_groups(exp_params, exclusion_set) #the total number of unique groups after filtering
        if verbose:
            print(f"Total unique groups in training set after filtering: {total_units}")
    else:
        model_class=ModelPDClassification
        additional_kwargs['class_counts']= counts
        additional_kwargs['num_classes']= exp_params['num_classes']
        additional_kwargs['use_balanced_weights']= exp_params['use_balanced_weights']
        #additional_kwargs['balancing_factor']= exp_params['balancing_factor']
        additional_kwargs['balanced_data']= exp_params['balanced_data']
        total_units = sum(counts)  # total number of samples in all classes
    example_input_array = model_class.make_example_input(k=exp_params['num_tiles'], n_slots=13, C=exp_params['num_channels'], 
                                                         H=exp_params['input_size'], W=exp_params['input_size'])
    lit_model = model_class(write_log,model=model,total_units=total_units,lr_backbone=exp_params['lr_backbone'], 
                         lr_classifier_head=exp_params['lr_classifier_head'], example_input_array=example_input_array, 
                         opt_groups=define_optimization_groups, num_epochs=exp_params['num_epochs'], lr_scheduling=exp_params['lr_scheduling'],
                         weight_decay=exp_params['weight_decay'], warmup_fraction=exp_params['warmup_fraction'], 
                         eta_min_cosine=exp_params['eta_min_cosine'], batch_size=exp_params['batch_size'],
                         accumulate_grad_batches=exp_params['accumulate_grad_batches'], world_size=1,**additional_kwargs)
    return lit_model

#model loading
def model_initialization(write_log,exp_params, verbose=True,val=False, **kwargs):
    backbone,transform = get_model(name=exp_params['model'], pretrained=True, 
                                   custom_pre_trained_weights=exp_params['custom_pre_trained_weights'],grayscale=exp_params['to_grayscale'])
    print("############# Model backbone loaded! #############")
    transform = get_transforms(exp_params, transform)
    out=test_output(exp_params['input_size'], backbone, channels=exp_params['num_channels']) #test the output of the backbone to determine the number of features for the classification head
    in_features = out.shape[1]  
    
    if exp_params['model_structure'] == 'SequenceQuestionnaireModel':
        n_slots = 13  # This is a fixed value based on your description
        d_model = kwargs.get('d_model', 128)
        n_heads  = kwargs.get('n_heads', 4)
        n_layers = kwargs.get('n_layers', 2)
        ff_mult = kwargs.get('ff_mult', 2)
        dropout = kwargs.get('dropout', 0.4)
        model = SequenceQuestionnaireModel(backbone,feat_dim=in_features, n_classes=exp_params['num_classes'], n_slots=n_slots, 
                                           d_model=d_model, n_heads=n_heads, n_layers=n_layers, ff_mult=ff_mult, view_agg='attention', dropout=dropout)
    elif exp_params['model_structure'] == 'SetQuestionnaireModel':
        d_model = kwargs.get('d_model', 128)
        ff_mult = kwargs.get('ff_mult', 2)
        dropout = kwargs.get('dropout', 0.4)
        count_norm = kwargs.get('count_norm', 2)
        use_spread = kwargs.get('use_spread', True)
        use_count_feature = kwargs.get('use_count_feature', True)
        use_attention_pool = kwargs.get('use_attention_pool', False)
        model = SetQuestionnaireModel(backbone,feat_dim=in_features, n_classes=exp_params['num_classes'],  
                                           d_model=d_model, ff_mult=ff_mult, view_agg='attention', dropout=dropout,count_norm=count_norm,
                                           use_spread=use_spread, use_count_feature=use_count_feature, use_attention_pool=use_attention_pool)
    else:
        raise ValueError(f"Unknown model_structure: {exp_params['model_structure']}")
    
    model = load_ln_checkpoint(model,exp_params['custom_pre_trained_weights'])

    if val:
        return model, transform
    
    unfreeze_layers(model,layer_names=exp_params['layers_to_unfreeze'])

    if verbose:
        write_log(f"Size of extracted representation: {in_features}") # <-- ADD THIS LINE
        write_log(f"Device of model after initialization: {next(model.parameters()).device}") # <-- ADD THIS LINE
        trainable_parameters_info = get_trainable_parameters_string(model)
        write_log("Model Architecture and Trainable Parameters right after initialization:")
        write_log(trainable_parameters_info)
    return model, transform

def logging_initialization():
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
    log_folder = os.path.join(OUTPUT_PATH,"custom_logs")
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

def compute_unique_groups(exp_params, exclusion_set):
    # Compute unique groups for training 
    train_groups = set()
    
    with open(exp_params['list_of_ids_paths'], 'rb') as f:
        id_list_df = pd.read_parquet(f)
    
    # Filter out excluded IDs for training
    train_df = id_list_df[~id_list_df['unique_id'].isin(exclusion_set)]
    #create group_id column (unique_id = XXXX_YYYY with YYYY the group_id)
    train_df['group_id'] = train_df['unique_id'].str.split('_').str[1]
    # Get number of unique groups in training set
    train_groups = set(train_df['group_id'].unique())
    
    return len(train_groups)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

def get_trainable_parameters_string(model):
    """Returns a string containing the names and parameter counts of layers that require gradients."""
    output_lines = []
    output_lines.append("\n--- Trainable Parameters ---")
    
    total_trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_trainable_params += num_params
            output_lines.append(f"  - Layer: {name} | Parameters: {num_params:,}")
            
    output_lines.append(f"Total Trainable Parameters: {total_trainable_params:,}")
    output_lines.append("--------------------------\n")
    
    return "\n".join(output_lines)

def save_experiment_logs(params_dict, status_suffix=""):
    """Helper function to write/append dictionary to the CSV log file."""
    log_path = os.path.join(params_dict['CHECKPOINT_PATH'], "experiments_log.csv")
        
    if not os.path.exists(log_path):
        df = pd.DataFrame([params_dict])
        df.to_csv(log_path, index=False)
    else:
        df = pd.read_csv(log_path)
        new_row = pd.DataFrame([params_dict])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(log_path, index=False)
    print(f"Experiment parameters successfully logged to {log_path}")

def check_incompatible_params(exp_params):
    return 

    assert not (exp_params['num_tiles'] > 1 and exp_params['data_modality'] == 'all'), \
    "Error: Data modality = 'all' and num_tiles > 1 are incompatible"

    assert not (exp_params['data_modality'] == 'all' and exp_params['use_grid']), \
    "Error: Data modality = 'all' and USE_GRID=True are incompatible"

if __name__ == "__main__":
    main(exp_params)
    