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

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset, melt_df
from src.utils.data_loading_utils import prepare_PD_dataset, generate_exclusion_set_PD
from src.utils.model_utils import SequenceQuestionnaireModel
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_PD, debug_print_batch_meta
from src.utils.image_processing import ResizeLongestSide, PadToSquare
from src.utils.training_utils import ModelPD, BestMetricTracker

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/PD/"
exp_params = {
    'list_of_ids_paths': "/home/a_morelli/datasets/id_lists/final_data_for_training.parquet",
    'data_folder': "final_png_whitebg",
    'grid_dict_path': "/mnt/beegfs02/scratch/a_morelli/datasets/PD_data_h5.pkl",

    #experiment parameters
    'data_modality': ['digit_full','digit_crop']+['digit' for _ in range(3)], # 'X', 'text', 'digit', 'all' (all returns 3x3x224x224 elements instead of 3x224x224)
    #or list e.g. ['digit_full','digit_crop','digit','digit','digit'] for 5 tiles
    'num_tiles': 3,
    'use_grid': True,
    'use_balanced_weights': True,
    'balancing_factor': 1, #even if float is converted to int with int(balancing_factor), balancing_factor controls for each case-control group are kept 
    'balanced_data': True, #note that this and balace_validation are independent
    'balance_validation': True, #if True the validation set is balanced, if False it is not balanced
    'majority_class_id': 0, 
    'threshold_num': 1,
    'num_classes': 1, #1 for BCE loss, 2 for crossentropy
    'filter_missing': 'last_q', #'all', 'last_q' #if all remove only ids with grid_pattern=0000..00 13 times, 
    #if 'last_q' with the first last_q equal to 0
    'censor_time': -1, #0, -1 (if keep all) or a positive value

    #model definition
    'model':"resnet18", #'swin_s' #'resnet18', 'custom_cnn', 'resnet34_layer1','resnet34_layer2','resnet34_layer3', 'resnet34', 'resnet50'
#clip-vit-large-patch14, clip-vit-large-patch14-inter
    'custom_pre_trained_weights': os.path.join(
    '/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/mnist',
    'resnet18/checkpoints/best-resnet18-mnist-epoch=05-val_loss=0.0181.ckpt'
), #None, see options below
    'model_structure': 'SequenceQuestionnaireModel',

    #Transforms definitions
    'custom_transform': 'pad_resize_normalize', #None, #if not None overrides the transform defined for the model with ta custom one
    'norm_mu': [0.06040578708052635, 0.06040578708052635, 0.06040578708052635],
    'norm_std': [0.23823712766170502, 0.23823712766170502, 0.23823712766170502],
    'apply_augmentation': None, #None, 'random_crop_half' ; if data_modality is a list the transform for each view mode will be determined
    #in the code based on the view name
    'invert_color':True,
    
    #Training params definition
    'lr_backbone': 1e-4,
    'lr_classifier_head': 1e-3,
    'lr_scheduling': 'cosine', #'cosine' # 'cosine', 'step', None
    'batch_size': 8,
    'num_epochs': 250,
    'patience': 250,
    'eta_min_cosine': 1e-6,
    'weight_decay': 1e-2, #0.05 (swi) #1e-2 (resnet)
    'warmup_fraction': 0.05,   # ~5% of total steps as warmup
    'input_size': 224,
    'layers_to_unfreeze': ['all','classifier'], #Update it for every model
    'seed': 42,
}
if isinstance(exp_params['data_modality'],list):
    exp_params['num_tiles'] = len(exp_params['data_modality'])
#options for custom_pre_trained_weights:
'''os.path.join(
    '/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/mnist',
    'resnet50/checkpoints/best-resnet18-mnist-epoch=28-val_loss=0.0197.ckpt'
),'''
#resnet50/checkpoints/best-resnet18-mnist-epoch=28-val_loss=0.0197.ckpt
#resnet18/checkpoints/best-resnet18-mnist-epoch=05-val_loss=0.0181.ckpt

SOURCE_PATTERN = os.path.join(SOURCE_PATH,exp_params['data_folder'])
SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")
SAVE_DEBUG_PATH = "/home/a_morelli/vscode_projects/model_training/data/debug_training"


huggingface_transform=True if exp_params['model'] in ['clip-vit-large-patch14-un', 'clip-vit-large-patch14-inter'] else False
exp_params['huggingface_transform'] = huggingface_transform

define_optimization_groups = [
        {'names': ['layer1','vision_model.conv1','vision_model.bn1'],'lr': 1e-5, 'lr_name': 'lr_1'},
        {'names': ['layer2'],'lr': 3e-5, 'lr_name': 'lr_2'},
        {'names': ['layer3'],'lr': 1e-4, 'lr_name': 'lr_3'},
        {'names': ['layer4'],'lr': 3e-4, 'lr_name': 'lr_4'},
        {'names': ['classifier'], 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_head'},
    ] # or None or other configurations fo other models'''


OUTPUT_PATH = os.path.join(SOURCE_PATH,f"{exp_params['model']}_model_results")
CHECKPOINT_PATH = os.path.join(OUTPUT_PATH, "checkpoints")
DEBUG_IMGS = True
VERBOSE = True
CLASS_COL = 'diag_park_final1_quest'

#if you need the following variables load them from the csv_data ->
#last available questionnaire for the subject, last_q = id_row['last_avail_q']
#variables_to_add = ['etudegp', 'profq2', 'lateralite', 'relative_age', 'birth_date', 'follow_up_time']
#filter these first when rebalancing: warning==1 if the control was selected with at least -> "at_least_warning": at_least_warning,
    
def main():
    args = get_args()
    worker = args.num_workers
    prefetch_factor = 2 if worker > 0 else None

    #set seed with lightning for reproducibility
    L.seed_everything(exp_params['seed'], workers=True)
    
    #load grid_files for selecting chunks from the images during the dataloading
    grid_dict = load_grid_dict(exp_params)

    #exclude controls from the training if i want to reduce the asimmetry of the dataset (for example if i want to have a 1:1 ratio between cases and controls)
    exclusion_set, val_exclusion_set, num_0, num_1 = prepare_exclusion_sets(exp_params,verbose=VERBOSE)

    write_log, current_version = logging_initialization() 

    model, transform = model_initialization(write_log,exp_params, verbose=VERBOSE)
    
    train_loader,val_loader,train_dataset,val_dataset= prepare_loaders(worker,prefetch_factor,exp_params,exclusion_set,val_exclusion_set, 
                                                                       grid_dict, transform)
    
    if DEBUG_IMGS:
        debug(train_loader,val_loader,exp_params)
    

    example_input_array = ModelPD.make_example_input(k=exp_params['num_tiles'], n_slots=13)
    lit_model = ModelPD(write_log,model=model, num_0=num_0, num_1=num_1, num_classes=exp_params['num_classes'],lr_backbone=exp_params['lr_backbone'], 
                         lr_classifier_head=exp_params['lr_classifier_head'], example_input_array=example_input_array, 
                         opt_groups=define_optimization_groups, num_epochs=exp_params['num_epochs'], lr_scheduling=exp_params['lr_scheduling'],
                         balancing_factor=exp_params['balancing_factor'], balanced_data=exp_params['balanced_data'], use_balanced_weights=exp_params['use_balanced_weights'], 
                         weight_decay=exp_params['weight_decay'], warmup_fraction=exp_params['warmup_fraction'], 
                         eta_min_cosine=exp_params['eta_min_cosine'], batch_size=exp_params['batch_size'])


    
    trainer, metrics_tracker = trainer_definition(current_version, exp_params)
    
    exp_params['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    exp_params['best_epoch'] = "N/A (Cancelled)"
    exp_params['best_val_loss'] = "N/A (Cancelled)"
    exp_params['best_val_acc'] = "N/A (Cancelled)"
    exp_params['best_train_acc'] = "N/A (Cancelled)"
    exp_params['current_version'] = current_version

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # If it finishes successfully, update the metrics dictionary with the real values
    exp_params["best_epoch"] = metrics_tracker.best_epoch
    exp_params["best_val_loss"] = metrics_tracker.best_val_loss
    exp_params["best_val_acc"] = metrics_tracker.best_val_acc
    exp_params["best_train_acc"] = metrics_tracker.best_train_acc
    
    # Save normal log
    save_experiment_logs(exp_params)
        
    



#### HELPER FUCNTIONS ####

def trainer_definition(current_version, exp_params):
    # 2. Setup Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",               # Monitor validation loss
        dirpath=os.path.join(CHECKPOINT_PATH,f'v_{current_version}'),           # Directory where weights will be saved
        filename="best-{epoch:02d}-{val_loss:.2f}",            # Filename for the best model
        save_top_k=1,                     # Save only the 1 best model
        mode="min",                       # Stop when val_loss stops minimizing
        save_last=False ,                   # Automatically creates 'last.ckpt' every epoch
        #enable_version_counter=False
    )

    periodic_ckpt = ModelCheckpoint(
        dirpath=os.path.join(CHECKPOINT_PATH, f'v_{current_version}'),
        filename="latest-{epoch:02d}",
        save_top_k=1,
        every_n_epochs=1,
        monitor=None,          # no metric -> saves the current/last state each time
    )

    # 3. Setup Early Stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=exp_params['patience'],
        mode="min",
        verbose=True
    )

    metrics_tracker = BestMetricTracker()
    
    # 1. Construct clean paths (removed trailing slash, kept version as a string/int)
    log_root = os.path.join(SOURCE_PATH, 'tensor_board_logging')
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

    # 4. Initialize Trainer and Fit
    trainer = L.Trainer(
        max_epochs=exp_params['num_epochs'],
        logger = tb_logger,
        accelerator="auto",                # Automatically selects GPU/CPU/MPS
        callbacks=[checkpoint_callback, early_stop_callback, metrics_tracker, periodic_ckpt],
        profiler="simple",  # Add this line to get a performance summary
        enable_progress_bar=False  # Remove this CPU overhead
    )
    # if you want you can set anothr kind of logger (not tensorboard but csv ..)
    return trainer, metrics_tracker

def debug(train_dataloader,val_dataloader,exp_params):
    batch = next(iter(train_dataloader))
    debug_print_batch_meta(batch)

    debug_images_PD(mean=exp_params['norm_mu'], std=exp_params['norm_std'], loader=train_dataloader, out_dir=os.path.join(SAVE_DEBUG_PATH,'train'))
    debug_images_PD(mean=exp_params['norm_mu'], std=exp_params['norm_std'], loader=val_dataloader, out_dir=os.path.join(SAVE_DEBUG_PATH,'val'))

def prepare_loaders(worker,prefetch_factor,exp_params,exclusion_set,val_exclusion_set, grid_dict,transform):
    print(f"Reading from {SHARD_PATTERN_train} and {SHARD_PATTERN_val}..")
    augmentation_transform = get_augmentation_transform(exp_params)
    print("Augmentation transform:", augmentation_transform)
    #assert 1==0 , "STOPPED HERE TO CHECK AUGMENTATION TRANSFORM"

    train_dataset = prepare_PD_dataset(SHARD_PATTERN_train, split_workers=True, batch_size=exp_params['batch_size'], transform=transform, exclusion_set=exclusion_set, modality=exp_params['data_modality'],
                       huggingface_transform=exp_params['huggingface_transform'],augmentation_transform=augmentation_transform, 
                       invert_color=exp_params['invert_color'],n_views=exp_params['num_tiles'], grid_dict = grid_dict, 
                       censor_time=exp_params['censor_time'])
    val_dataset = prepare_PD_dataset(SHARD_PATTERN_val, split_workers=True, batch_size=exp_params['batch_size'], transform=transform, exclusion_set=val_exclusion_set, modality=exp_params['data_modality'],
                       huggingface_transform=exp_params['huggingface_transform'],augmentation_transform=augmentation_transform, 
                       invert_color=exp_params['invert_color'],n_views=exp_params['num_tiles'], grid_dict = grid_dict,
                       censor_time=exp_params['censor_time'])
    
    train_loader = DataLoader(
        train_dataset, 
        num_workers=worker, 
        batch_size=None, 
        prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        num_workers=worker, 
        batch_size=None, 
        prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=True
    )
    return train_loader, val_loader, train_dataset, val_dataset

def model_initialization(write_log,exp_params, verbose=True):
    backbone,transform = get_model(name=exp_params['model'], pretrained=True, 
                                   custom_pre_trained_weights=exp_params['custom_pre_trained_weights'])
    print("############# Model backbone loaded! #############")
    transform = get_transforms(exp_params, transform)
    out=test_output(exp_params['input_size'], backbone)
    in_features = out.shape[1]  
    
    if exp_params['model_structure'] == 'SequenceQuestionnaireModel':
        model = SequenceQuestionnaireModel(backbone,feat_dim=in_features, n_classes=exp_params['num_classes'], n_slots=13, 
                                           d_model=512, n_heads=8, n_layers=4, ff_mult=4, view_agg='attention', dropout=0.1)
    else:
        raise ValueError(f"Unknown model_structure: {exp_params['model_structure']}")
    
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

def prepare_exclusion_sets(exp_params,verbose=True):
    exclusion_set = set()
    val_exclusion_set = set()
     
    csv_data = pd.read_parquet(exp_params['list_of_ids_paths'])
    if verbose:
        print("Initial CSV data loaded. First row example:")
        for col in csv_data.columns:
            print(f"{col}: {csv_data[col].iloc[0]}")
        print('#' * 50)
        print("Unique subjects in the dataset:", csv_data['unique_id'].nunique())
        print("Unique subjects in training set:", csv_data[csv_data['split'] == 'train']['unique_id'].nunique())
        print("Unique subjects in validation set:", csv_data[csv_data['split'] == 'val']['unique_id'].nunique())
        print('#' * 50)
        print("Class distribution in the entire dataset:\n", csv_data[CLASS_COL].value_counts())
        print('#' * 50)
    
    #compute the number of samples for each class in the training set
    num_0 = len(csv_data[(csv_data[CLASS_COL] == 0) & (csv_data['split'] == 'train')])
    num_1 = len(csv_data[(csv_data[CLASS_COL] == 1) & (csv_data['split'] == 'train')])

    if exp_params['balanced_data']:
        exclusion_set = generate_exclusion_set_PD(csv_data,exp_params, split='train') 
    if exp_params['balance_validation']:
        val_exclusion_set = generate_exclusion_set_PD(csv_data,exp_params, split='val')
    if verbose:
        print(len(exclusion_set), "samples will be excluded from the training set to achieve balancing.")
        print('#' * 50)
    return exclusion_set, val_exclusion_set, num_0,num_1

def load_grid_dict(exp_params):
    if exp_params['use_grid']:
        """Load the grid dictionary from a pickle file."""
        with open(exp_params['grid_dict_path'], "rb") as f:
            grid_dict = pickle.load(f)
        print("Grid dictionary loaded successfully. Number of entries:", len(grid_dict))
        return grid_dict
    else:
        print("Grid usage is disabled. No grid dictionary will be loaded.")
        return None

def get_transforms(exp_params, transform):
    if exp_params['custom_transform'] is None:
        return transform
    elif exp_params['custom_transform'] == 'pad_resize_normalize':
        return T.Compose(
                [
                    PadToSquare(fill=0),
                    T.Resize((exp_params['input_size'], exp_params['input_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=exp_params['norm_mu'], 
                                std=exp_params['norm_std']),
                ]
            )
    elif exp_params['custom_transform'] == 'resize_normalize':
        return T.Compose(
                [
                    T.Resize((exp_params['input_size'], exp_params['input_size'])),
                    T.ToTensor(),
                    T.Normalize(mean=exp_params['norm_mu'], 
                                std=exp_params['norm_std']),
                ]
            )
    else:
        raise ValueError(f"Unknown custom_transform: {exp_params['custom_transform']}")

def get_augmentation_transform(exp_params):
    def get_single_augmentation_transform(t_modality):
        if t_modality is None:
            return None
        elif t_modality == 'random_crop_half':
            return T.Compose([
                T.RandomCrop(
                    int(exp_params['input_size'] / 2), 
                    pad_if_needed=True, 
                    padding_mode='constant', 
                    fill=(255,255,255) # <-- White fill for RGB PIL images
                )
            ]) 
        elif t_modality == 'grid':
            return 'grid'
        else:
            raise ValueError(f"Unknown augmentation_transform: {exp_params['apply_augmentation']}")
    if isinstance(exp_params['data_modality'], list):
        '''in this case i have to return a list of transforms
        for the grid sampling it returns grid because i cnanot pre-load a transform'''
        print("Multiple data modalities detected. Applying augmentation transforms based on modality:")
        list_of_transforms = []
        modality_to_transform = {
            'digit_full': None,
            'digit_crop': 'random_crop_half',
            'digit':'grid',
            'text_full': None,
            'text_crop': 'random_crop_half',
            'text':'grid',
        }
        for t in exp_params['data_modality']:
            if t in modality_to_transform:
                list_of_transforms.append(get_single_augmentation_transform(modality_to_transform[t]))
            else:
                raise ValueError(f"Unknown data_modality: {t}")
        return list_of_transforms
    else:
        print("Single data modality detected. Applying augmentation transform:", exp_params['apply_augmentation'])
        return get_single_augmentation_transform(exp_params['apply_augmentation'])
    

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
    log_path = os.path.join(CHECKPOINT_PATH, "experiments_log.csv")
    
    # Modify the experiment name if it was cancelled to explicitly track it
    if status_suffix:
        params_dict["EXPERIMENT_NAME"] = f"{params_dict['EXPERIMENT_NAME']}_{status_suffix}"
        
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
    main()
    