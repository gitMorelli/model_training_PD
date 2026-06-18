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
from src.utils.data_loading_utils import prepare_handedness_dataset, prepare_handedness_dataset_all, generate_exclusion_set_val
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, TiledJoinedModels
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_dataset
from src.utils.image_processing import ResizeLongestSide, PadToSquare
from src.utils.training_utils import LitModel

#PATHS
SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"
#LIST_OF_IDS_HANDEDNESS_PATH = os.path.join(SOURCE_PATH,"handedness_model_ids.csv")
#LIST_OF_IDS_HANDEDNESS_PATH = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs_w_sentences.csv"
LIST_OF_IDS_HANDEDNESS_PATH = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv"
#data_folder = "png_resized_padded_whitebg", "all_png_resized_padded" 
data_folder = "all_no_grids_png_whitebg" 
#data_folder = "all_full_sentences_png_whitebg" 
SOURCE_PATTERN = os.path.join(SOURCE_PATH,data_folder)
SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")
SAVE_DEBUG_PATH = "/home/a_morelli/vscode_projects/model_training/data/debug_training"


#QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(q) for q in range(1,14)]

#Model definition
MODEL = 'resnet50' #'swin_s' #'resnet18', 'custom_cnn', 'resnet34_layer1','resnet34_layer2','resnet34_layer3', 'resnet34', 'resnet50'
#clip-vit-large-patch14, clip-vit-large-patch14-inter
huggingface_transform=True if MODEL in ['clip-vit-large-patch14-un', 'clip-vit-large-patch14-inter'] else False
transform_override = True #if true overrides the transform defined for the model with ta custom one
CLASSIFICATION_HEAD = 'MLPClassifier1' #'MLPClassifier1'#'MLPClassifier1' # 'linear', 'regularized_linear', 'MLPClassifier1'
PARAMS = {
    'dropout': 0.6,
    'hidden_sizes': [64],
    'with_input_norm': 'batch_norm'
}
lr_backbone=1e-4
lr_classifier_head=1e-3 
lr_scheduling = 'cosine' #'cosine' # 'cosine', 'step', None
batch_size = 32
ETA_MIN_COSINE = 1e-6
WEIGHT_DECAY = 1e-2 #0.05 (swi) #1e-2 (resnet)
WARMUP_FRACTION = 0.05   # ~5% of total steps as warmup
num_epochs = 150
patience = 50
input_size = 224
layers_to_unfreeze = ['all','classifier'] #Update it for every model
define_optimization_groups = [
        {'names': ['layer1','vision_model.conv1','vision_model.bn1'],'lr': 1e-5, 'lr_name': 'lr_1'},
        {'names': ['layer2'],'lr': 3e-5, 'lr_name': 'lr_2'},
        {'names': ['layer3'],'lr': 1e-4, 'lr_name': 'lr_3'},
        {'names': ['layer4'],'lr': 3e-4, 'lr_name': 'lr_4'},
        {'names': ['classifier'], 'lr': lr_classifier_head, 'lr_name': 'lr_head'},
    ] # or None or other configurations fo other models'''
custom_pre_trained_weights = os.path.join(
    '/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/mnist',
    'resnet50/checkpoints/best-resnet18-mnist-epoch=28-val_loss=0.0197.ckpt'
)
#resnet50/checkpoints/best-resnet18-mnist-epoch=28-val_loss=0.0197.ckpt
#resnet18/checkpoints/best-resnet18-mnist-epoch=05-val_loss=0.0181.ckpt
#None



TEST = '' #'balanced_loss', 'balanced_data', 'balanced_data_and_loss'
EXPERIMENT_NAME = f"{MODEL}_{data_folder}"
OUTPUT_PATH = os.path.join(SOURCE_PATH,f"{MODEL}_model_results")
TRAIN = True  # Set to False to skip training and only run validation evaluation
CHECKPOINT_PATH = os.path.join(OUTPUT_PATH, "checkpoints")
checkpoint_to_load='v_1/best-epoch=01-val_loss=0.69.ckpt'#best.ckpt , None last.ckpt
DEBUG_IMGS = False
SEED=42
DATA_MODALITY = 'digit' # 'X', 'text', 'digit', 'all' (all returns 3x3x224x224 elements instead of 3x224x224)
NUM_tiles = 5
USE_GRID = True


BALANCED_DATA = True
USE_BALANCED_WEIGHTS = False
BALANCING_FACTOR = 1
MAJORITY_CLASS_ID = 0
THRESHOLD_NUM = 1

CUSTOM_TRANSFORM = T.Compose(
            [
                PadToSquare(fill=0),
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.06040578708052635, 0.06040578708052635, 0.06040578708052635], 
                            std=[0.23823712766170502, 0.23823712766170502, 0.23823712766170502]),
            ]
        )
'''T.Compose(
            [
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.06040578708052635, 0.06040578708052635, 0.06040578708052635], 
                            std=[0.23823712766170502, 0.23823712766170502, 0.23823712766170502]),
            ]
        )
'''

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

class BestMetricTracker(L.Callback):
    def __init__(self):
        super().__init__()
        self.best_train_acc = 0.0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        # Prevent tracking during the initial sanity check pass
        if trainer.sanity_checking:
            return

        # Fetch the logged metrics dictionary
        metrics = trainer.callback_metrics
        
        # Extract values (handling the _epoch suffix for training metrics)
        train_acc = metrics.get("train_acc_epoch") 
        val_loss = metrics.get("val_loss")
        val_acc = metrics.get("val_acc")
        
        current_epoch = trainer.current_epoch

        # Keep track of global maximums / minimums
        if train_acc is not None:
            self.best_train_acc = max(self.best_train_acc, train_acc.item())
        
        if val_acc is not None:
            self.best_val_acc = max(self.best_val_acc, val_acc.item())
            
        if val_loss is not None and val_loss.item() < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            # If you want the epoch where the best validation loss happened:
            self.best_epoch = current_epoch


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

def main():
    args = get_args()
    worker = args.num_workers
    prefetch_factor = 2
    decode_approach = 'pil'
    load_in_memory = False
    split_workers = True
    transform = None
    num_classes=1 #1 for BCE loss, 2 for crossentropy
    exclusion_set = set() # you can add here the ids you want to exclude from the dataset (for example because they are corrupted or for debugging purposes)
    val_exclusion_set = set()
    apply_augmentation = False
    invert_color=True

    if NUM_tiles > 1 and DATA_MODALITY == 'all':
        print("Warning: Data modality = 'all' and NUM_tiles>1 are incompatible ")
        return 
    if DATA_MODALITY == 'all' and USE_GRID:
        print("Warning: Data modality = 'all' and USE_GRID=True are incompatible ")
        return
    
    grid_dict = None
    if USE_GRID:
        with open("/mnt/beegfs02/scratch/a_morelli/datasets/rr_data_h5.pkl", "rb") as f:
            grid_dict = pickle.load(f)
        print("Grid dictionary loaded successfully. Number of entries:", len(grid_dict))

    if apply_augmentation:
        #add a random crop transform without resizing
        augmentation_transform = T.Compose([
            #ResizeLongestSide(448),
            T.RandomCrop(
                112, 
                pad_if_needed=True, 
                padding_mode='constant', 
                fill=(255,255,255) # <-- White fill for RGB PIL images
            )
        ]) 
    else:
        augmentation_transform = None

    if DATA_MODALITY == 'all':
        selection_modality = 'text' 
    else:
        selection_modality = DATA_MODALITY 
    csv_data = pd.read_csv(LIST_OF_IDS_HANDEDNESS_PATH)
    '''print('Test num_text exclusion:')
    print(csv_data.loc[csv_data['ident_projet'] == 'A4V2C8D0'].T)
    print("Columns in the CSV:", csv_data.columns.tolist())
    return'''
    csv_data, num_less_than_1_rows = melt_df(csv_data, modality=selection_modality, threshold=THRESHOLD_NUM)
    exclusion_set.update(num_less_than_1_rows)
    val_exclusion_set.update(num_less_than_1_rows)
    print(len(exclusion_set), "samples will be excluded from the dataset based on the num_filtering process for modality", selection_modality)
    #includes elements that are -1 (hence no extracted file)
    print("CSV after melting:", csv_data.head())
    #cols: 'ident_projet', 'lateralite', 'q_5_num_X', 'q_5_num_text', 'q_5_num_digit', 'split']

    train_data = csv_data[csv_data['split'] == 'train']
    print(f"Training samples with at least {THRESHOLD_NUM} chunck for modality {selection_modality}: {len(train_data)}")
    #get the number of samples for each class
    class_counts = train_data['lateralite'].value_counts()
    print(f"Class distribution in training set (after filtering for modality {selection_modality}):\n{class_counts}")
    num_0 = class_counts.get(0.0, 0)  # Count of class 0 (e.g., right-handed)
    num_1 = class_counts.get(1.0, 0)  # Count of class 1 (e.g., left-handed)
    rate = num_1/num_0 if num_0 > 0 else 0
    print(f"Number of samples per class: Class 0 = {num_0}, Class 1 = {num_1}; Total = {num_0 + num_1}")
    
    if BALANCED_DATA:
        exclusion_set.update( generate_exclusion_set_val(csv_data, data_modality=selection_modality,
                                                    majority_class_id=MAJORITY_CLASS_ID, balancing_factor=BALANCING_FACTOR, 
                                                    label_col='lateralite', id_col='ident_projet', split='train') )
    print(len(exclusion_set), "samples will be excluded from the training set to achieve balancing.")

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
    if TRAIN:
        with open(log_file,'w') as f:
            f.write("Log file for experiment version: " + str(current_version) + "\n")
    #define a writign function that creates the file if it doesn't exist and append if it exists and write only if TRAIN==True
    def write_log(message):
        if TRAIN:
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


    backbone,transform = get_model(name=MODEL, pretrained=True, custom_pre_trained_weights=custom_pre_trained_weights)
    print("############# Model backbone loaded! #############")
    if transform_override:
        transform = CUSTOM_TRANSFORM
    #check on which device the backbone is
    #print(f"Backbone device: {next(backbone.parameters()).device}")
    out=test_output(input_size, backbone)
    #print(out)
    in_features = out.shape[1]
    in_features*= 3 if DATA_MODALITY=='all' else 1
    in_features*= NUM_tiles if NUM_tiles > 1 else 1
    classificaton_head = get_classification_head(name=CLASSIFICATION_HEAD,in_features=in_features,num_classes=num_classes,**PARAMS)
    if DATA_MODALITY == 'all' or NUM_tiles>1:
        model = TiledJoinedModels(backbone, classificaton_head) #it gets the dimension by itself
    else:
        model = JoinedModels(backbone, classificaton_head)
    unfreeze_layers(model,layer_names=layers_to_unfreeze)

    write_log(f"Device of model after initialization: {next(model.parameters()).device}") # <-- ADD THIS LINE
    trainable_parameters_info = get_trainable_parameters_string(model)
    write_log("Model Architecture and Trainable Parameters right after initialization:")
    write_log(trainable_parameters_info)


    print(f"Reading from {SHARD_PATTERN_train} and {SHARD_PATTERN_val} with decode approach '{decode_approach}' and load_in_memory={load_in_memory}")
    if TRAIN:
        if 'all' in EXPERIMENT_NAME:
            train_dataset = prepare_handedness_dataset_all(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                                    split_workers=split_workers, batch_size=batch_size, 
                                                    transform=transform, modality=DATA_MODALITY, exclusion_set=exclusion_set, 
                                                    huggingface_transform=huggingface_transform, augmentation_transform=augmentation_transform,
                                                    invert_color=invert_color, n_views=NUM_tiles, grid_dict=grid_dict)
        else:
            train_dataset = prepare_handedness_dataset(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                                    split_workers=split_workers, batch_size=batch_size, 
                                                    transform=transform, modality=DATA_MODALITY, 
                                                    rate=rate, balanced_data=BALANCED_DATA, exclusion_set=exclusion_set, augmentation_transform=augmentation_transform,
                                                    grid_dict=grid_dict, invert_color=invert_color)
    if DEBUG_IMGS and TRAIN:
        # Iterate through the first few items to see what labels are coming out
        for i, (img, label, id,q,mode) in enumerate(train_dataset):
            print(f"Sample {i}: Label {label}, ID {id}, Q {q}, Mode {mode}")
            if i > 10: break
    
    #print(f"Number of dataset samples (train): {len(train_dataset)}")
    if BALANCED_DATA:
        val_exclusion_set.update( generate_exclusion_set_val(csv_data, data_modality=selection_modality,
                                                    majority_class_id=MAJORITY_CLASS_ID, balancing_factor=BALANCING_FACTOR, 
                                                    label_col='lateralite', id_col='ident_projet', split='val') )
    if 'all' in EXPERIMENT_NAME:
        val_dataset = prepare_handedness_dataset_all(SHARD_PATTERN_val, decode_approach=decode_approach, load_in_memory=load_in_memory,
                                                     split_workers=split_workers, batch_size=batch_size,
                                                     transform=transform, modality=DATA_MODALITY, exclusion_set=val_exclusion_set, 
                                                     huggingface_transform=huggingface_transform, augmentation_transform=augmentation_transform,
                                                     invert_color=invert_color, n_views=NUM_tiles, grid_dict=grid_dict)
    else:
        val_dataset = prepare_handedness_dataset(SHARD_PATTERN_val, decode_approach=decode_approach, load_in_memory=load_in_memory,
                                                    split_workers=split_workers, batch_size=batch_size, 
                                                    transform=transform, modality=DATA_MODALITY, 
                                                    balanced_data=False, exclusion_set=val_exclusion_set, augmentation_transform=augmentation_transform, 
                                                    grid_dict=grid_dict, invert_color=invert_color)
    if DEBUG_IMGS and TRAIN:
        #sample N data points at random from the train dataset, save them in an image with the corresponding label
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        n_stacked=NUM_tiles
        if DATA_MODALITY == 'all':
            n_stacked = 3
        debug_images_dataset(train_dataset, output_path=os.path.join(SAVE_DEBUG_PATH,'train.png'), num_immagini=16, mean=None, std=None, n_stacked=n_stacked)
        debug_images_dataset(val_dataset, output_path=os.path.join(SAVE_DEBUG_PATH,'train.png'), num_immagini=16, mean=None, std=None, n_stacked=n_stacked)
 
    print("Preparing dataloaders .. ")
    #raise Exception("Debugging: Stopping after dataset preparation and image debugging. Check 'anteprima_dataset.png' for a visual preview of the data and verify labels in the console output.")

    if TRAIN:
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

    if DATA_MODALITY == 'all':
        example_input_array = torch.randn(1, 3,3, 224, 224)  # For visualizing the graph in TensorBoard
    elif NUM_tiles > 1:
        example_input_array = torch.randn(1, NUM_tiles, 3, 224, 224)
    else:
        example_input_array = torch.randn(1, 3, 224, 224)
    lit_model = LitModel(write_log,model=model, num_0=num_0, num_1=num_1, num_classes=num_classes,lr_backbone=lr_backbone, 
                         lr_classifier_head=lr_classifier_head, example_input_array=example_input_array, 
                         opt_groups=define_optimization_groups, num_epochs=num_epochs, lr_scheduling=lr_scheduling,
                         balancing_factor=BALANCING_FACTOR, balanced_data=BALANCED_DATA, use_balanced_weights=USE_BALANCED_WEIGHTS, 
                         weight_decay=WEIGHT_DECAY, warmup_fraction=WARMUP_FRACTION, 
                         eta_min_cosine=ETA_MIN_COSINE, batch_size=batch_size)


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
        patience=patience,
        mode="min",
        verbose=True
    )

    metrics_tracker = BestMetricTracker()

    if TRAIN:
        # 1. Construct clean paths (removed trailing slash, kept version as a string/int)
        log_root = os.path.join(SOURCE_PATH, 'tensor_board_logging')
        version_dir = os.path.join(log_root, MODEL, 'version_'+str(current_version)) #tensorboard automatically adds 'version_' prefix, 
        #so we match that format here.

        # 2. Wipe the old folder if it exists
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)

        # 3. Initialize the logger
        tb_logger = TensorBoardLogger(
            save_dir=log_root,
            name=MODEL,
            log_graph=True,
            version=current_version  # Works perfectly as an integer or string
        )
    else:
        tb_logger=False
    '''
    csv_logger = CSVLogger(
        save_dir="text_logs/",
        name="experiment_1"
    )
    '''

    # 4. Initialize Trainer and Fit
    trainer = L.Trainer(
        max_epochs=num_epochs,
        logger = tb_logger,
        accelerator="auto",                # Automatically selects GPU/CPU/MPS
        callbacks=[checkpoint_callback, early_stop_callback, metrics_tracker, periodic_ckpt],
        profiler="simple",  # Add this line to get a performance summary
        enable_progress_bar=False  # Remove this CPU overhead
    )
    # if you want you can set anothr kind of logger (not tensorboard but csv ..)

    if TRAIN:
        experiment_params = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "MODEL": MODEL,
            "TEST": TEST,
            "EXPERIMENT_NAME": EXPERIMENT_NAME,
            "DATA_MODALITY": DATA_MODALITY,
            "MODEL_MODALITY": layers_to_unfreeze,
            "BALANCED_DATA": BALANCED_DATA,
            "USE_BALANCED_WEIGHTS": USE_BALANCED_WEIGHTS,
            "NUM_0": num_0,
            "NUM_1": num_1,
            "RATE": rate,
            "BALANCING_FACTOR": BALANCING_FACTOR,
            "batch_size": batch_size,
            "lr": [lr_backbone, lr_classifier_head],
            "num_epochs": num_epochs,
            "patience": patience,
            "current_version": current_version,
            "best_epoch": "N/A (Cancelled)",
            "best_val_loss": "N/A (Cancelled)",
            "best_val_acc": "N/A (Cancelled)",
            "best_train_acc": "N/A (Cancelled)",
        }

        def handle_slurm_cancel(signum, frame):
            print(f"\n[SLURM] Job received signal {signum} (Cancellation/Timeout). Logging parameters before exiting...")
            
            save_experiment_logs(experiment_params, status_suffix="CANCELLED")
            sys.exit(0)

        # Register the handler for SIGTERM (Slurm cancel/timeout)
        signal.signal(signal.SIGTERM, handle_slurm_cancel)

        # This replaces your entire training loop function
        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        #save experiment parameters in a dict 
        #get the best_epoch the model was saved at, the best val_loss and val_acc
        # If it finishes successfully, update the metrics dictionary with the real values
        experiment_params["best_epoch"] = metrics_tracker.best_epoch
        experiment_params["best_val_loss"] = metrics_tracker.best_val_loss
        experiment_params["best_val_acc"] = metrics_tracker.best_val_acc
        experiment_params["best_train_acc"] = metrics_tracker.best_train_acc
        
        # Save normal log
        save_experiment_logs(experiment_params)
        
    else:
        print("\n--- Starting Validation Evaluation ---")
    
        # 1. Gather predictions using the best checkpoint saved during training
        # Setting ckpt_path="best" tells Lightning to automatically find your top model
        ckpt_path=os.path.join(CHECKPOINT_PATH,checkpoint_to_load) 
        # 3. Load the checkpoint file (weights are skipped, only reading metadata)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # 4. Extract the exact epoch
        best_epoch = checkpoint["epoch"]
        print(f"The best model was saved at epoch: {best_epoch}")
        outputs = trainer.predict(lit_model, dataloaders=val_loader,ckpt_path = ckpt_path)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
        
        # 2. Concatenate all batch outputs into unified tensors
        all_probs = torch.cat([batch["probs"] for batch in outputs])
        all_preds = torch.cat([batch["preds"] for batch in outputs])
        all_labels = torch.cat([batch["labels"] for batch in outputs])

        #show 10 example outputs and the corresponding expected label
        print("\n--- Sample Predictions ---")
        for i in range(10):
            print(f"Probs: {all_probs[i].tolist()} | Predicted: {all_preds[i].item()} | True Label: {all_labels[i].item()}")
        
        # 3. Convert to numpy arrays for statistics calculation
        y_pred = all_preds.numpy()
        y_true = all_labels.numpy()
        
        # 4. Compute and display metrics using scikit-learn
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n================ VALIDATION STATISTICS ================")
        print("\n--- Classification Report ---")
        # Adjust target_names to match your two classes if needed
        print(classification_report(y_true, y_pred, target_names=["Right", "Left"]))
        
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_true, y_pred))
        print("=======================================================")

        
if __name__ == "__main__":
    main()
    