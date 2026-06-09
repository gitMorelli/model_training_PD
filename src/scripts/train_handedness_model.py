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
import random
import shutil
#from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
import re
import signal
import sys

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset
from src.utils.data_loading_utils import prepare_handedness_dataset, prepare_handedness_dataset_all, generate_exclusion_set_val
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, TiledJoinedModels
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_dataset

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"

#LIST_OF_IDS_HANDEDNESS_PATH = os.path.join(SOURCE_PATH,"handedness_model_ids.csv")
LIST_OF_IDS_HANDEDNESS_PATH = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv"

#data_folder = "png_resized_padded_whitebg", "all_png_resized_padded" 
data_folder = "all_no_grids_png_whitebg" 
SOURCE_PATTERN = os.path.join(SOURCE_PATH,data_folder)

SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")


#QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(q) for q in range(1,14)]

#Model definition
MODEL = 'resnet50' #'resnet18', 'custom_cnn', 'resnet34_layer1','resnet34_layer2','resnet34_layer3', 'resnet34', 'resnet50'
#clip-vit-large-patch14, clip-vit-large-patch14-inter
huggingface_transform=True if MODEL in ['clip-vit-large-patch14-un', 'clip-vit-large-patch14-inter'] else False
transform_override = True #if true overrides the transform defined for the model with ta custom one
CLASSIFICATION_HEAD = 'linear' #'MLPClassifier1'#'MLPClassifier1' # 'linear', 'regularized_linear', 'MLPClassifier1'
PARAMS = {
    'dropout': 0.2,
    'hidden_sizes': [32],
    'with_input_norm': 'batch_norm'
}
lr_backbone=1e-4
lr_classifier_head=1e-3 
lr_scheduling = None #'cosine' # 'cosine', 'step', None
ETA_MIN_COSINE = 1e-6
num_epochs = 50
patience = 20
layers_to_unfreeze = ['all','classifier'] #Update it for every model
define_optimization_groups = [
        {'names': ['layer1','vision_model.conv1','vision_model.bn1'],'lr': 1e-5, 'lr_name': 'lr_1'},
        {'names': ['layer2'],'lr': 3e-5, 'lr_name': 'lr_2'},
        {'names': ['layer3'],'lr': 1e-4, 'lr_name': 'lr_3'},
        {'names': ['layer4'],'lr': 3e-4, 'lr_name': 'lr_4'},
        {'names': ['classifier'], 'lr': lr_classifier_head, 'lr_name': 'lr_head'},
    ] # or None or other configurations fo other models
custom_pre_trained_weights = os.path.join(
    '/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/mnist',
    'resnet50/checkpoints/best-resnet18-mnist-epoch=28-val_loss=0.0197.ckpt'
)

#None
input_size = 224

TEST = 'all_with_pre_trained_weights_resnet50' #'balanced_loss', 'balanced_data', 'balanced_data_and_loss'
EXPERIMENT_NAME = f"{MODEL}_{data_folder}"
OUTPUT_PATH = os.path.join(SOURCE_PATH,f"{MODEL}_model_results")
TRAIN = True  # Set to False to skip training and only run validation evaluation
CHECKPOINT_PATH = os.path.join(OUTPUT_PATH, "checkpoints")
checkpoint_to_load='v_1/best-epoch=01-val_loss=0.69.ckpt'#best.ckpt , None last.ckpt
DEBUG_IMGS = True
SEED=42
DATA_MODALITY = 'all' # 'X', 'text', 'digit', 'all' (all returns 3x3x224x224 elements instead of 3x224x224)
NUM_tiles = 1

BALANCED_DATA = True
USE_BALANCED_WEIGHTS = False
BALANCING_FACTOR = 1
MAJORITY_CLASS_ID = 0


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()


class LitModel(L.LightningModule):
    def __init__(self, write_log,model,num_0, num_1,num_classes=2, lr_backbone=1e-4,
                 lr_classifier_head=1e-3,example_input_array=torch.randn(1, 3, 224, 224),
                 opt_groups=None,num_epochs=10,lr_scheduling='cosine'):
        super().__init__()
        self.save_hyperparameters()
        self.opt_groups = opt_groups

        self.num_1 = num_1
        self.num_0 = num_1 * BALANCING_FACTOR if BALANCED_DATA else num_0
        self.total = self.num_0 + self.num_1

        #cross entropy
        weight_0 = self.total / (2 * self.num_0)
        weight_1 = self.total / (2 * self.num_1)
        # For BCE loss, the weight is applied to the positive class (Left-handed / Class 1)
        # Formula: pos_weight = majority_class_count / minority_class_count
        pos_weight_val = self.num_0 / self.num_1 if self.num_1 > 0 else 1.0

        # Array matching [weight_for_class_0, weight_for_class_1]
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)
        #BCE loss
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight)
        '''
        Watch out for device mismatches: A common mistake is doing self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight])) without using register_buffer. 
        If you do that, the weight tensor stays on the CPU, and the moment your model moves to a GPU, your code will crash with a runtime device mismatch error. 
        Using self.register_buffer binds the tensor to the module's lifetime and device state seamlessly
        '''

        self.model = model

        if num_classes == 2:
            if USE_BALANCED_WEIGHTS:
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif num_classes == 1:
            if USE_BALANCED_WEIGHTS:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        
        self.num_classes = num_classes
        self.lr_backbone = lr_backbone
        self.lr_classifier_head = lr_classifier_head
        # Initialize attributes to store sample counts
        self.train_sample_count = 0
        self.write_log = write_log
        self.example_input_array = example_input_array
        self.lr_scheduling = lr_scheduling

    def forward(self, x):
        return self.model(x)
    
    # --- Epoch Start Hooks ---
    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.write_log(f"Device of model at start of training: {next(self.model.parameters()).device}")

        # Reset the counter at the beginning of every training epoch
        self.train_sample_count = 0

    def training_step(self, batch, batch_idx):
        inputs, labels, *_ = batch

        if self.num_classes == 1:
            labels = labels.float().unsqueeze(1)
        outputs = self(inputs)

        # Check for NaN/inf in model outputs
        if not torch.isfinite(outputs).all():
            self.write_log(f"Warning: NaN or inf detected in model outputs at step {self.trainer.global_step}.")
            # You might want to return or handle this case, e.g., by skipping the step
            return None

        # Accumulate the number of samples in the current batch
        self.train_sample_count += inputs.size(0)
        loss = self.criterion(outputs, labels)

        # Check for NaN/inf in loss
        if not torch.isfinite(loss):
            self.write_log(f"Warning: NaN or inf detected in loss at step {self.trainer.global_step}. Skipping update.")
            return None
        
        # Calculate accuracy
        if self.num_classes == 1:
            preds = (outputs > 0.0).float()
        else:
            _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data).float() / inputs.size(0)
        
        # Log metrics (Lightning tracks epoch averages automatically)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        # Log learning rates directly from the optimizer
        opt = self.optimizers()
        if self.opt_groups:
            for i, group in enumerate(self.opt_groups):
                self.log(group['lr_name'], opt.param_groups[i]['lr'], on_step=False, on_epoch=True)
        else:
            self.log("lr_backbone", opt.param_groups[0]['lr'], on_step=False, on_epoch=True)
            self.log("lr_classifier_head", opt.param_groups[1]['lr'], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, *_ = batch

        if self.num_classes == 1:
            labels = labels.float().unsqueeze(1)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        if self.num_classes == 1:
            preds = (outputs > 0.0).float()
        else:
            _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data).float() / inputs.size(0)
        
        # 'val_loss' must be logged so the callbacks can monitor it
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
    
    # --- Epoch End Hooks ---
    def on_train_epoch_end(self):
        # 1. Identify trainable layers and calculate their sizes
        trainable_layers_info = []
        total_trainable_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # param.shape gives the tensor dimensions (e.g., [512, 2])
                # param.numel() gives the total number of scalar elements in that tensor
                layer_str = f"  - {name} | Shape: {list(param.shape)} | Parameters: {param.numel():,}"
                trainable_layers_info.append(layer_str)
                total_trainable_params += param.numel()

        # Optional: Print to the terminal console so you can see it live during execution
        self.write_log(f"\n[Epoch {self.current_epoch + 1}] Total Trainable Parameters: {total_trainable_params:,}")
        
        # 2. Check if it is the first epoch and write everything to your log file
        if self.current_epoch in [0,1]:
            self.write_log(f"\n--- Epoch {self.current_epoch + 1} Summary ---")
            # Your existing tracking metadata
            self.write_log(f"Epoch {self.current_epoch + 1}: Total Training Samples Processed: {self.train_sample_count}\n")
            self.write_log(f"Expected total is: {self.total}\n")
            self.write_log(f"Class 0 samples: {self.num_0}, Class 1 samples: {self.num_1}\n")
            self.write_log(f"Balancing Factor: {BALANCING_FACTOR}\n")
            self.write_log(f"Balanced Data: {BALANCED_DATA}, Use Balanced Weights: {USE_BALANCED_WEIGHTS}\n")
            self.write_log(f"Weights for Loss Function: {self.class_weights.tolist()}\n")
            
            # New: Append the model architecture specifics
            self.write_log("\n--- Trainable Model Architecture Summary ---\n")
            self.write_log(f"Total Trainable Parameters Count: {total_trainable_params:,}\n")
            self.write_log("Trainable Layers Structure:\n")
            for layer_info in trainable_layers_info:
                self.write_log(f"{layer_info}\n")

    def configure_optimizers(self):
        if self.opt_groups:
            param_groups = []
            for group in self.opt_groups:
                params = [param for name, param in self.model.named_parameters() 
                                   if any(key in name.lower() for key in group['names']) and param.requires_grad]
                lr = group['lr']
                lr_name = group['lr_name'] 
                param_groups.append({'params': params, 'lr': lr})
                
        else:
            backbone_params = []
            head_params = []
            
            # Segregate parameters based on whether they belong to the classification head
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Adjust the string matching here depending on how 'JoinedModels' names your layers.
                # Usually, new heads are named 'classifier', 'fc', or 'head'.
                if any(key in name.lower() for key in ['classifier']):
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            param_groups = [
                {'params': backbone_params, 'lr': self.lr_backbone},  # All existing ResNet18 layers
                {'params': head_params, 'lr': self.lr_classifier_head}       # Only the new final linear layer
            ]      
            '''# Apply a 10x smaller learning rate to the backbone
            optimizer = torch.optim.Adam([
                {'params': backbone_params, 'lr': self.lr_backbone},
                {'params': head_params, 'lr': self.lr_classifier_head}
            ])'''

        optimizer = optim.AdamW(param_groups, weight_decay=1e-2)

        if self.lr_scheduling == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.num_epochs,
                eta_min=ETA_MIN_COSINE,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val_loss",
                },
            }
        else:
            return {"optimizer": optimizer}

    def predict_step(self, batch, batch_idx):
        inputs, labels, subject_id,*_ = batch
        outputs = self(inputs)

        if self.num_classes == 1:
            preds = (outputs > 0.0).float()
            probs = torch.sigmoid(outputs)  # Get probabilities for the positive class
        else:
            _, preds = torch.max(outputs, 1)
            # Convert logits to probabilities for both classes
            probs = torch.softmax(outputs, dim=1)
        
        # Detach and move to CPU to avoid hoarding GPU memory
        return {
            "probs": probs.detach().cpu(),
            "preds": preds.detach().cpu(), 
            "labels": labels.detach().cpu(),
            "subject_id": subject_id  # Assuming subject_id is already a CPU tensor or a list of strings
        }
    
    def on_after_backward(self):
        # This hook is called after loss.backward() and before optimizer.step()
        # We check gradients only on the first batch of the first training epoch
        if self.trainer.global_step == 0:
            self.write_log("\n--- Gradient Check (First Batch) ---")
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Print the mean absolute gradient for key layers
                    if 'layer4' in name or 'classifier' in name:
                        grad_abs_mean = param.grad.abs().mean().item()
                        self.write_log(f"Layer '{name}': Mean Abs Gradient = {grad_abs_mean:.2e}")
                        if grad_abs_mean < 1e-8:
                            self.write_log(f"  -> WARNING: Potential vanishing gradient in layer {name}")
            self.write_log("-------------------------------------\n")

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

def melt_df(df,modality):
    exclusion_set = set()
    avail_columns=[f'q_{q}_num_{modality}' for q in QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS]
    df_source = df[['ident_projet', 'lateralite','split'] + avail_columns]
    df_long = df_source.melt(
        id_vars=['ident_projet', 'lateralite','split'], 
        value_vars=avail_columns,
        var_name='original_col', 
        value_name='score'
    )
    print(f"Length of melted df before filtering: {len(df_long)}")

    # 3. Extract the 'q' number from the column name
    # This regex looks for 'q_' followed by digits at the start of the string
    df_long['questionnaire'] = df_long['original_col'].str.extract(r'^q_(\d+)_').astype(int)

    df_long['ident_projet'] = df_long['ident_projet'].astype(str) + '_' + df_long['questionnaire'].astype(str)

    # 2. Filter rows where the score/value is >= 1
    df_filtered = df_long[df_long['score'] < 1]
    ident_projets_to_exclude = set(df_filtered['ident_projet'].unique())
    df_long = df_long[df_long['score'] >= 1]

    print(f"Length of melted df after filtering: {len(df_long)}")

    # 4. Drop the temporary columns to get your final desired structure
    new_df = df_long[['ident_projet', 'lateralite','split']].reset_index(drop=True)

    return new_df,ident_projets_to_exclude

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
    batch_size = 32
    prefetch_factor = 2
    decode_approach = 'pil'
    load_in_memory = False
    split_workers = True
    transform = None
    num_classes=1 #1 for BCE loss, 2 for crossentropy
    exclusion_set = set() # you can add here the ids you want to exclude from the dataset (for example because they are corrupted or for debugging purposes)
    val_exclusion_set = set()
    apply_augmentation = True
    invert_color=True

    if NUM_tiles > 1 and DATA_MODALITY == 'all':
        print("Warning: Data modality = 'all' and NUM_tiles>1 are incompatible ")
        return 


    if apply_augmentation:
        #add a random crop transform without resizing
        augmentation_transform = T.Compose([
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
    csv_data, num_less_than_1_rows = melt_df(csv_data, modality=selection_modality)
    exclusion_set.update(num_less_than_1_rows)
    val_exclusion_set.update(num_less_than_1_rows)
    print(len(exclusion_set), "samples will be excluded from the dataset based on the num_filtering process for modality", selection_modality)
    #includes elements that are -1 (hence no extracted file)
    print("CSV after melting:", csv_data.head())
    #cols: 'ident_projet', 'lateralite', 'q_5_num_X', 'q_5_num_text', 'q_5_num_digit', 'split']

    train_data = csv_data[csv_data['split'] == 'train']
    print(f"Training samples with at least 1 chunck for modality {selection_modality}: {len(train_data)}")
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
        transform = T.Compose(
            [
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.06040578708052635, 0.06040578708052635, 0.06040578708052635], 
                            std=[0.23823712766170502, 0.23823712766170502, 0.23823712766170502]),
            ]
        )
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
                                                    invert_color=invert_color, n_views=NUM_tiles)
        else:
            train_dataset = prepare_handedness_dataset(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                                    split_workers=split_workers, batch_size=batch_size, 
                                                    transform=transform, modality=DATA_MODALITY, 
                                                    rate=rate, balanced_data=BALANCED_DATA, exclusion_set=exclusion_set, augmentation_transform=augmentation_transform,
                                                    invert_color=invert_color)
    if DEBUG_IMGS and TRAIN:
        # Iterate through the first few items to see what labels are coming out
        for i, (img, label, id,q,mode) in enumerate(train_dataset):
            print(f"Sample {i}: Label {label}, ID {id}, Q {q}, Mode {mode}")
            if i > 10: break
    
    #print(f"Number of dataset samples (train): {len(train_dataset)}")
    val_exclusion_set.update( generate_exclusion_set_val(csv_data, data_modality=selection_modality,
                                                   majority_class_id=MAJORITY_CLASS_ID, balancing_factor=BALANCING_FACTOR, 
                                                   label_col='lateralite', id_col='ident_projet', split='val') )
    if 'all' in EXPERIMENT_NAME:
        val_dataset = prepare_handedness_dataset_all(SHARD_PATTERN_val, decode_approach=decode_approach, load_in_memory=load_in_memory,
                                                     split_workers=split_workers, batch_size=batch_size,
                                                     transform=transform, modality=DATA_MODALITY, exclusion_set=val_exclusion_set, 
                                                     huggingface_transform=huggingface_transform, augmentation_transform=augmentation_transform,
                                                     invert_color=invert_color, n_views=NUM_tiles)
    else:
        val_dataset = prepare_handedness_dataset(SHARD_PATTERN_val, decode_approach=decode_approach, load_in_memory=load_in_memory,
                                                    split_workers=split_workers, batch_size=batch_size, 
                                                    transform=transform, modality=DATA_MODALITY, balanced_data=False, exclusion_set=val_exclusion_set,
                                                    augmentation_transform=augmentation_transform)
    if DEBUG_IMGS and TRAIN:
        #sample N data points at random from the train dataset, save them in an image with the corresponding label
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        n_stacked=NUM_tiles
        if DATA_MODALITY == 'all':
            n_stacked = 3
        debug_images_dataset(train_dataset, output_path="data/anteprima_dataset.png", num_immagini=16, mean=None, std=None, n_stacked=n_stacked)
 
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
                         opt_groups=define_optimization_groups, num_epochs=num_epochs, lr_scheduling=lr_scheduling)


    # 2. Setup Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",               # Monitor validation loss
        dirpath=os.path.join(CHECKPOINT_PATH,f'v_{current_version}'),           # Directory where weights will be saved
        filename="best-{epoch:02d}-{val_loss:.2f}",            # Filename for the best model
        save_top_k=1,                     # Save only the 1 best model
        mode="min",                       # Stop when val_loss stops minimizing
        save_last=True ,                   # Automatically creates 'last.ckpt' every epoch
        #enable_version_counter=False
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
        callbacks=[checkpoint_callback, early_stop_callback, metrics_tracker],
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
    