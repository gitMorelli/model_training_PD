import tarfile
import time
import io
import torch
from PIL import Image
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

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP
from src.utils.visualization import debug_images_dataset

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"

LIST_OF_IDS_HANDEDNESS_PATH = os.path.join(SOURCE_PATH,"handedness_model_ids.csv")
folder_name = "png_resized_padded"
SOURCE_PATTERN = os.path.join(SOURCE_PATH,folder_name)

SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")


QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']

MODEL = 'resnet18' #'resnet18', 'custom_cnn'
unique_name = "test"
EXPERIMENT_NAME = f"{unique_name}_{MODEL}_{folder_name}"
OUTPUT_PATH = os.path.join(SOURCE_PATH,f"{MODEL}_extracted_features")
MODEL_LOAD_PATH = os.path.join(SOURCE_PATH,f"{MODEL}_model_results")
CHECKPOINT_PATH = os.path.join(MODEL_LOAD_PATH, "checkpoints")
checkpoint_to_load='v_11/best-epoch=00-val_loss=0.81.ckpt'#best.ckpt , None last.ckpt
DEBUG_IMGS = True
GET_STATISTICS = False
SEED=42
MODEL_MODALITY = 'feature_ext' # 'full', 'feature_ext', 'partial_unfr' 
CLASSIFICATION_HEAD = 'mlp' # 'linear', 'mlp', 'regularized_linear'

def prepare_handedness_dataset(shard_pattern, decode_approach='pil', load_in_memory=False, 
                               split_workers=True, batch_size=4, transform=None):
    
    # 1. Create a closure to pass arguments into the compose function
    def create_flattener(transform_func):
        def flatten_samples(src):
            """Takes an iterator of samples and yields multiple individual images."""
            for sample in src:
                label = None
                subject_id = "unknown"
                
                # Step A: Find the JSON file first to get the label and subject
                for key, value in sample.items():
                    if key.endswith("json"):
                        label = torch.tensor(value.get("label", -1), dtype=torch.long)
                        subject_id = value.get("subject", "unknown")
                        break # Found it, no need to keep checking other keys for json
                        
                # Step B: Filter at the sample level
                # If the sample is missing a label or has a -1 label, skip the whole sample
                if label is None or label.item() == -1:
                    continue
                    
                # Step C: Process and YIELD images one by one
                for key, value in sample.items():
                    if key.endswith((".png", ".jpg", ".jpeg")):
                        parts = key.split('.')
                        
                        # Expected format: q5.number_random.png
                        if len(parts) == 3:
                            questionnaire = parts[0]
                            modality_type = parts[1]
                            
                            try:
                                # Apply transformations
                                if transform_func is not None:
                                    img_tensor = transform_func(value) 
                                elif isinstance(value, torch.Tensor):
                                    img_tensor = value
                                else:
                                    img_tensor = T.ToTensor()(value)
                                
                                # YIELD ONE ROW AT A TIME
                                yield img_tensor, label, subject_id, questionnaire, modality_type
                                
                            except Exception as e:
                                print(f"Skipping corrupted image {key}: {e}")
                                
        return flatten_samples

    # 2. File gathering
    shard_files = glob.glob(shard_pattern)
    shard_files.sort()

    if load_in_memory:
        return 0 # Handle your in-memory logic here
        
    # 3. Build the WebDataset Pipeline
    dataset = wds.WebDataset(shard_files, shardshuffle=100)

    if split_workers:
        dataset = dataset.select(wds.split_by_worker)

    # 4. Apply the flattening and batching
    dataset = (dataset
        .decode(decode_approach)
        .compose(create_flattener(transform)) # This replaces .map() and .select()
        .batched(batch_size)
    )
    
    return dataset

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()


def model_selection(model,classification_head,model_modality,num_classes=2):
    if model in ['resnet18']:
        # 1. Load the pre-trained model
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 2. Modify the final layer first
        num_features = model.fc.in_features
        if classification_head == 'linear':
            model.fc = nn.Linear(num_features, num_classes)
        elif classification_head == 'regularized_linear':
            model.fc = nn.Sequential(
                nn.BatchNorm1d(num_features),    # Normalize the activations from the previous layer
                nn.Dropout(p=0.5),               # Randomly zero out 50% of the neurons to prevent overfitting
                nn.Linear(num_features, num_classes) # Final classification layer
            )
        elif classification_head == 'mlp':
            model.fc = CustomMLP(input_size=num_features, hidden_sizes=[16], output_size=num_classes, 
                                 activation='relu', dropout=0.5, batchnorm=True, with_input_norm='batch_norm')

        # Determine layers to unfreeze based on modality
        if model_modality == 'feature_ext':
            list_layers = ['fc']
        elif model_modality == 'partial_unfr':
            list_layers = ['layer4', 'fc']
        else:
            list_layers = None  # Flag to indicate EVERYTHING should be trainable

        # 3. Dynamic Selective Freezing Loop
        for name, param in model.named_parameters():
            # If list_layers is None, the 'or' statement short-circuits and sets True for all parameters
            if list_layers is None or any(layer in name for layer in list_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif model == 'custom_cnn':
        model = CustomBinaryCNN()

def prepare_dataloaders(decode_approach='pil',load_in_memory=False, split_workers=True, batch_size=4, transform = None, prefetch_factor=2, worker=8):
    data_loaders = {'train': None, 'val': None, 'test': None}
    datasets = {'train': None, 'val': None, 'test': None}
    train_dataset = prepare_handedness_dataset(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                            split_workers=split_workers, batch_size=batch_size, 
                                            transform=transform)
    val_dataset = prepare_handedness_dataset(SHARD_PATTERN_val, decode_approach=decode_approach, load_in_memory=load_in_memory,
                                                split_workers=split_workers, batch_size=batch_size, 
                                                transform=transform)
    
    datasets['train'] = train_dataset
    datasets['val'] = val_dataset

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

    data_loaders['train'] = train_loader
    data_loaders['val'] = val_loader
    return data_loaders, datasets

def run_inference_to_dataframe(model, dataloader, device):
    # 4. Initialize storage lists
    all_results = []
    all_targets = []      # Added to track true labels
    
    # Using simple Python lists for string metadata
    all_ids = []
    all_questionnaires = []
    all_modalities = []

    # Set model to evaluation mode
    model.eval()

    # 5. Run inference
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, ids, questionnaires, modalities = batch 
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Store tensors on CPU
            all_results.append(outputs.cpu())
            all_targets.append(targets.cpu()) # Store targets!
            
            # Use .extend() instead of .append() for strings. 
            # If batch size is 4, this adds 4 individual strings to the list
            # instead of adding 1 tuple of 4 strings.
            all_ids.extend(ids)
            all_questionnaires.extend(questionnaires)
            all_modalities.extend(modalities)

    # 6. Concatenate tensor lists into final tensors
    final_results = torch.cat(all_results, dim=0)
    final_targets = torch.cat(all_targets, dim=0)

    # 7. Convert to a Pandas DataFrame
    # If final_results is 2D (e.g. shape [N, num_classes]), .tolist() puts 
    # the array of logits/probabilities into a single cell as a Python list.
    df = pd.DataFrame({
        'subject_id': all_ids,
        'questionnaire': all_questionnaires,
        'modality': all_modalities,
        'true_label': final_targets.numpy(),
        'model_output': final_results.numpy().tolist() 
    })
    
    # Optional: If this is classification, you probably want the argmax prediction as its own column
    if final_results.ndim > 1 and final_results.shape[1] > 1:
        df['predicted_label'] = final_results.argmax(dim=1).numpy()

    return df


def main():
    args = get_args()
    worker = args.num_workers
    batch_size = 16
    prefetch_factor = 2
    decode_approach = 'pil'
    load_in_memory = False
    split_workers = True
    transform = None
    num_classes=2

    # Automatically use GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu = torch.cuda.get_device_name(device_id)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
        print(f"--- GPU DIAGNOSTICS ---")
        print(f"Active GPU: {gpu}")
        print(f"CUDA_VISIBLE_DEVICES: {visible}")

    #define a transform that normalize to the imagenet mean and std
    transform = T.Compose([
        #T.Resize((224, 224)),  # ResNet18 expects 224x224 input; in this case data ia already resized
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    print(f"Reading from {SHARD_PATTERN_train} and {SHARD_PATTERN_val} with decode approach '{decode_approach}' and load_in_memory={load_in_memory}")
    
    
    data_loaders,datasets = prepare_dataloaders(decode_approach=decode_approach, load_in_memory=load_in_memory, split_workers=split_workers, 
                        batch_size=batch_size, transform=transform, prefetch_factor=prefetch_factor, worker=worker)
    if DEBUG_IMGS:
        # Iterate through the first few items to see what labels are coming out
        for i, (img, label, key) in enumerate(datasets['train']):
            print(f"Sample {i}: Label {label}, Key {key}")
            if i > 10: break
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        debug_images_dataset(datasets['train'], output_path="data/anteprima_dataset.png", num_immagini=16, mean=None, std=None)
    
    #raise Exception("Debugging: Stopping after dataset preparation and image debugging. Check 'anteprima_dataset.png' for a visual preview of the data and verify labels in the console output.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_selection(MODEL,CLASSIFICATION_HEAD,MODEL_MODALITY,num_classes=num_classes)
    model.to(device)
    # if you want you can set anothr kind of logger (not tensorboard but csv ..)
    ckpt_path=os.path.join(CHECKPOINT_PATH,checkpoint_to_load) 
    # 3. Load the checkpoint file (weights are skipped, only reading metadata)
    checkpoint = torch.load(ckpt_path, map_location=device, weigthts_only=True)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # If it's just the raw weights
        model.load_state_dict(checkpoint)
    model.eval()  # Set the model to evaluation mode
    
    splits = ['train', 'val']
    dataframes = []
    for split in splits:
        print(f"Extracting representations for {split} set...")
        dataloader = data_loaders[split]
        df = run_inference_to_dataframe(dataloader, device, model)
        df['split'] = split
        dataframes.append(df.copy())
        del df
    final_df = pd.concat(dataframes, ignore_index=True)

    final_df.to_parquet(os.path.join(OUTPUT_PATH,f"{EXPERIMENT_NAME}_results.parquet"))

    #merge with the original csv data
    


if __name__ == "__main__":
    main()
    