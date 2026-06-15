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
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import json

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset, prepare_handedness_dataset, prepare_handedness_dataset_all
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, get_model, test_output, get_classification_head, ConcatenateViews
from src.utils.model_utils import JoinedModels, unfreeze_layers, load_backbone_from_lightning_ckpt
from src.utils.visualization import debug_images_dataset

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"

LIST_OF_IDS_HANDEDNESS_PATH = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv"

#folder_name = "png_resized_padded", "png_resized_padded_whitebg"
folder_name = "all_no_grids_png_whitebg"  
SOURCE_PATTERN = os.path.join(SOURCE_PATH,folder_name)

SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")

MODEL = 'clip-vit-large-patch14-inter' #'resnet18' #'resnet18', 'custom_cnn'
input_size = 224

#load from standard pre_trained model (eg full resnet)
custom_pre_trained_weights = None 

#load the backbone weights from one of your backbone+classifier weights
#MODEL_LOAD_PATH = os.path.join(SOURCE_PATH,f"{MODEL}_model_results")
#CHECKPOINT_PATH = os.path.join(MODEL_LOAD_PATH, "checkpoints")
checkpoint_to_load= None #'v_11/best-epoch=00-val_loss=0.81.ckpt'#best.ckpt , None last.ckpt

OUTPUT_PATH = os.path.join(SOURCE_PATH,'feature_extraction',f"{MODEL}_extracted_features")
DEBUG_IMGS = True
GET_STATISTICS = False
SEED=42
DATA_MODALITY = "all" # text,digit,X,all 
NUM_tiles = 1 #num tiles to concatenate in a single extraction
NUM_augmentations = 2 #number of times to process the same image with a random augmentation
invert_color = True
exclusion_set = set()
huggingface_transform = False
huggingface_transform=True if MODEL in ['clip-vit-large-patch14-un', 'clip-vit-large-patch14-inter'] else False
apply_augmentation = True
transform_override = True
transform_override = False if MODEL in ['clip-vit-large-patch14-un', 'clip-vit-large-patch14-inter'] else transform_override
unique_name = "clip"
SAVE_FOLDER = os.path.join(OUTPUT_PATH, f"{unique_name}_{MODEL}_{DATA_MODALITY}_tiles{NUM_tiles}_aug{NUM_augmentations}")

current_user = "Andrea Morelli"
generated_at = datetime.now().isoformat()
custom_metadata = {
    'current_user': current_user,
    'generated_at': generated_at,
    'model': MODEL,
    'apply_augmentation': apply_augmentation,
    'invert_color': invert_color,
    'custom_pre_trained_weights': custom_pre_trained_weights,
    'checkpoint_to_load': checkpoint_to_load,
}

'''
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
'''

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

def prepare_dataloaders(
        decode_approach='pil',
        load_in_memory=False, 
        split_workers=True, 
        batch_size=4, 
        transform = None, 
        augmentation_transform=None, 
        prefetch_factor=2, 
        worker=8,
        exclusion_set=set(),
        data_modality='text',
        huggingface_transform=False,
        invert_color=invert_color,
        n_views=1):
    data_loaders = {'train': None, 'val': None, 'test': None}
    datasets = {'train': None, 'val': None, 'test': None}
    train_dataset = prepare_handedness_dataset_all(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                                    split_workers=split_workers, batch_size=batch_size, 
                                                    transform=transform, modality=data_modality, exclusion_set=exclusion_set, 
                                                    huggingface_transform=huggingface_transform, augmentation_transform=augmentation_transform,
                                                    invert_color=invert_color, n_views=n_views)
    val_dataset = prepare_handedness_dataset_all(SHARD_PATTERN_val, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                                    split_workers=split_workers, batch_size=batch_size, 
                                                    transform=transform, modality=data_modality, exclusion_set=exclusion_set, 
                                                    huggingface_transform=huggingface_transform, augmentation_transform=augmentation_transform,
                                                    invert_color=invert_color, n_views=n_views)
    
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

def run_inference_to_dataframe(
    model,
    dataloader,
    device,
    data_modality,
    num_tiles,
    num_augmentations,
    output_path=None,
    custom_metadata={},
    auto=True,
):
    """
    Extract backbone representations and (optionally) save them to disk.
 
    Returns
    -------
    metadata : pd.DataFrame
        One row per (subject_id, questionnaire, augmentation_version), with
        columns: subject_id, questionnaire, augmentation_version,
        data_modality, modality, true_label. NO representation vectors here.
    representations : np.ndarray  (float32)
        Row i lines up exactly with row i of `metadata`. Shape depends on the
        (data_modality, num_tiles) configuration:
            single modality, tiles == 1 -> (num_rows, N)
            data_modality == 'all'      -> (num_rows, 3, N)   axis1: text,digit,cX
            single modality, tiles == k -> (num_rows, k, N)   axis1: tile0..k-1
 
    If `output_path` is given (a path stem, e.g. "out/run01"), writes:
        <stem>_metadata.parquet         the metadata table
        <stem>_representations.npy      the aligned float32 array
        <stem>_layout.json              shape, dtype, axis-1 meaning
    """
 
    # ---- 1. Validate the allowed configuration ------------------------------
    if data_modality == 'all' and num_tiles != 1:
        raise ValueError(
            "data_modality='all' is only valid with num_tiles=1 "
            f"(got num_tiles={num_tiles})."
        )
 
    # ---- 2. Decide the axis-1 layout up front -------------------------------
    if data_modality == 'all':
        axis1_labels = ['text', 'digit', 'X']   # axis-1 indices 0,1,2
        split = True
        expected_ndim = 3
    elif num_tiles == 1:
        axis1_labels = None                        # output is (B, N), no axis 1
        split = False
        expected_ndim = 2
    else:
        axis1_labels = [f'{data_modality}_tile{i}' for i in range(num_tiles)]
        split = True
        expected_ndim = 3
 
    model.eval()
    per_run_meta = []
    per_run_repr = []
 
    # ---- 3. Outer augmentation loop -----------------------------------------
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=auto):
            for aug in range(num_augmentations):
                run_outputs, run_targets = [], []
                run_ids, run_questionnaires = [], []
    
                for batch in dataloader:
                    inputs, targets, ids, questionnaires, _ = batch
                    inputs = inputs.to(device)
    
                    outputs = model(inputs)
    
                    run_outputs.append(outputs.float().cpu())
                    run_targets.append(targets.cpu())
                    run_ids.extend(ids)
                    run_questionnaires.extend(questionnaires)
    
                outputs_tensor = torch.cat(run_outputs, dim=0)
                targets_tensor = torch.cat(run_targets, dim=0)
                outputs_np = outputs_tensor.numpy().astype(np.float32, copy=False)
    
                # Catch config/model mismatch early.
                if outputs_np.ndim != expected_ndim:
                    raise ValueError(
                        f"Output ndim {outputs_np.ndim} != expected {expected_ndim} "
                        f"for data_modality='{data_modality}', num_tiles={num_tiles}."
                    )
                if split and outputs_np.shape[1] != len(axis1_labels):
                    raise ValueError(
                        f"Output axis-1 size {outputs_np.shape[1]} != "
                        f"{len(axis1_labels)} expected ({axis1_labels})."
                    )
    
                meta = pd.DataFrame({
                    'subject_id': run_ids,
                    'questionnaire': run_questionnaires,
                    'augmentation_version': aug,         # broadcast scalar
                    'true_label': targets_tensor.numpy().tolist(),
                })
    
                per_run_meta.append(meta)
                per_run_repr.append(outputs_np)
 
    # ---- 4. Assemble aligned outputs ----------------------------------------
    # metadata row i  <->  representations[i].  Alignment holds even with a
    # shuffling loader, because both lists are filled from the same batches
    # in lockstep within each pass.
    metadata = pd.concat(per_run_meta, ignore_index=True)
    representations = np.concatenate(per_run_repr, axis=0)
 
    # ---- 5. Optional save ----------------------------------------------------
    if output_path is not None:
        stem = Path(output_path)
        stem.parent.mkdir(parents=True, exist_ok=True)
 
        meta_file = stem.with_name(stem.name + '_metadata.parquet')
        repr_file = stem.with_name(stem.name + '_representations.npy')
        layout_file = stem.with_name(stem.name + '_layout.json')
 
        metadata.to_parquet(meta_file, index=False)
        np.save(repr_file, representations)
 
        layout = {
            'data_modality': data_modality,
            'num_tiles': num_tiles,
            'num_augmentations': num_augmentations,
            'num_rows': int(representations.shape[0]),
            'representation_shape': list(representations.shape),
            'feature_dim_N': int(representations.shape[-1]),
            'dtype': str(representations.dtype),
            'axis1_labels': axis1_labels,  # None when output is (num_rows, N)
            'alignment': 'representations[i] corresponds to metadata row i',
        }
        layout.update(custom_metadata)  # Add any extra metadata fields provided by the user
        with open(layout_file, 'w') as f:
            json.dump(layout, f, indent=2)
 
    return metadata, representations


def main():
    args = get_args()
    worker = args.num_workers
    batch_size = 32
    prefetch_factor = 2
    decode_approach = 'pil'
    load_in_memory = False
    split_workers = True
    transform = None

    if NUM_tiles > 1 and DATA_MODALITY == 'all':
        print("Warning: Data modality = 'all' and NUM_tiles>1 are incompatible ")
        return 
    if checkpoint_to_load and custom_pre_trained_weights:
        print("Warning: checkpoint_to_load and custom_pre_trained_weights are mutually exclusive. Please choose one.")
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

    # Automatically use GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu = torch.cuda.get_device_name(device_id)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
        print(f"--- GPU DIAGNOSTICS ---")
        print(f"Active GPU: {gpu}")
        print(f"CUDA_VISIBLE_DEVICES: {visible}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Reading from {SHARD_PATTERN_train} and {SHARD_PATTERN_val} with decode approach '{decode_approach}' and load_in_memory={load_in_memory}")
    
    
    
    model,transform = get_model(name=MODEL, pretrained=True, custom_pre_trained_weights=custom_pre_trained_weights)
    if NUM_tiles > 1 or DATA_MODALITY == 'all':
        model = ConcatenateViews(model)
    if transform_override:
        transform = T.Compose(
            [
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.06040578708052635, 0.06040578708052635, 0.06040578708052635], 
                            std=[0.23823712766170502, 0.23823712766170502, 0.23823712766170502]),
            ]
        )

    data_loaders,datasets = prepare_dataloaders(decode_approach=decode_approach, load_in_memory=load_in_memory, split_workers=split_workers, 
                        batch_size=batch_size, transform=transform, augmentation_transform=augmentation_transform,
                        prefetch_factor=prefetch_factor, worker=worker, 
                        exclusion_set=exclusion_set, data_modality=DATA_MODALITY, 
                        huggingface_transform=huggingface_transform,
                        n_views=NUM_tiles)
    if DEBUG_IMGS:
        # Iterate through the first few items to see what labels are coming out
        for i, (img, label, key, *_) in enumerate(datasets['train']):
            print(f"Sample {i}: Label {label}, Key {key}")
            if i > 10: break
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        n_stacked=NUM_tiles
        if DATA_MODALITY == 'all':
            n_stacked = 3
        debug_images_dataset(datasets['train'], output_path="data/anteprima_dataset.png", num_immagini=16, mean=None, std=None, n_stacked=n_stacked)
    
    #raise Exception("Debugging: Stopping after dataset preparation and image debugging. onsole output.")

    if checkpoint_to_load:
        # if you want you can set anothr kind of logger (not tensorboard but csv ..)
        ckpt_path=os.path.join(checkpoint_to_load) 
        load_backbone_from_lightning_ckpt(model, ckpt_path)
    
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    import time
    '''import time
    for batch in data_loaders['train']:
        t0 = time.perf_counter()
        inputs, *_ = batch
        t1 = time.perf_counter()
        inputs = inputs.to(device)
        t2 = time.perf_counter()
        outputs = model(inputs)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        print(f"load={t1-t0:.3f}s  transfer={t2-t1:.3f}s  model={t3-t2:.3f}s")
        break
    return'''
    # warmup
    '''device = "cuda"
    inputs = torch.randn(16, 3,3, 224, 224, device=device)
    inputs = torch.randn(16, 3, 224, 224, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(3):
            model(inputs)
    torch.cuda.synchronize()

    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs, *_ = model(inputs)
    torch.cuda.synchronize()
    print(f"{(time.time()-t0)*1000:.1f} ms")
    return'''
    splits = ['train', 'val']
    for split in splits:
        print(f"Extracting representations for {split} set...")
        dataloader = data_loaders[split]
        custom_metadata['split'] = split
        save_path = os.path.join(SAVE_FOLDER, f"{split}")
        run_inference_to_dataframe(model,dataloader, device, data_modality=DATA_MODALITY, 
                                        num_tiles=NUM_tiles, num_augmentations=NUM_augmentations, 
                                        output_path=save_path, custom_metadata=custom_metadata)
    


if __name__ == "__main__":
    main()
    