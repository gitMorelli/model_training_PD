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
# 4. Compute and display metrics using scikit-learn
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np
import json
import gc

from src.utils.data_loading_utils import prepare_loaders_PD, prepare_exclusion_sets_PD
from src.utils.data_loading_utils import explore_data, return_file_paths, load_grid_dict, prepare_test_exclusion_set
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, TiledJoinedModels
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_dataset
from src.utils.image_processing import get_augmentation_transform, get_transforms
from src.utils.training_utils import LitModel, set_automatic_hyperparameters
from src.scripts.train_PD_model import get_input_modality, model_initialization, logging_initialization


params = {
    'selected_problem': "PD",#"PD", # "handedness"
    'pre_training': False, #True if you want to use the pre-trained model on E3N dataset, False if you want to train from scratch

    "data_modality": get_input_modality('window_view_minimal'), 
    "num_tiles": 3,

    'model': 'clip-vit-large-patch14-inter',#'resnet18',#'clip-vit-large-patch14-inter', 
    'custom_pre_trained_weights':  None,
    'pretrained': True, #True, False, e.g. for resnet if True loads the imagenet weights for the backbone, if False loads the backbone with random weights
    'norm_mu': 'PD_window',
    "input_size": 224,
    'mean_and_std': 'handedness',
    'custom_transform': None,#set to none for using the default transform for the model
    "apply_augmentation": None,#'random_crop_half', #None, 
    "invert_color": True, #even if set to true it is ignored when debug == True
    "use_grid": True,
    "to_grayscale": True,

    "seed": 42, 
    "balanced_data": False,
    'balance_validation': False, #if True the validation set is balanced, if False it is not balanced
    "balancing_factor": 1,
    "majority_class_id": 0,
    "threshold_num": 1,
    'filter_missing': 'last_q', #'all', 'last_q' #if all remove only ids with grid_pattern=0000..00 13 times, 
    #if 'last_q' with the first last_q equal to 0
    'censor_time': 'all', 
    'filter_modality': 'digit', #None, 'X', 'text', 'digit' (if None keep all modalities)
    'grouped': False, #if true i have all elements from the same case-control group in the batch and train to distinguish the case from the controls
    'bce_aux_weight': 0.3, #weight for the BCE loss on the auxiliary output (the one that predicts the case-control group)
    'synthetic': None, #['original','progressive_thickening','progressive_slant','progressive_size_drift', 
    #'progressive_baseline_wave', 'progressive_tremor', 'progressive_ink_density'], #or None
    'synthetic_proportions': [0.5, 0.2, 0.2, 0.1], #if synthetic is not None, the proportions of each synthetic class in the training set (must sum to 1)


    #dataloader params
    "batch_size": 16,
    "prefetch_factor": 4,
    "decode_approach": "pil",
    "load_in_memory": False,
    "split_workers": True,

    "debug": False,
    "feature_extraction": False, #this True forces the debug mode true
    "add_to_existing": None, #None if you want to create a new feature extraction table, path if you want to append (it will add columns)
}

params['list_of_ids_paths'], params['data_folder'], params['grid_dict_path'] = return_file_paths(params['selected_problem'], 
                                                                                                 params['grouped'], params['pre_training'])
params = set_automatic_hyperparameters(params)

SHARD_PATTERN_train = os.path.join(params['data_folder'],"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(params['data_folder'],"val/worker*_shard-*.tar")
SHARD_PATTERN_test = os.path.join(params['data_folder'],"test/worker*_shard-*.tar")
VERBOSE = True
SAVE_FOLDER_PATH = "/home/a_morelli/models/model_training_logs/PD/representation_extraction"
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(q) for q in range(1,14)]
CLASS_COL='diag_park_final1_quest'


def main(params):
    args = get_args()

    max_batches=None #None for running all

    model,transform = get_model(name=params['model'], pretrained=params['pretrained'], 
                                   custom_pre_trained_weights=params['custom_pre_trained_weights'],grayscale=params['to_grayscale'])
    
    grid_dict = load_grid_dict(params)
    transform = get_transforms(params, transform)
    exclusion_set, val_exclusion_set, _ = prepare_exclusion_sets_PD(
        params, verbose=VERBOSE, class_col=CLASS_COL)
    test_exclusion_set, _ = prepare_test_exclusion_set(
        params, verbose=VERBOSE, class_col=CLASS_COL)
    train_df = pd.read_parquet(params['list_of_ids_paths'])

    common = dict(worker=args.num_workers,
                  prefetch_factor=params['prefetch_factor'],
                  exp_params=params, grid_dict=grid_dict,
                  transform=transform, train_df=train_df, persistent_workers=False, one_only=True)

    specs = {
        'train': lambda: prepare_loaders_PD(
            exclusion_set=exclusion_set, val_exclusion_set=val_exclusion_set,
            SHARD_PATTERN_train=SHARD_PATTERN_train,
            SHARD_PATTERN_val=SHARD_PATTERN_val, **common)[0],
        'val':   lambda: prepare_loaders_PD(
            exclusion_set=val_exclusion_set, val_exclusion_set=val_exclusion_set,
            SHARD_PATTERN_train=SHARD_PATTERN_val,
            SHARD_PATTERN_val=SHARD_PATTERN_val, **common)[0],
        'test':  lambda: prepare_loaders_PD(
            exclusion_set=test_exclusion_set, val_exclusion_set=test_exclusion_set,
            SHARD_PATTERN_train=SHARD_PATTERN_test,
            SHARD_PATTERN_val=SHARD_PATTERN_test, **common)[0],
    }

    frames = [_read_and_release(fn, model, create_row, max_batches, split)
              for split, fn in specs.items()]
    df = pd.concat(frames, ignore_index=True)

    print(df["rep"].iloc[0].shape, df["rep"].iloc[0].dtype)
    print(df["rep"].map(lambda x: getattr(x, "shape", None)).value_counts())

    save_results(params, df)
########## PD ################
def _read_and_release(make_loader, model,create_row, max_batches, split):
    """Build a loader, read it, then fully tear down its workers."""
    loader = make_loader()
    try:
        df = extract_loader(model,loader, create_row, max_batches=max_batches)
        df['split'] = split
        return df
    finally:
        # the iterator owns the worker processes; drop it first
        it = getattr(loader, "_iterator", None)
        if it is not None:
            loader._iterator = None
            del it
        del loader
        gc.collect()

def create_row(qs, sid, smodalities, srepresentations):
    rows = []
    for i, q in enumerate(qs):
        for modality, rep in zip(smodalities[i], srepresentations):
            row = {"subject_id": sid}
            row["q"] = q
            row["modality"] = modality
            row["rep"] = rep.numpy()  # Convert tensor to 1dim numpy array
            #print(f"Subject {sid}: representations shape: {rep.shape}")
            rows.append(row)
    return rows

def extract_loader(model, loader, create_row, slot_to_q=None, max_batches=None,
                   device=None, amp=True, amp_dtype=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # autocast only makes sense on cuda here; silently disable it elsewhere
    use_amp = bool(amp) and device.type == "cuda"
    if amp and not use_amp:
        print("autocast requested but device is not cuda -> running in full precision")
    if amp_dtype is None:
        # bf16 if the card supports it (Ampere+), otherwise fp16
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"Using device: {device}, autocast: {use_amp}" + (f" ({amp_dtype})" if use_amp else ""))

    model.to(device)
    model.eval()

    list_of_rows = []
    n_batch = 0
    counters = [0 for _ in range(13)]
    if slot_to_q is None:
        slot_name = lambda s: f"{s + 1}"
    elif callable(slot_to_q):
        slot_name = slot_to_q
    else:
        slot_name = lambda s: slot_to_q.get(s, f"{s + 1}")

    start_time = time.time()
    with torch.inference_mode():
        for batch in loader:
            frames, seq_ids, slot_ids, lengths, labels, resizing_factors, subject_ids, modalities = batch
            seq_ids  = seq_ids.cpu()
            slot_ids = slot_ids.cpu()
            lengths  = lengths.cpu()
            B = lengths.size(0)
            N = seq_ids.size(0)
            _, k = frames.shape[:2]

            frames = frames.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                representations = model(frames.flatten(0, 1))
            # .float() is a no-op when autocast is off and already fp32
            representations = representations.detach().float().cpu()

            counts = torch.bincount(seq_ids, minlength=B)
            mismatch = (counts != lengths).nonzero(as_tuple=True)[0].tolist()
            if mismatch:
                raise ValueError(f"Length mismatch for subjects {mismatch}: "
                                 f"lengths={lengths} vs counts={counts.tolist()}")

            for b in range(B):
                sel = (seq_ids == b).nonzero(as_tuple=True)[0]
                sel = sel[torch.argsort(slot_ids[sel])]
                slots = slot_ids[sel].tolist()
                qs = [slot_name(s) for s in slots]
                for q in qs:
                    q_index = int(q) - 1
                    if q_index < 0 or q_index >= len(counters):
                        raise ValueError(f"Questionnaire index {q_index} out of range for counters list.")
                    counters[q_index] += 1
                sid = subject_ids[b] if subject_ids is not None else f"subject_{b}"
                smodalities = [modalities[s] for s in sel.tolist()]
                srepresentations = representations[sel]
                row = create_row(qs, sid, smodalities, srepresentations)
                if isinstance(row, list):
                    list_of_rows.extend(row)
                else:
                    list_of_rows.append(row)

            if n_batch % 10 == 0:
                print(f"Processed {n_batch} batches, total rows collected: {len(list_of_rows)}")
                print("#" * 50)

            n_batch += 1
            if max_batches is not None and n_batch >= max_batches:
                break

    elapsed_time = time.time() - start_time
    print(f"Processed {n_batch} batches in {elapsed_time:.2f} seconds, batches per second: {n_batch/elapsed_time:.2f}")
    print(f"Final counts per questionnaire: {counters}", flush=True)
    print("#" * 50)
    return pd.DataFrame(list_of_rows)

def save_results(params, df):
    if params['add_to_existing'] is not None:
        #append to existing file
        df_existing = pd.read_csv(params['add_to_existing'])
        df = pd.concat([df_existing, df], ignore_index=True)
    else:
        timestamp = time.strftime("%d%m%Y_%H%M%S")
        save_folder = os.path.join(SAVE_FOLDER_PATH,params['model'], timestamp)
        os.makedirs(save_folder, exist_ok=True)

        #save as parquet file
        save_name = f"representations_{params['selected_problem']}.parquet"
        save_path = os.path.join(save_folder, save_name)
        df = df.reset_index(drop=True)
        df["rep_idx"] = np.arange(len(df))
        df.to_parquet(save_path, index=False)
        #embs = np.stack(df["rep"].to_numpy())   # (N, 1024)
        #df.drop(columns=["rep"]).to_parquet(save_path, index=False)

        #save the representations in a npy file
        '''npy_save_path = os.path.join(save_folder, "repr.npy")
        np.save(npy_save_path, embs)'''

        #save params to a json file
        params_save_name = f"metadata_{params['selected_problem']}.json"
        json_save_path = os.path.join(save_folder, params_save_name)
        json.dump(params, open(json_save_path, 'w'), indent=4)

        '''
        loading
        df = pd.read_csv("meta.csv")
        embs = np.load("repr.npy")
        df["repr"] = list(embs[df["repr_idx"].to_numpy()])
        '''

    return

########## HANDEDNESS ###########
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()


def melt_df(df,modality,threshold=1):
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
    df_filtered = df_long[df_long['score'] < threshold]
    ident_projets_to_exclude = set(df_filtered['ident_projet'].unique())
    df_long = df_long[df_long['score'] >= threshold]

    print(f"Length of melted df after filtering: {len(df_long)}")

    # 4. Drop the temporary columns to get your final desired structure
    new_df = df_long[['ident_projet', 'lateralite','split']].reset_index(drop=True)

    return new_df,ident_projets_to_exclude

def _scalar(x):
    """Convert torch tensors / numpy scalars to plain Python values."""
    if hasattr(x, "item"):      # torch tensor or numpy scalar
        try:
            return x.item()
        except (ValueError, RuntimeError):
            return x            # non-scalar tensor — leave as-is
    return x


def build_row(subject_id, questionnaire, modality_type, label,
              shard_name, img_properties):
    props = dict(img_properties)          # copy so we don't mutate the source
    width, height = props.pop("size")     # split (width, height) into two cols

    return {
        "subject_id":    _scalar(subject_id),
        "questionnaire": _scalar(questionnaire),
        "modality_type": _scalar(modality_type),
        "label":         _scalar(label),
        "shard_name":    shard_name,
        "width":         _scalar(width),
        "height":        _scalar(height),
        "format":        props.pop("format"),
        "num_channels_original": props.pop("num_channels_original"),
        "mode":          props.pop("mode"),
        "ratio":         _scalar(props.pop("ratio")),
        "mean_intensity":     _scalar(props.pop("mean_intensity")),
        "std_intensity":      _scalar(props.pop("std_intensity")),
        "ink_density_binary": _scalar(props.pop("ink_density_binary")),
        **props,   # catch any extra keys you add later, automatically
    }

if __name__ == "__main__":
    main(params)