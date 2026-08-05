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

from src.utils.data_loading_utils import prepare_loaders_PD, prepare_exclusion_sets_PD
from src.utils.data_loading_utils import explore_data, return_file_paths, load_grid_dict, prepare_test_exclusion_set
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, TiledJoinedModels
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_dataset
from src.utils.image_processing import get_augmentation_transform, get_transforms
from src.utils.training_utils import LitModel, set_automatic_hyperparameters
from src.scripts.train_PD_model import get_input_modality


params = {
    'selected_problem': "PD",#"PD", # "handedness"

    "data_modality": get_input_modality('window_view'), 
    "num_tiles": 3,

    'model': 'resnet18', 
    "input_size": 224,
    'mean_and_std': 'handedness',
    'custom_transform': 'pad_resize_pil',#remember to use the pil versions for debugging 'pad_resize_normalize'->'pad_resize_pil', 
    "apply_augmentation": 'random_crop_half',#'random_crop_half', #None, 
    "invert_color": True, #even if set to true it is ignored when debug == True
    "use_grid": True,

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
    "batch_size": 2,
    "prefetch_factor": 4,
    "decode_approach": "pil",
    "load_in_memory": False,
    "split_workers": True,

    "debug": False,
    "feature_extraction": True, #this True forces the debug mode true
    "add_to_existing": None, #None if you want to create a new feature extraction table, path if you want to append
}

params['list_of_ids_paths'], params['data_folder'], params['grid_dict_path'] = return_file_paths(params['problem'], params['grouped'], params['pre_training'])
params = set_automatic_hyperparameters(params)

if 'pil' not in params['custom_transform']:
    raise ValueError("For debug mode, transform_func must contain 'pil' ")
if not params['feature_extraction']:
    raise ValueError("This script is for debugging purposes only. Set 'feature_extraction' to True in params.")
if not (params['debug'] and params['feature_extraction']):
    raise ValueError("Set both 'debug' and 'feature_extraction' to True in params.")



SHARD_PATTERN_train = os.path.join(params['data_folder'],"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(params['data_folder'],"val/worker*_shard-*.tar")
SHARD_PATTERN_test = os.path.join(params['data_folder'],"test/worker*_shard-*.tar")
VERBOSE = True
SAVE_FOLDER_PATH = "/home/a_morelli/models/model_training_logs/PD/feature_extraction"
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(q) for q in range(1,14)]
CLASS_COL='diag_park_final1_quest'


def main(params):
    args = get_args()

    max_batches=300
    train_loader, val_loader, test_loader = get_dataloader(args,params)

    result_df_train = read_loader(train_loader, create_row, max_batches=max_batches) 
    result_df_train['split'] = 'train'
    result_df_val = read_loader(val_loader, create_row, max_batches=max_batches)
    result_df_val['split'] = 'val'
    result_df_test = read_loader(test_loader, create_row, max_batches=max_batches)
    result_df_test['split'] = 'test'

    df = pd.concat([result_df_train, result_df_val, result_df_test], ignore_index=True)

    save_results(params)
########## PD ################
def get_dataloader(args,params):
    worker = args.num_workers

    grid_dict = load_grid_dict(params)

    transform = get_transforms(params, None)

    #exclude controls from the training if i want to reduce the asimmetry of the dataset (for example if i want to have a 1:1 ratio between cases and controls)
    exclusion_set, val_exclusion_set, counts = prepare_exclusion_sets_PD(params,verbose=VERBOSE,class_col=CLASS_COL)
    test_exclusion_set, test_counts = prepare_test_exclusion_set(params,verbose=VERBOSE,class_col=CLASS_COL)

    train_df = pd.read_parquet(params['list_of_ids_paths'])

    train_loader,val_loader,_,_= prepare_loaders_PD(worker,params['prefetch_factor'],params, exclusion_set, val_exclusion_set,
                                                        grid_dict, transform, SHARD_PATTERN_train, SHARD_PATTERN_val, train_df=train_df)
    test_loader,_,_,_= prepare_loaders_PD(worker,params['prefetch_factor'],params, test_exclusion_set, test_exclusion_set,
                                                        grid_dict, transform, SHARD_PATTERN_test, SHARD_PATTERN_test, train_df=train_df)

    return train_loader, val_loader, test_loader

def create_row(qs,sid, smodalities, smeta):
    row = {
        "subject_id": sid,
    }
    def map_modality(modality):
        if 'digit' in modality:
            return 'digit'
        elif 'text' in modality:
            return 'text'
        elif 'X' in modality:
            return 'X'
        else:
            return modality 
    for i, q in enumerate(qs):
        for modality, meta in zip(smodalities[i], smeta[i]):
            if 'original' in modality:
                row[f'q_{q}_width_{modality}'] = meta['width']
                row[f'q_{q}_height_{modality}'] = meta['height']
            row[f'q_{q}_InkDensity_{modality}'] = meta['ink_density'] #don't use underscores for the property name
            #row[f'q_{q}_sharpness_{map_modality(modality)}'] = meta['sharpness']
            #row[f'q_{q}_imputed_{modality}'] = meta['imputed'] #never imputed for any questionnaire -> irrelevant
    return row
def read_loader(loader, create_row, slot_to_q=None, max_batches=None):
    list_of_rows = []
    n_batch = 0
    counters = [0 for _ in range(13)] #initialize a counter for each questionnaire (0-12)

    if slot_to_q is None: #slot_ids goes from 0 to 12 -> i have to add 1 to obtain the name of the questionnaire
        slot_name = lambda s: f"{s + 1}"
    elif callable(slot_to_q):
        slot_name = slot_to_q
    else:
        slot_name = lambda s: slot_to_q.get(s, f"{s + 1}")

    for batch in loader:
        # unpack, tolerating the 5- or 6-element variant
        frames, seq_ids, slot_ids, lengths, labels, resizing_factors,subject_ids, modalities = batch

        seq_ids  = seq_ids.cpu()
        slot_ids = slot_ids.cpu()
        lengths  = lengths.cpu()
        B = lengths.size(0)
        N = seq_ids.size(0)

        # consistency check: lengths must match the frame counts implied by seq_ids
        counts = torch.bincount(seq_ids, minlength=B)
        mismatch = (counts != lengths).nonzero(as_tuple=True)[0].tolist()
        
        if mismatch:
            raise ValueError(f"Length mismatch for subjects {mismatch}: " 
                             f"lengths={lengths} vs counts={counts.tolist()}")

        # per-subject breakdown
        for b in range(B):
            sel = (seq_ids == b).nonzero(as_tuple=True)[0]
            sel = sel[torch.argsort(slot_ids[sel])]            # slot order
            slots = slot_ids[sel].tolist()

            qs    = [slot_name(s) for s in slots] #get the questionnaires for this subject
            #add 1 to the corresponding counter for each questionnaire
            for q in qs:
                q_index = int(q) - 1 #convert to int and subtract 1 to get the index
                counters[q_index] += 1
                if q_index < 0 or q_index >= len(counters):
                    raise ValueError(f"Questionnaire index {q_index} out of range for counters list.")

            sid = subject_ids[b] if subject_ids is not None else f"subject_{b}" # get the subject id

            smodalities = [modalities[s] for s in sel.tolist()]
            smeta = [frames[s] for s in sel.tolist()]
            row = create_row(qs,sid, smodalities, smeta)
            #row can also be a list of rows if i want to have one row per modality instead of one row per subject
            if isinstance(row, list):
                list_of_rows.extend(row)
            else:
                list_of_rows.append(row)
        if n_batch % 10 == 0:
            print(f"Processed {n_batch} batches, total rows collected: {len(list_of_rows)}")
            print(f"Current counts per questionnaire: {counters}", flush=True)
            print("#"*50)
        
        n_batch += 1
        if max_batches is not None and n_batch >= max_batches:
            break
    df = pd.DataFrame(list_of_rows)
    return df

def save_results(params):
    if params['add_to_existing'] is not None:
        #append to existing file
        df_existing = pd.read_csv(params['add_to_existing'])
        df = pd.concat([df_existing, df], ignore_index=True)
    else:
        timestamp = time.strftime("%d%m%Y")
        os.makedirs(SAVE_FOLDER_PATH, exist_ok=True)
        save_name = f"statistics_{params['selected_problem']}_{timestamp}.csv"
        save_path = os.path.join(SAVE_FOLDER_PATH, save_name)
        #save params to a json file
        params_save_name = f"metadata_{params['selected_problem']}_{timestamp}.json"
        json_save_path = os.path.join(SAVE_FOLDER_PATH, params_save_name)
        json.dump(params, open(json_save_path, 'w'), indent=4)
        df.to_csv(save_path, index=False)

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