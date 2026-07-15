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

from src.utils.data_loading_utils import prepare_loaders_PD, prepare_exclusion_sets_PD
from src.utils.data_loading_utils import explore_data
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, TiledJoinedModels
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_dataset
from src.utils.image_processing import get_augmentation_transform, get_transforms
from src.utils.training_utils import LitModel


params = {
    'selected_problem': "PD",#"PD", # "handedness"

    "data_modality": ['digit_full','digit_crop']+['digit']+
    ['text_full']+['text']+['X_crop'], # 'X', 'text', 'digit', 'all' (all returns 3x3x224x224 elements instead of 3x224x224)
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
    "invert_color": True,
    'filter_missing': 'last_q', #'all', 'last_q' #if all remove only ids with grid_pattern=0000..00 13 times, 
    #if 'last_q' with the first last_q equal to 0
    'censor_time': -1, #0, -1 (if keep all) or a positive value
    'filter_modality': 'digit', #None, 'X', 'text', 'digit' (if None keep all modalities)

    #dataloader params
    "batch_size": 2,
    "prefetch_factor": 2,
    "decode_approach": "pil",
    "load_in_memory": False,
    "split_workers": True,

    "debug": True, 
}
if 'pil' not in params['custom_transform']:
    raise ValueError("For debug mode, transform_func must contain 'pil' ")
if not params['debug']:
    raise ValueError("This script is for debugging purposes only. Set 'debug' to True in params.")
if isinstance(params['data_modality'],list):
    params['num_tiles'] = len(params['data_modality'])
huggingface_transform=True if params['model'] in ['clip-vit-large-patch14-un', 'clip-vit-large-patch14-inter'] else False
params['huggingface_transform'] = huggingface_transform

if params['selected_problem'] == "handedness":
    SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"
    CSV_LOAD_PATH = "/home/a_morelli/vscode_projects/model_training/data/inspect_statistics/merged_statistics_w_predictions_w_original.csv"

    #LIST_OF_IDS_HANDEDNESS_PATH = os.path.join(SOURCE_PATH,"handedness_model_ids.csv")
    params['list_of_ids_paths'] = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv"

    #data_folder = "png_resized_padded_whitebg", "all_png_resized_padded", "all_png_whitebg" , "all_no_grids_png_whitebg" 
    data_folder = "all_no_grids_png_whitebg" #"all_no_grids_png_resized_half_whitebg"
    SAVE_PATH = "/home/a_morelli/vscode_projects/model_training/data/dataset_info"

    params["h5_data_path"] = "/mnt/beegfs02/scratch/a_morelli/datasets/rr_data_h5.pkl"
elif params['selected_problem'] == "PD":
    SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/PD/"
    CSV_LOAD_PATH = ""

    #LIST_OF_IDS_HANDEDNESS_PATH = os.path.join(SOURCE_PATH,"handedness_model_ids.csv")
    params['list_of_ids_paths'] = "/home/a_morelli/datasets/id_lists/PD_training_set_8_7_26.parquet"

    #data_folder = "png_resized_padded_whitebg", "all_png_resized_padded", "all_png_whitebg" , "all_no_grids_png_whitebg" 
    data_folder = "final_png_whitebg" #"all_no_grids_png_resized_half_whitebg"
    SAVE_PATH = "/home/a_morelli/vscode_projects/model_training/data/dataset_info_PD"

    params["h5_data_path"] = "/mnt/beegfs02/scratch/a_morelli/datasets/PD_data_h5.pkl"

SOURCE_PATTERN = os.path.join(SOURCE_PATH,data_folder)
SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")
VERBOSE = True
SAVE_FOLDER_PATH = "/home/a_morelli/datasets/id_lists/statistics"
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(q) for q in range(1,14)]


def main(params):
    args = get_args()
    worker = args.num_workers
    prefetch_factor = 2

    if params['selected_problem'] == "handedness":
        n_test=-1

        dataset_val = explore_data(SHARD_PATTERN_val, batch_size=None)
        dataset_train = explore_data(SHARD_PATTERN_train, batch_size=None)
        
        loader_val = DataLoader(
            dataset_val, 
            num_workers=worker, 
            batch_size=None, 
            prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
            pin_memory=True
        )
        loader_train = DataLoader(  
            dataset_train, 
            num_workers=worker, 
            batch_size=None, 
            prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
            pin_memory=True
        )

        rows = []
        i=0
        for sample in loader_val:
            rows.append(build_row(*sample))
            i+=1
            if i>=n_test and n_test>0:
                break
        for sample in loader_train:
            rows.append(build_row(*sample))
            i+=1
            if i>=n_test and n_test>0:
                break 
        df = pd.DataFrame(rows)
        with pd.option_context('display.max_columns', None, 'display.max_colwidth', None):
            print(df.head())
        #check if there are rows with same subject_id, questionnaire and modality_type
        duplicates = df[df.duplicated(subset=['subject_id', 'questionnaire', 'modality_type'], keep=False)]
        if not duplicates.empty:
            print("Found duplicates based on subject_id, questionnaire, and modality_type:")
            print(duplicates)
        else:
            print("No duplicates found based on subject_id, questionnaire, and modality_type.")
        #create a timestamp
    elif params['selected_problem'] == "PD":
        max_batches=None
        train_loader, val_loader = get_dataloader(args,params)
        print("#"*100)
        print("*"*100)
        print("#"*100)
        result_df_train = read_loader(train_loader, max_batches=max_batches)  
        '''with pd.option_context('display.max_columns', None, 'display.max_colwidth', None):
            print("Train DataFrame:")
            print(result_df_train.head())'''
        result_df_train['split'] = 'train'
        result_df_val = read_loader(val_loader, max_batches=max_batches)
        result_df_val['split'] = 'val'
        df = pd.concat([result_df_train, result_df_val], ignore_index=True)
    timestamp = time.strftime("%d%m%Y")
    os.makedirs(SAVE_FOLDER_PATH, exist_ok=True)
    save_name = f"statistics_{params['selected_problem']}_{timestamp}.csv"
    save_path = os.path.join(SAVE_FOLDER_PATH, save_name)
    #save params to a json file
    import json
    params_save_name = f"metadata_{params['selected_problem']}_{timestamp}.json"
    json_save_path = os.path.join(SAVE_FOLDER_PATH, params_save_name)
    json.dump(params, open(json_save_path, 'w'), indent=4)
    df.to_csv(save_path, index=False)
########## PD ################
def get_dataloader(args,params):
    worker = args.num_workers

    grid_dict = None
    if params['use_grid']:
        with open(params['h5_data_path'], "rb") as f:
            grid_dict = pickle.load(f)
        print("Grid dictionary loaded successfully. Number of entries:", len(grid_dict))

    transform = get_transforms(params, None)

    if params['selected_problem'] == "PD":
        #exclude controls from the training if i want to reduce the asimmetry of the dataset (for example if i want to have a 1:1 ratio between cases and controls)
        exclusion_set, val_exclusion_set, counts = prepare_exclusion_sets_PD(params,verbose=VERBOSE,class_col='diag_park_final1_quest')
        train_loader,val_loader,_,_= prepare_loaders_PD(worker,params['prefetch_factor'],params, exclusion_set, val_exclusion_set,
                                                         grid_dict, transform, SHARD_PATTERN_train, SHARD_PATTERN_val)
    else:
        raise ValueError(f"Unknown selected_problem: {params['selected_problem']}")

    return train_loader, val_loader

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
def read_loader(loader, subject_ids=None, slot_to_q=None, max_batches=None):
    list_of_rows = []
    n_batch = 0
    for batch in loader:
        # unpack, tolerating the 5- or 6-element variant
        frames, seq_ids, slot_ids, lengths, labels, resizing_factors,subject_ids, modalities = batch

        seq_ids  = seq_ids.cpu()
        slot_ids = slot_ids.cpu()
        lengths  = lengths.cpu()
        B = lengths.size(0)
        N = seq_ids.size(0)

        if slot_to_q is None: #slot_ids goes from 0 to 12 -> i have to add 1 to obtain the name of the questionnaire
            slot_name = lambda s: f"{s + 1}"
        elif callable(slot_to_q):
            slot_name = slot_to_q
        else:
            slot_name = lambda s: slot_to_q.get(s, f"{s + 1}")

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

            sid = subject_ids[b] if subject_ids is not None else f"subject_{b}" # get the subject id

            smodalities = [modalities[s] for s in sel.tolist()]
            smeta = [frames[s] for s in sel.tolist()]
            row = create_row(qs,sid, smodalities, smeta)
            list_of_rows.append(row)
        
        n_batch += 1
        if max_batches is not None and n_batch >= max_batches:
            break
    df = pd.DataFrame(list_of_rows)
    return df
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