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

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset
from src.utils.data_loading_utils import explore_data
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, TiledJoinedModels
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_dataset
from src.utils.image_processing import ResizeLongestSide
from src.utils.training_utils import LitModel

#PATHS
SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"
LIST_OF_IDS_HANDEDNESS_PATH = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv"
data_folder = "all_no_grids_png_whitebg" 
#data_folder = "all_full_sentences_png_whitebg" 
SOURCE_PATTERN = os.path.join(SOURCE_PATH,data_folder)
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")
SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SAVE_FOLDER_PATH = "/home/a_morelli/datasets/handedness/sharded_data_statistics"

#QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(q) for q in range(1,14)]


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

def main():
    args = get_args()
    worker = args.num_workers
    prefetch_factor = 2
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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_name = f"statistics_{data_folder}_{timestamp}.csv"
    save_path = os.path.join(SAVE_FOLDER_PATH, save_name)
    df.to_csv(save_path, index=False)

    

if __name__ == "__main__":
    main()