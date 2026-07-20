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
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
import pickle
import traceback
import cProfile, pstats

from src.utils.data_loading_utils import melt_df, prepare_loaders_PD, prepare_exclusion_sets_PD, merge_properties_from_full_dataset_PD, synthetic_data_override
from src.utils.data_loading_utils import prepare_handedness_dataset, prepare_handedness_dataset_all, generate_exclusion_set_val, test_handedness_dataset_all
from src.utils.image_processing import ResizeLongestSide, get_augmentation_transform, get_transforms, get_mu_std, ALL_SYNTHETIC_TRANSFORMS
from src.utils.visualization import debug_images_dataset, save_img_with_info_views, save_img_with_info, tensor_debug_info, debug_images_PD
from src.utils.visualization import SubjectViewer, launch_interactive_PD

params = {
    'selected_problem': "PD",#"PD", # "handedness"
    'pre_filter_csv': False,
    'columns_to_add': ['rempli_seulq12'],

}

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
    params['list_of_ids_paths'] = "/home/a_morelli/datasets/id_lists/PD_training_set_13_7_26.parquet"
    params['full_dataset'] = "/home/a_morelli/datasets/id_lists/final_table_with_all_info_8_7_26.csv"


VERBOSE = True


def main(params,run_random_samples_from_loader=False, 
         run_study_loader = False, show_grids=False, 
         run_compute_time=True, run_debug_from_shards=False,
         run_explore_files=False):
    
    pre_csv = pd.read_parquet(params['list_of_ids_paths'])
    pre_csv = merge_properties_from_full_dataset_PD(params,pre_csv, params['columns_to_add'], verbose=VERBOSE)
    new_path = os.path.dirname(params['list_of_ids_paths'])
    today_day = time.strftime("%d_%m_%y")
    pre_csv.to_parquet(os.path.join(new_path, f"PD_training_set_{today_day}.parquet"), index=False)
    

if __name__ == "__main__":
    main(params)
    