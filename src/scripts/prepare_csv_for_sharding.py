from datetime import datetime
import tarfile
import webdataset as wds
import glob
import os
import json
import pandas as pd
import torchvision.transforms as T
from PIL import Image
import io
import math
import multiprocessing as mp
import concurrent.futures
import numpy as np
import cv2
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Iterable

from src.utils.image_processing import convert_background_to_white, get_tiles, recolor_border_via_profiles
from src.utils.file_utils import get_id_data_from_h5_file 
from src.utils.data_loading_utils import prepare_pre_training

params={
    'LIST_OF_IDS_PD_PATH': "/home/a_morelli/datasets/id_lists/PD_training_set_20_07_26.parquet",
    'LIST_OF_IDS_PD_PRE_MATCHING_PATH': "/home/a_morelli/datasets/id_lists/final_table_for_matching_splitted_13_7_26.csv",
    'seed': 42,
}
save_path = "/home/a_morelli/datasets/shards"

def main():
    random.seed(params['seed'])

    lists_of_ids = {}
    problem_classes = ['for_PDpretraining','for_PD','for_PD_grouped']
    for problem_class in problem_classes:
        lists_of_ids[problem_class] = {}

    data = pd.read_parquet(params['LIST_OF_IDS_PD_PATH'])
    data_pre = pd.read_csv(params['LIST_OF_IDS_PD_PRE_MATCHING_PATH']) #has all the ids
    data_pre = prepare_pre_training(data_pre, data) #remove all the ids selected for training and prepare grid_pattern and avail_pattern

    for train_split in ['train','val','test']:
        split_data = data[data['split'] == train_split].copy()
        split_data['group_id'] = split_data['unique_id'].apply(lambda x: x.split('_')[1]) #extract the group id from the unique_id 

        lists_of_ids['for_PD_grouped'][train_split] = split_data['group_id'].astype(str).tolist()[:] #unique_id is in the form XXXXX_YY with YY the matching group,

        id_list = split_data['unique_id'].astype(str).tolist()[:] #unique_id is in the form XXXXX_YY with YY the matching group,
        random.shuffle(id_list)
        lists_of_ids['for_PD'][train_split] = id_list


        split_data_pre = data_pre[data_pre['split'] == train_split].copy()
        id_list_pre = split_data_pre['unique_id'].astype(str).tolist()[:] #unique_id is in the form XXXXX_YY with YY the matching group, 
        random.shuffle(id_list_pre)
        lists_of_ids['for_PDpretraining'][train_split] = id_list_pre
    
    # Save the lists of IDs in a single file
    # provenance metadata
    lists_of_ids['params'] = params
    
    os.makedirs(save_path, exist_ok=True)
    #get the current date
    current_date = pd.Timestamp.now().strftime("%d_%m_%y")
    with open(os.path.join(save_path, f"pre_computed_lists_{current_date}.json"), "w") as f:
        json.dump(lists_of_ids, f)
    
if __name__ == "__main__":
    main()