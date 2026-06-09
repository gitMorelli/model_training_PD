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
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.data_loading_utils import load_representations_handedness
from src.utils.model_utils import get_sklearn_model

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/feature_extraction/resnet18_extracted_features/imagenet_resnet18_text_tiles1_aug3" 
SEED=42
DATA_MODALITY = "text" # text,digit,X,all 
NUM_tiles = 1 #num tiles to concatenate in a single extraction
NUM_augmentations = 3 #number of times to process the same image with a random augmentation

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

def balance_df_data_handedness(train, val):
    #get the value counts for the true_label column
    class_counts = train['true_label'].value_counts()
    num_0 = class_counts.get(0.0, 0)  # Count of class 0 (e.g., right-handed)
    num_1 = class_counts.get(1.0, 0)  # Count of class 1 (e.g., left-handed)
    rate = num_1/num_0 if num_0 > 0 else 0
    print(f"Number of samples per class: Class 0 = {num_0}, Class 1 = {num_1}; Total = {num_0 + num_1}")

    #get the number of unique 'subject_id' for the class 1
    unique_subjects_class_1 = train[train['true_label'] == 1.0]['subject_id'].nunique()
    #get the list of unique 'subject_id' for the class 0
    unique_subjects_class_0 = train[train['true_label'] == 0.0]['subject_id'].unique()
    #select a random sample of unique 'subject_id' from the class 0 equal to the number of unique 'subject_id' in class 1
    random_subjects_class_0 = random.sample(list(unique_subjects_class_0), unique_subjects_class_1)
    #filter the train dataframe to keep only the rows with the selected 'subject_id' for class 0 and all the rows for class 1
    train = train[(train['true_label'] == 1.0) | ((train['true_label'] == 0.0) & (train['subject_id'].isin(random_subjects_class_0)))]

    #do the same for the val dataframe
    unique_subjects_class_1_val = val[val['true_label'] == 1.0]['subject_id'].nunique()
    unique_subjects_class_0_val = val[val['true_label'] == 0.0]['subject_id'].unique()
    random_subjects_class_0_val = random.sample(list(unique_subjects_class_0_val), unique_subjects_class_1_val)
    val = val[(val['true_label'] == 1.0) | ((val['true_label'] == 0.0) & (val['subject_id'].isin(random_subjects_class_0_val)))]

    print("After balancing the classes:")
    #get the value counts for the true_label column
    class_counts = train['true_label'].value_counts()
    num_0 = class_counts.get(0.0, 0)  # Count of class 0 (e.g., right-handed)
    num_1 = class_counts.get(1.0, 0)  # Count of class 1 (e.g., left-handed)
    rate = num_1/num_0 if num_0 > 0 else 0
    print(f"Number of samples per class: Class 0 = {num_0}, Class 1 = {num_1}; Total = {num_0 + num_1}")
    return train, val

def main():
    args = get_args()
    worker = args.num_workers
    batch_size = 16
    prefetch_factor = 2
    decode_approach = 'pil'
    load_in_memory = False
    split_workers = True
    transform = None
    selected_model = "logreg"
    sklearn_model_parameters = {}

    splits = ["train", "val", "test"]
    #get the filenames in the source path
    filenames = os.listdir(SOURCE_PATH)
    print(filenames)  # ['file1.txt', 'file2.py', 'folder', ...]
    dict_filenames = {}
    for split in splits:
        for filename in filenames:
            if split in filename:
                dict_filenames[split] = filename
                break
    print(dict_filenames)

    train, _ = load_representations_handedness(
        output_path=os.path.join(SOURCE_PATH, dict_filenames["train"]), 
        as_dataframe=True
    )    

    val, _ = load_representations_handedness(
        output_path=os.path.join(SOURCE_PATH, dict_filenames["val"]), 
        as_dataframe=True
    )

    print("df columns -> ", list(train.columns))
    #print(train.iloc[0].to_dict())

    train,val = balance_df_data_handedness(train, val)

    # unpack
    X_train = np.stack(train['repr_text'].values)
    y_train = train['true_label'].values

    X_val = np.stack(val['repr_text'].values)
    y_val = val['true_label'].values

    model = get_sklearn_model(name=selected_model, **sklearn_model_parameters)
    pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Normalize features
            (selected_model, model)  # Train GBM classifier
        ])

    #flush the output buffer to ensure that the print statements are printed before the training starts
    print("Flushing the output..", flush=True)
    
    pipeline.fit(X_train, y_train)
    #print(pipeline.named_steps[selected_model].n_outputs_)
    #print(pipeline.named_steps[selected_model].classes_)

    # evaluate
    y_pred = pipeline.predict(X_val)
    '''
    y_prob= pipeline.predict_proba(X_train.values)[:,1]
        #y_pred = pipeline.predict(X_train.values)
        y_pred=(y_prob>= 0.5).astype(int)
    '''
    print(classification_report(y_val, y_pred))
    
    
    

    
    


if __name__ == "__main__":
    main()
    