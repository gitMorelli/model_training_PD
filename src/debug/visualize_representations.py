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
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
import umap
from matplotlib import pyplot as plt

from src.utils.data_loading_utils import load_representations_handedness
from src.utils.model_utils import get_sklearn_model

representation_name = "unique_resnet18_digit_tiles1_aug2"
SOURCE_PATH = os.path.join("/mnt/beegfs02/scratch/a_morelli/model_training/handedness/feature_extraction/",
"resnet18_extracted_features/", representation_name)
STATISTICS_LOAD_PATH = os.path.join("/mnt/beegfs02/scratch/a_morelli/model_training/handedness/",
                                    "resnet18_model_results/checkpoints/v_30/merged_statistics_w_predictions_w_original.csv")
SEED=42
DATA_MODALITY = "digit" # text,digit,X,all 
NUM_tiles = 1 #num tiles to concatenate in a single extraction
NUM_augmentations = 1 #number of augmentations to consider
SAVE_PATH = "/home/a_morelli/vscode_projects/model_training/data/representations"
MODEL_SPECIFIC_SAVE_PATH = os.path.dirname(STATISTICS_LOAD_PATH)
use_PCA = True
pca_size = 50

metadata = {
    "source_path": SOURCE_PATH,
    "statistics_load_path": STATISTICS_LOAD_PATH,
    "data_modality": DATA_MODALITY,
    "num_tiles": NUM_tiles,
    "num_augmentations": NUM_augmentations,
    "seed": SEED,
    "use_PCA": use_PCA,
    "pca_size": pca_size,
}

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

def prepare_data():
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

    return train, val, dict_filenames

def generate_umap_plot(train, val, label_col, repr_column):
    title = f"umap_{label_col}_{repr_column}.png"
    print(f"Generating UMAP plot for {label_col} vs {repr_column}...")
    X = np.stack(train[repr_column].values).astype(np.float32)
    assert X.ndim == 2, f"expected 2D, got {X.shape} — column probably not parsed"
    y = train[label_col].values
    assert np.isfinite(X).all(), "NaNs/infs present"

    # choice A: standardize + euclidean
    X = StandardScaler().fit_transform(X)
    if X.shape[1] > pca_size and use_PCA:
        X = PCA(n_components=pca_size, random_state=0).fit_transform(X)

    reducer = umap.UMAP(
        n_neighbors=15,      # ↑ = more global structure, ↓ = more local detail
        min_dist=0.1,        # ↓ = tighter clumps, ↑ = more spread
        metric="euclidean",  # use "cosine" if you skipped standardization
        n_components=2,
        random_state=42,
    )
    emb2d = reducer.fit_transform(X)

    plt.figure(figsize=(8, 7))
    for lab in np.unique(y):
        m = y == lab
        plt.scatter(emb2d[m, 0], emb2d[m, 1], s=6, label=str(lab), alpha=0.7)
    plt.legend(markerscale=2, fontsize=8); plt.tight_layout(); 
#save the figure
    save_dir = os.path.join(SAVE_PATH,representation_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, title), dpi=300)
    plt.close()

    print("Saving UMAP embeddings as numpy arrays...")
    #save the embeddings as a numpy array
    np.save(os.path.join(save_dir, f"umap_{label_col}_{repr_column}_embeddings.npy"), emb2d)

    #save the associated csv ignoring the representation columns
    cols_to_drop = [col for col in train.columns if col.startswith("repr_")]    
    train=train.drop(columns=cols_to_drop)
    train.to_csv(os.path.join(save_dir, f"umap_{label_col}_{repr_column}_metadata.csv"), index=False)
    
    
    return emb2d

def clustering_analysis(dict_filenames,label_column='true_label',repr_column= "repr_text"):

    #load the saved umap embeddings
    emb2d = np.load(os.path.join(SAVE_PATH,representation_name, f"umap_{label_column}_{repr_column}_embeddings.npy"))
    #load the saved data
    metadata = pd.read_csv(os.path.join(SAVE_PATH,representation_name, f"umap_{label_column}_{repr_column}_metadata.csv"))
    
    # --- cluster in the RELIABLE space (high-dim / PCA), not the 2D coords ---
    clusterer = HDBSCAN(
        min_cluster_size=50,    # smallest group you'd accept as a "cluster" — main knob
        min_samples=10,         # ↑ = more conservative, more points called noise
        metric="euclidean",     # match what you used for UMAP (cosine→use a cosine-compatible setup)
    )

    #add the repr_column to the metadata dataframe, merge on the metadata when all the columns in metadata ['subject_id','true_label' ...] are the same
    shared = metadata.columns.tolist()                     # the n matching columns
    train, _ = load_representations_handedness(
        output_path=os.path.join(SOURCE_PATH, dict_filenames["train"]), 
        as_dataframe=True
    )  
    print("Length of train: ", len(train))
    print("Length of metadata: ", len(metadata))
    train = metadata.merge(train, on=shared, how="left")
    print("Length of train after merge with metadata: ", len(train))

    X = np.stack(train[repr_column].values).astype(np.float32)  
    X = StandardScaler().fit_transform(X)
    if X.shape[1] > pca_size and use_PCA:
        X = PCA(n_components=pca_size, random_state=0).fit_transform(X)

    cluster_labels = clusterer.fit_predict(X)   # X = the high-dim/PCA matrix, NOT emb2d

    # attach labels back to the dataframe so you can pull IDs
    train = train.copy()
    train["cluster"] = cluster_labels

    print(train["cluster"].value_counts().sort_index())  # -1 is noise

    # all ids in cluster 3
    #ids = train.loc[train["cluster"] == 3, id_col].tolist()

    # ids per cluster, excluding noise, as a dict
    '''clusters = {
        c: train.loc[train["cluster"] == c, id_col].tolist()
        for c in sorted(train["cluster"].unique()) if c != -1
    }'''

    plt.figure(figsize=(8, 7))
    labs = train["cluster"].to_numpy()
    for c in sorted(set(labs)):
        m = labs == c
        color = "lightgray" if c == -1 else None   # noise in gray
        plt.scatter(emb2d[m, 0], emb2d[m, 1], s=6, alpha=0.7,
                    label=("noise" if c == -1 else f"cluster {c}"), c=color)
    plt.legend(markerscale=2, fontsize=8); plt.tight_layout(); 
    # save plot
    save_dir = os.path.join(SAVE_PATH,representation_name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"umap_{label_column}_{repr_column}_clusters.png"), dpi=300)

    #add the cluster label to the metadata dataframe and save it as a csv
    metadata["cluster"] = cluster_labels
    metadata.to_csv(os.path.join(save_dir, f"umap_{label_column}_{repr_column}_metadata_with_clusters.csv"), index=False)

def get_ids_from_clusters(label_col, repr_column):
    metadata_w_cluster_label = pd.read_csv(os.path.join(SAVE_PATH,representation_name, f"umap_{label_col}_{repr_column}_metadata_with_clusters.csv"))
    #get the unique values of the cluster label
    unique_clusters = metadata_w_cluster_label['cluster'].unique()
    print(f"Unique clusters in metadata: {unique_clusters}")

    #print numerosity of each cluster
    cluster_counts = metadata_w_cluster_label['cluster'].value_counts()
    print(f"Cluster counts: {cluster_counts}")
    
    filtered_data = metadata_w_cluster_label[metadata_w_cluster_label['cluster'] == 1]  
    print(filtered_data.head(10))
    return filtered_data


def main(clean_folder=True, generate_umaps=True, run_clustering_analysis=True,analyze_umaps=False, run_get_ids_from_clusters=False,
         copy_to_model_folder=True):
    args = get_args()
    modalities_to_analyze = ["repr_digit"]
    labels_to_consider = ["true_label"]

    if clean_folder:
        #clean the save folder for the representation_name
        save_dir = os.path.join(SAVE_PATH,representation_name)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            print(f"Cleaned the save folder: {save_dir}")
        else:
            print(f"Save folder does not exist: {save_dir}")

    statistics = pd.read_csv(STATISTICS_LOAD_PATH)
    train, val, dict_filenames = prepare_data()

    #show the unique_values of the 'augmentation_version' column in the train dataframe
    unique_augmentations = train['augmentation_version'].unique()
    print(f"Unique augmentations in train dataframe: {unique_augmentations}")
    augmentations_to_consider=[i for i in range(NUM_augmentations)]
    train = train[train['augmentation_version'].isin(augmentations_to_consider)] #keep only one version of the data

    if generate_umaps:
        for modality in modalities_to_analyze:
            for label in labels_to_consider:
                print(f"Generating UMAP plot for label: {label} and modality: {modality}")
                generate_umap_plot(train, val, label, modality)
        #visualize other plots changing the label_col 
     
    if run_clustering_analysis:
        for modality in modalities_to_analyze:
            for label in labels_to_consider:
                print(f"Clustering analysis for label: {label} and modality: {modality}")
                clustering_analysis(dict_filenames, label_column=label, repr_column=modality)
    
    if run_get_ids_from_clusters:
        get_ids_from_clusters("true_label", "repr_text")


    if analyze_umaps:
        #load the saved umap embeddings and analyze them with sklearn classification report
        embeddings_digit = np.load(os.path.join(SAVE_PATH,representation_name, f"umap_true_label_repr_digit_embeddings.npy"))
        embeddings_text = np.load(os.path.join(SAVE_PATH,representation_name, f"umap_true_label_repr_text_embeddings.npy"))
        embeddings_X = np.load(os.path.join(SAVE_PATH,representation_name, f"umap_true_label_repr_X_embeddings.npy"))
        #identify clusters in the umap embeddings using sklearn KMeans and generate a classification report
        classifier = get_sklearn_model("KMeans", n_clusters=2, random_state=42)
        
    
    if copy_to_model_folder:
        #copy the generated umap plots and embeddings to the model specific folder
        save_dir = os.path.join(SAVE_PATH,representation_name)
        model_specific_save_dir = MODEL_SPECIFIC_SAVE_PATH
        os.makedirs(model_specific_save_dir, exist_ok=True)
        for file in os.listdir(save_dir):
            full_file_name = os.path.join(save_dir, file)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, model_specific_save_dir)
                print(f"Copied {full_file_name} to {model_specific_save_dir}")
        #save metadata to json
        metadata_save_path = os.path.join(model_specific_save_dir, "metadata_visualization_clustering.json")
        with open(metadata_save_path, 'w') as f:
            json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    main()

#dfs description for reference
#representation df: 'subject_id', 'questionnaire', 'augmentation_version', 'true_label', 'repr_text', 'repr_digit', 'repr_X'
