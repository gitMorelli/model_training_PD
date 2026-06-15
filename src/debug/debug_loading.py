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

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset, melt_df
from src.utils.data_loading_utils import prepare_handedness_dataset, prepare_handedness_dataset_all, generate_exclusion_set_val, test_handedness_dataset_all
from src.utils.image_processing import ResizeLongestSide
from src.utils.visualization import debug_images_dataset
SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"

#LIST_OF_IDS_HANDEDNESS_PATH = os.path.join(SOURCE_PATH,"handedness_model_ids.csv")
LIST_OF_IDS_HANDEDNESS_PATH = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv"

#data_folder = "png_resized_padded_whitebg", "all_png_resized_padded", "all_png_whitebg" , "all_no_grids_png_whitebg" 
data_folder = "all_no_grids_png_resized_half_whitebg"
SOURCE_PATTERN = os.path.join(SOURCE_PATH,data_folder)

SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")

#QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(q) for q in range(1,14)]

SAVE_PATH = "/home/a_morelli/vscode_projects/model_training/data/dataset_info"
input_size = 224
DEBUG_IMGS = True
SEED=42
DATA_MODALITY = 'all' # 'X', 'text', 'digit', 'all'

BALANCED_DATA = True
BALANCING_FACTOR = 1
MAJORITY_CLASS_ID = 0
THRESHOLD_NUM = 1
NUM_tiles = 1


def save_img_with_info(image_data,properties_text,path):
    # Create a 1-row, 2-column figure layout
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1.5, 1]})

    # Column 1: Display the actual image
    axs[0].imshow(image_data, cmap='viridis')
    axs[0].set_title("Processed Sample", fontsize=14, fontweight='bold')
    axs[0].axis('off')  # Hide image axis ticks

    # Column 2: Turn off the plot lines and render the long text block
    axs[1].axis('off')
    axs[1].text(
        x=0.0, y=1.0,               # Coordinate starting point (top-left of this subplot)
        s=properties_text, 
        fontsize=11, 
        fontfamily='monospace',     # Monospace keeps alignment neat
        verticalalignment='top', 
        horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='gray', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')

def save_img_with_info_views(image_list, text_properties_list, path):
    n = len(image_list)
    # n rows, 2 columns. squeeze=False keeps axs 2D even when n == 1
    fig, axs = plt.subplots(
        n, 2,
        figsize=(12, 6 * n),
        gridspec_kw={'width_ratios': [1.5, 1]},
        squeeze=False
    )

    for i, (image_data, properties_text) in enumerate(zip(image_list, text_properties_list)):
        # Column 1: Display the actual image
        axs[i][0].imshow(image_data, cmap='viridis')
        axs[i][0].set_title("Processed Sample", fontsize=14, fontweight='bold')
        axs[i][0].axis('off')  # Hide image axis ticks

        # Column 2: Turn off the plot lines and render the long text block
        axs[i][1].axis('off')
        axs[i][1].text(
            x=0.0, y=1.0,               # Coordinate starting point (top-left of this subplot)
            s=properties_text,
            fontsize=11,
            fontfamily='monospace',     # Monospace keeps alignment neat
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='gray', alpha=0.5)
        )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # avoid keeping figures open in memory across calls

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

def calculate_mean_std(dataloader):
    # Running sums for channels
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    print("Calculating mean and std...")
    for data, *_ in dataloader:
        # data shape: [batch_size, channels, height, width]
        # We average over batch, height, and width (dims 0, 2, 3) to keep channel dim (1)
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    # Global mean
    mean = channels_sum / num_batches
    
    # Global standard deviation: sqrt( E[X^2] - (E[X])^2 )
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    return mean, std


def get_dataloader(args):
    worker = args.num_workers
    batch_size = 32
    prefetch_factor = 2
    decode_approach = 'pil'
    load_in_memory = False
    split_workers = True
    transform = None
    apply_augmentation = False
    invert_color=True

    if apply_augmentation:
        #add a random crop transform without resizing
        augmentation_transform = T.Compose([
            #resize to 448x448
            ResizeLongestSide(448),
            T.RandomCrop(
                224, 
                pad_if_needed=True, 
                padding_mode='constant', 
                fill=(255, 255, 255) # <-- White fill for RGB PIL images
            )
        ])
    else:
        augmentation_transform = None
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),          # Scales pixels to [0, 1]
        T.Normalize(mean=[0.06040578708052635, 0.06040578708052635, 0.06040578708052635], 
                        std=[0.23823712766170502, 0.23823712766170502, 0.23823712766170502]),
    ])


    exclusion_set = set() # you can add here the ids you want to exclude from the dataset (for example because they are corrupted or for debugging purposes)
    #### EXPECTED class properties #############
    ############################################
    if DATA_MODALITY == 'all':
        selection_modality = 'text' 
    else:
        selection_modality = DATA_MODALITY 
    csv_data = pd.read_csv(LIST_OF_IDS_HANDEDNESS_PATH)
    print("Columns in the CSV:", csv_data.columns.tolist())
    csv_data, num_less_than_1_rows = melt_df(csv_data, modality=selection_modality, threshold=THRESHOLD_NUM)
    exclusion_set.update(num_less_than_1_rows)
    print("CSV after melting:", csv_data.head())
    #cols: 'ident_projet', 'lateralite', 'q_5_num_X', 'q_5_num_text', 'q_5_num_digit', 'split']
    train_data = csv_data[csv_data['split'] == 'train']
    print(f"Training samples with at least 1 chunck for modality {DATA_MODALITY}: {len(train_data)}")

    #get the number of samples for each class
    class_counts = train_data['lateralite'].value_counts()
    print(f"Class distribution in training set (after filtering for modality {DATA_MODALITY}):\n{class_counts}")

    num_0 = class_counts.get(0.0, 0)  # Count of class 0 (e.g., right-handed)
    num_1 = class_counts.get(1.0, 0)  # Count of class 1 (e.g., left-handed)
    rate = num_1/num_0 if num_0 > 0 else 0
    print(f"Number of samples per class: Class 0 = {num_0}, Class 1 = {num_1}; Total = {num_0 + num_1}")
    ################################################
    ###############################################
    
    if BALANCED_DATA:
        exclusion_set.update(generate_exclusion_set_val(csv_data, data_modality=DATA_MODALITY,
                                                    majority_class_id=MAJORITY_CLASS_ID, balancing_factor=BALANCING_FACTOR, 
                                                    label_col='lateralite', id_col='ident_projet', split='train') )
    print(len(exclusion_set), "samples will be excluded from the training set to achieve balancing.")

    #Load your dataset here
    '''train_dataset = test_handedness_dataset_all(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                            split_workers=split_workers, batch_size=batch_size, 
                                            transform=transform, modality=DATA_MODALITY, exclusion_set=exclusion_set, 
                                            huggingface_transform=False, augmentation_transform=augmentation_transform,
                                            invert_color=invert_color)'''
    train_dataset = prepare_handedness_dataset_all(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                            split_workers=split_workers, batch_size=batch_size, 
                                            transform=transform, modality=DATA_MODALITY, exclusion_set=exclusion_set, 
                                            huggingface_transform=False, augmentation_transform=augmentation_transform,
                                            invert_color=invert_color)

    train_loader = DataLoader(
        train_dataset, 
        num_workers=worker, 
        batch_size=None, 
        prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=True
    )

    if DATA_MODALITY == 'all':
        example_input_array = torch.randn(3,3, 224, 224)  # For visualizing the graph in TensorBoard
    elif NUM_tiles > 1:
        example_input_array = torch.randn(NUM_tiles, 3, 224, 224)
    else:
        example_input_array = torch.randn(3, 224, 224)
    expected_shape = example_input_array.shape

    return train_loader, expected_shape
###########
def study_dataloader():
    #this function studies the dataloader and prints out the properties of the images and labels over all batches
    pass
def random_samples_from_dataloader(args,out_folder,batches_to_show=3):
    os.makedirs(out_folder, exist_ok=True)
    #this functions shows images and properties of a random sample of images from the dataloader
    train_loader, expected_shape = get_dataloader(args)
    n_batches=0
    for batch_idx, batch in enumerate(train_loader):
        n_batches += 1
        out_folder_this_batch=os.path.join(out_folder,f"batch_{n_batches}")
        os.makedirs(out_folder_this_batch, exist_ok=True)

        img_tensor, label, subject_id_batch, questionnaire_batch, *_ = batch
        for i,subject_id in enumerate(subject_id_batch): 
            full_id = f"{subject_id}_{questionnaire_batch[i][1:]}"

            if len(expected_shape) == 3:
                num_views=1
            else:
                num_views=expected_shape[0]

            list_of_views=[]
            list_of_properties=[]
            for j in range(num_views):
                if num_views > 1:
                    single_img = img_tensor[i][j]
                else:
                    single_img = img_tensor[i]
                # Convert the single 3D tensor to PIL
                img_pil = T.ToPILImage()(single_img.cpu())
                image_data = np.array(img_pil)
                list_of_views.append(image_data)
                
                properties_text = f"Subject ID: {subject_id}\n Label: {label[i]}\n Questionnaire: {questionnaire_batch[i]}\n View: {j+1}/{num_views}"
                list_of_properties.append(properties_text)
            save_img_with_info_views(list_of_views, list_of_properties, os.path.join(out_folder_this_batch, f"sample_{i}.png"))
        if n_batches >= batches_to_show:
            break
def specific_samples_from_shards():
    #this function enables selecting specific ids and showing the images from the shards directly
    pass

def main(random_samples_from_loader=True):
    args = get_args()
    random.seed(SEED)

    if random_samples_from_loader:
        random_samples_from_dataloader(args, out_folder=os.path.join(SAVE_PATH, "random_samples"), batches_to_show=3)
        
    
if __name__ == "__main__":
    main()
    