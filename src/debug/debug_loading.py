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
from src.utils.data_loading_utils import return_file_paths
from src.utils.image_processing import ResizeLongestSide, get_augmentation_transform, get_transforms, get_mu_std, ALL_SYNTHETIC_TRANSFORMS
from src.utils.visualization import debug_images_dataset, save_img_with_info_views, save_img_with_info, tensor_debug_info, debug_images_PD
from src.utils.visualization import SubjectViewer, launch_interactive_PD
from src.utils.training_utils import set_automatic_hyperparameters
from src.debug.create_statistics_on_images import read_loader

params = {
    'selected_problem': "PD",#"PD", # "handedness"
    'grouped': False, #if true i have all elements from the same case-control group in the batch and train to distinguish the case from the controls
    'pre_training': True,
    'debug': True,

    'pre_filter_csv': False,
    'integrate_csv': False,
    'columns_to_add': ['rempli_seulq12'],
    'interactive_visualization': False,
    'interactive_properties': ['unique_id', 'split',
                               'case_control','last_avail_q',
                               'rempli_pattern','grid_pattern',
                               'case_grid_pattern','q_5_num_X',
                               'synth_label'],

    "data_modality": ['X_crop']+['digit_full','digit_crop']+['digit' for _ in range(3)]+['text_full','text_crop']+ ['text' for _ in range(3)], # 'X', 'text', 'digit', 'all' (all returns 3x3x224x224 elements instead of 3x224x224)
    "num_tiles": 3,
    'synthetic': None,#ALL_SYNTHETIC_TRANSFORMS, #or None
    'synthetic_proportions': [1/len(ALL_SYNTHETIC_TRANSFORMS) for _ in range(len(ALL_SYNTHETIC_TRANSFORMS))], #if synthetic is not None, the proportions of each synthetic class in the training set (must sum to 1)


    'model': 'resnet18',  
    "input_size": 224,
    'norm_mu': 'imagenet', #imagenet,handedness,mnist
    'norm_std': 'imagenet',
    'custom_transform': 'pad_resize_normalize', #None
    "apply_augmentation": 'random_crop_half',#'random_crop_half', #None, 
    "invert_color": True,
    "use_grid": True,

    "seed": 42,
    "balanced_data": True,
    'balance_validation': True, #if True the validation set is balanced, if False it is not balanced
    "balancing_factor": 2,
    "majority_class_id": 0,
    "threshold_num": 1,
    "invert_color": True,
    "to_grayscale": False,
    'filter_missing': 'last_q', #'all', 'last_q' #if all remove only ids with grid_pattern=0000..00 13 times, 
    #if 'last_q' with the first last_q equal to 0
    'censor_time': 'all', #long (keep only long sequences)
    'filter_modality': 'digit', #None, 'X', 'text', 'digit' (if None keep all modalities)

    #dataloader params
    "batch_size": 4,
    "prefetch_factor": 2,
    "decode_approach": "pil",
    "load_in_memory": False,
    "split_workers": True,

}

params['list_of_ids_paths'], params['data_folder'], params["h5_data_path"] = return_file_paths(params['selected_problem'], 
                                                                                                 params['grouped'], params['pre_training'])
params = set_automatic_hyperparameters(params)


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

    params['full_dataset'] = "/home/a_morelli/datasets/id_lists/final_table_with_all_info_8_7_26.csv"

    SAVE_PATH = "/home/a_morelli/vscode_projects/model_training/data/dataset_info_PD"


SHARD_PATTERN_train = os.path.join(params['data_folder'],"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(params['data_folder'],"val/worker*_shard-*.tar")
VERBOSE = True


def main(params,run_random_samples_from_loader=False, 
         run_study_loader = True, show_grids=False, 
         run_compute_time= False, run_debug_from_shards=False,
         run_explore_files=False):
    args = get_args()
    random.seed(params['seed'])
    mean,std = get_mu_std(params, verbose=VERBOSE)

    params['norm_mu'] = mean
    params['norm_std'] = std

    params = synthetic_data_override(params, verbose=True) #if i am using the synthetic version i need this line

    if params['pre_filter_csv']:
        pre_filtered_csv = run_pre_filtering()
    else:
        pre_filtered_csv = None
    
    if params['integrate_csv']:
        #integrate the csv with the original csv with all the information
        pre_filtered_csv = merge_properties_from_full_dataset_PD(params,pre_filtered_csv, params['columns_to_add'], verbose=VERBOSE)
    
    if run_random_samples_from_loader:
        if params['interactive_visualization']:
            interactive_random_samples(args, params, batches_to_show=3,mean=mean, std=std,pre_filtered_csv=pre_filtered_csv)
        else:
            random_samples_from_dataloader(args,params, out_folder=os.path.join(SAVE_PATH, "random_samples"), batches_to_show=6, 
                                        mean=mean, std=std, no_text=False, pre_filtered_csv=pre_filtered_csv)
    
    if run_compute_time:
        compute_time_to_iterate_on_dataloader(args, params, pre_filtered_csv=pre_filtered_csv)
    
    if run_study_loader:
        df = study_dataloader(args, params, max_batches=10)
        print(df.head())

    ### From here they work only for the handedness case #####
    if show_grids:
        grids_of_random_samples(args, out_folder=os.path.join(SAVE_PATH, "grids"), batches_to_show=3, mean=mean, std=std)
        
    
    if run_debug_from_shards:
        augmentation_transform = T.Compose([
            #resize to 448x448
            #ResizeLongestSide(448),
            T.RandomCrop(
                112, 
                pad_if_needed=True, 
                padding_mode='constant', 
                fill=(255, 255, 255) # <-- White fill for RGB PIL images
            )
        ])
        transform = T.Compose([
            T.Resize((params['input_size'], params['input_size'],)),
            T.ToTensor(),          # Scales pixels to [0, 1]
            T.Normalize(mean=mean, 
                            std=std),
        ])
        debug_from_shards(params,args, out_folder=os.path.join(SAVE_PATH, "debug_from_shards"), 
                          augmentation_transform=augmentation_transform, transform=transform, invert_color=params['invert_color'])
        #add some other grids (eg larger than higher, bad predictions, ...)
    if run_explore_files:
        explore_files(filename='worker94_shard-000000/D3I3J0N6.q1.hand.png')


########### Tensor visualization utils ############

def denorm(t, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std  = torch.tensor(std).view(-1, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1)

def _denorm(imgs, mean_c, std_c):
    """imgs: (K, C, H, W); mean_c/std_c: per-channel (C,) for ONE view."""
    m = torch.as_tensor(mean_c).view(-1, 1, 1)
    s = torch.as_tensor(std_c).view(-1, 1, 1)
    return (imgs * s + m).clamp(0, 1)


def _grid_one_view(imgs, mean_c, std_c, nrow, ax=None, title=None):
    """Render one view's images (K, C, H, W) as a grid onto ax (or a new fig)."""
    imgs = _denorm(imgs.detach().cpu().float(), mean_c, std_c)
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(
            figsize=(nrow * 1.6, max(1, len(imgs) / nrow) * 1.6)
        )
    ax.imshow(grid.permute(1, 2, 0).numpy())     # C×H×W → H×W×C
    if title:
        ax.set_title(title)
    ax.axis('off')
    return ax


def show_batch_grid(batch, mean, std, nrow=8, max_imgs=64,
                    title="batch", path=None, stacked=True):
    """Tile a multi-view batch into per-view grids.

    batch : (B, n, C, H, W)   — n views/modalities per sample
    mean  : (n, C)            — per-modality, per-channel denorm constants
    std   : (n, C)
    stacked=True  -> one figure, one row of grids per view
    stacked=False -> a separate figure per view (path gets _view{i} suffix)
    """
    batch = batch[:max_imgs].detach().cpu().float()
    assert batch.ndim == 5, f"expected (B, n, C, H, W), got {tuple(batch.shape)}"
    B, n, C, H, W = batch.shape
    mean = torch.as_tensor(mean).float()         # (n, C)
    std  = torch.as_tensor(std).float()
    assert mean.shape[0] == n, f"{mean.shape[0]} mean rows vs {n} views"

    rows = (B + nrow - 1) // nrow                # grid rows per view

    if stacked:
        fig, axs = plt.subplots(
            n, 1,
            figsize=(nrow * 1.6, rows * 1.6 * n),
            squeeze=False,
        )
        for i in range(n):
            _grid_one_view(
                batch[:, i], mean[i], std[i], nrow,
                ax=axs[i][0], title=f"{title} — view {i}",
            )
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    else:
        for i in range(n):
            ax = _grid_one_view(
                batch[:, i], mean[i], std[i], nrow,
                title=f"{title} — view {i}",
            )
            if path:
                stem, _, ext = path.rpartition('.')
                vp = f"{stem}_view{i}.{ext}" if stem else f"{path}_view{i}"
                ax.figure.savefig(vp, dpi=150, bbox_inches='tight')
                plt.close(ax.figure)
            else:
                plt.show()
    

#######################################
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()


####### FUnctions to get properties of dataloaders #############
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
@torch.no_grad()
def compute_per_modality_norm(loader, num_modalities=None, max_batches=None):
    """Compute per-modality, per-channel mean & std over a RAW (un-normalized) loader.

    Expects batches of shape (B, n, C, H, W). Returns tensors of shape (n, C):
        mean[i], std[i]  ->  the Normalize constants for modality i.
    Accumulates in float64 to avoid catastrophic cancellation in the variance.
    """
    pix_sum = pix_sqsum = None
    pix_count = 0  # pixels per channel per modality (B*H*W summed over batches)

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = imgs.double()                       # (B, n, C, H, W)
        if pix_sum is None:
            n, c = x.shape[1], x.shape[2]
            pix_sum   = torch.zeros(n, c, dtype=torch.float64)
            pix_sqsum = torch.zeros(n, c, dtype=torch.float64)

        pix_sum   += x.sum(dim=(0, 3, 4))       # reduce B,H,W -> (n, C)
        pix_sqsum += (x ** 2).sum(dim=(0, 3, 4))
        pix_count += x.shape[0] * x.shape[3] * x.shape[4]

    mean = pix_sum / pix_count                  # (n, C)
    var  = pix_sqsum / pix_count - mean ** 2
    std  = var.clamp_min(0).sqrt()              # clamp guards tiny negative drift

    if num_modalities is not None:
        assert mean.shape[0] == num_modalities, \
            f"found {mean.shape[0]} modalities, expected {num_modalities}"
    return mean.float(), std.float()

@torch.no_grad()
def dataset_overview(loader, num_modalities=None, num_classes=None, max_batches=50):
    """Per-modality summary for batches of shape (B, n, C, H, W).

    Reports, for each modality: pixel mean/std (per channel), value range,
    and spatial sizes seen. Optionally tallies label balance if the loader
    yields (imgs, labels).
    """
    sum_ = sqsum = None
    vmin = vmax = None
    count = 0
    sizes = defaultdict(set)                    # modality -> set of (H, W)
    label_counts = torch.zeros(num_classes) if num_classes else None

    for b, batch in enumerate(loader):
        if b >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            imgs, labels = batch[0], (batch[1] if len(batch) > 1 else None)
        else:
            imgs, labels = batch, None

        x = imgs.double()                       # (B, n, C, H, W)
        n, c = x.shape[1], x.shape[2]
        if sum_ is None:
            sum_   = torch.zeros(n, c, dtype=torch.float64)
            sqsum  = torch.zeros(n, c, dtype=torch.float64)
            vmin   = torch.full((n, c),  float('inf'),  dtype=torch.float64)
            vmax   = torch.full((n, c), -float('inf'), dtype=torch.float64)

        sum_  += x.sum(dim=(0, 3, 4))
        sqsum += (x ** 2).sum(dim=(0, 3, 4))
        count += x.shape[0] * x.shape[3] * x.shape[4]

        # per-modality, per-channel min/max
        cur_min = x.amin(dim=(0, 3, 4))         # (n, C)
        cur_max = x.amax(dim=(0, 3, 4))
        vmin = torch.minimum(vmin, cur_min)
        vmax = torch.maximum(vmax, cur_max)

        for i in range(n):
            sizes[i].add(tuple(x.shape[-2:]))

        if label_counts is not None and labels is not None:
            for l in labels.view(-1):
                label_counts[int(l)] += 1

    mean = (sum_ / count)                       # (n, C)
    std  = (sqsum / count - mean ** 2).clamp_min(0).sqrt()

    # --- report ---
    for i in range(mean.shape[0]):
        print(f"modality {i}")
        print(f"  mean (per-ch): {[round(v, 4) for v in mean[i].tolist()]}")
        print(f"  std  (per-ch): {[round(v, 4) for v in std[i].tolist()]}")
        print(f"  range        : [{vmin[i].min().item():+.4f}, "
              f"{vmax[i].max().item():+.4f}]")
        print(f"  sizes        : {sizes[i]}")
    if label_counts is not None:
        print(f"label balance : {label_counts.tolist()}")

    return {
        "mean": mean.float(), "std": std.float(),
        "min": vmin.float(),  "max": vmax.float(),
        "sizes": dict(sizes),
        "label_counts": label_counts,
    }
#########################################################################

####### Filtering functions ###############
def run_pre_filtering(params, subgroup='all',verbose = True):
    if params['selected_problem'] == "PD":
        pre_filtered_csv = pd.load_parquet(params['list_of_ids_paths']) 
        if verbose:
            print(f"Initial number of samples in the dataset: {len(pre_filtered_csv)}")
        if subgroup == 'cases':
            pre_filtered_csv = pre_filtered_csv[pre_filtered_csv['diag_park_final1_quest'] == 1]
        if verbose:
            print(f"Number of samples after filtering for subgroup '{subgroup}': {len(pre_filtered_csv)}")
            print("#"*50)
    else:
        raise ValueError(f"Filtering not available for selected_problem: {params['selected_problem']}")
    
    return pre_filtered_csv

####### LOADING DATASET AND DATALOADER #############
def get_dataloader(args,params,pre_filtered_csv):
    worker = args.num_workers

    grid_dict = None
    if params['use_grid']:
        with open(params['h5_data_path'], "rb") as f:
            grid_dict = pickle.load(f)
        print("Grid dictionary loaded successfully. Number of entries:", len(grid_dict))

    augmentation_transform = get_augmentation_transform(params) 

    transform = get_transforms(params, None)

    if params['selected_problem'] == "handedness":
        train_loader,expected_shape = handedness_dataloading(worker,params, transform, augmentation_transform, grid_dict, 
                                                             params['list_of_ids_paths'], SHARD_PATTERN_train)
    elif params['selected_problem'] == "PD":
        #exclude controls from the training if i want to reduce the asimmetry of the dataset (for example if i want to have a 1:1 ratio between cases and controls)
        exclusion_set, val_exclusion_set, counts = prepare_exclusion_sets_PD(params,verbose=VERBOSE,class_col='diag_park_final1_quest',
                                                                                   pre_computed_csv=pre_filtered_csv)
        train_df = pd.read_parquet(params['list_of_ids_paths'])
        if params['synthetic'] is None and params['pre_training']:
            train_df['fake_label'] = 1 
        train_loader,_,_,_= prepare_loaders_PD(worker,params['prefetch_factor'],params, exclusion_set, val_exclusion_set,
                                                         grid_dict, transform, SHARD_PATTERN_train, SHARD_PATTERN_val, train_df=train_df)
        expected_shape = torch.randn(params['num_tiles'], 3, 224, 224)

    return train_loader, expected_shape
def handedness_dataloading(worker,params, transform, augmentation_transform, grid_dict, list_of_ids_path, shard_pattern_train):
    exclusion_set = set() # you can add here the ids you want to exclude from the dataset (for example because they are corrupted or for debugging purposes)
    #### EXPECTED class properties #############
    ############################################
    if params['data_modality'] == 'all':
        selection_modality = 'text' 
    else:
        selection_modality = params['data_modality'] 
    csv_data = pd.read_csv(list_of_ids_path)
    print("Columns in the CSV:", csv_data.columns.tolist())
    csv_data, num_less_than_1_rows = melt_df(csv_data, modality=selection_modality, threshold=params['threshold_num'])
    exclusion_set.update(num_less_than_1_rows)
    print("CSV after melting:", csv_data.head())
    #cols: 'ident_projet', 'lateralite', 'q_5_num_X', 'q_5_num_text', 'q_5_num_digit', 'split']
    train_data = csv_data[csv_data['split'] == 'train']
    print(f"Training samples with at least 1 chunck for modality {params['data_modality']}: {len(train_data)}")

    #get the number of samples for each class
    class_counts = train_data['lateralite'].value_counts()
    print(f"Class distribution in training set (after filtering for modality {params['data_modality']}):\n{class_counts}")

    num_0 = class_counts.get(0.0, 0)  # Count of class 0 (e.g., right-handed)
    num_1 = class_counts.get(1.0, 0)  # Count of class 1 (e.g., left-handed)
    rate = num_1/num_0 if num_0 > 0 else 0
    print(f"Number of samples per class: Class 0 = {num_0}, Class 1 = {num_1}; Total = {num_0 + num_1}")
    ################################################
    ###############################################
    
    if params['balanced_data']:
        exclusion_set.update(generate_exclusion_set_val(csv_data, data_modality=params['data_modality'],
                                                    majority_class_id=params['majority_class_id'], balancing_factor=params['balancing_factor'], 
                                                    label_col='lateralite', id_col='ident_projet', split='train') )
    print(len(exclusion_set), "samples will be excluded from the training set to achieve balancing.")

    #Load your dataset here
    '''train_dataset = test_handedness_dataset_all(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                            split_workers=split_workers, batch_size=batch_size, 
                                            transform=transform, modality=params['data_modality'], exclusion_set=exclusion_set, 
                                            huggingface_transform=False, augmentation_transform=augmentation_transform,
                                            invert_color=invert_color)'''
    train_dataset = prepare_handedness_dataset_all(shard_pattern_train, decode_approach=params['decode_approach'], load_in_memory=params['load_in_memory'], 
                                            split_workers=params['split_workers'], batch_size=params['batch_size'], 
                                            transform=transform, modality=params['data_modality'], exclusion_set=exclusion_set, 
                                            huggingface_transform=False, augmentation_transform=augmentation_transform,
                                            invert_color=params['invert_color'], grid_dict=grid_dict, n_views=params['num_tiles'])

    train_loader = DataLoader(
        train_dataset, 
        num_workers=worker, 
        batch_size=None, 
        prefetch_factor=params['prefetch_factor'], # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=True
    )

    if params['data_modality'] == 'all':
        example_input_array = torch.randn(3,3, 224, 224)  # For visualizing the graph in TensorBoard
    elif params['num_tiles'] > 1 or params['use_grid']:
        example_input_array = torch.randn(params['num_tiles'], 3, 224, 224)
    else: 
        example_input_array = torch.randn(3, 224, 224)
    expected_shape = example_input_array.shape

    return train_loader, expected_shape

######## MAIN FUNCTIONS ###################
def study_dataloader(args, params, pre_filtered_csv=None, max_batches=None):
    train_loader, expected_shape = get_dataloader(args, params, pre_filtered_csv)

    def create_row(qs,sid, smodalities, smeta):
        rows = []
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
                    row = {}
                    row['subject_id'] = sid
                    row['q'] = q
                    row['modality'] = map_modality(modality)
                    row['memory'] = meta['memory_size_bytes']
                    row['pixel_sum'] = meta['pixel_sum']
                    row['pixel_sq_sum'] = meta['pixel_sq_sum']
                    row['num_pixels'] = meta['num_pixels']
                    rows.append(row)
        return rows
    df = read_loader(train_loader, create_row, max_batches=max_batches)
    return df
    
    
def compute_time_to_iterate_on_dataloader(args, params, pre_filtered_csv=None, per_batch=True):
    import time
    start = time.time()
    train_loader, expected_shape = get_dataloader(args, params, pre_filtered_csv)

    if per_batch:
        it = iter(train_loader)
        t_0 = time.time()
        for _ in range(60):      # drain the pre-filled queue
            next(it)
        t = time.time()
        print('Per batch time (short)' , (t-t_0) / 60, 'seconds')
        N = 500
        for _ in range(N):
            next(it)
        print('Per batch time' , (time.time() - t) / N, 'seconds')
    else: #total
        for batch_idx, batch in enumerate(train_loader):
            img_tensor, *_ = batch
            #print(f"Batch {batch_idx}: img_tensor shape: {img_tensor.shape}, label shape: {label.shape}, subject_id_batch shape: {subject_id_batch.shape}, questionnaire_batch shape: {questionnaire_batch.shape}")
        end = time.time()
        elapsed_time = end - start
        print(f"Time taken to iterate over the dataloader: {elapsed_time:.2f} seconds")
    
    '''ds = iter(train_dataset)      # bypass DataLoader, single process
    for _ in range(3): next(ds)
    cProfile.run('[next(ds) for _ in range(20)]', '/tmp/prof')
    pstats.Stats('/tmp/prof').sort_stats('cumulative').print_stats(25)'''

#show batches
def random_samples_from_dataloader(args, params, out_folder,batches_to_show=3, mean=None,std=None, no_text=False, pre_filtered_csv=None):
    os.makedirs(out_folder, exist_ok=True)
    #this functions shows images and properties of a random sample of images from the dataloader
    train_loader, expected_shape = get_dataloader(args, params, pre_filtered_csv)

    if pre_filtered_csv is None:
        pre_filtered_csv = pd.read_parquet(params['list_of_ids_paths'])
        print("Loaded pre_filtered_csv, for interactive visualization, from file:", params['list_of_ids_paths'])
    def get_meta(subject_id, cols=params['interactive_properties'], pre_filtered_csv=pre_filtered_csv):
        """Look up a single row by unique_id and return a subset of columns."""
        match = pre_filtered_csv.loc[pre_filtered_csv["unique_id"] == subject_id]
        if match.empty:
            return f"(no row for unique_id={subject_id})"
        row = match.iloc[0]
        return {c: row[c] for c in cols if c in row.index}

    if params['selected_problem'] == "handedness":
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
                #print('shape tensor: ', img_tensor.shape)
                for j in range(num_views):
                    if num_views > 1:
                        single_img = img_tensor[i][j]
                    else:
                        single_img = img_tensor[i]
                    
                    properties_text = tensor_debug_info(single_img, name=f"view {len(list_of_views)}", norm_mean=mean, norm_std=std)
                    if no_text:
                        properties_text = ""
                    #tensor_debug_info(single_img, name=f"view {len(list_of_views)}")

                    # Convert the single 3D tensor to PIL
                    img_pil = T.ToPILImage()(denorm(single_img, mean, std))
                    image_data = np.array(img_pil)
                    list_of_views.append(image_data)
                    
                    properties_text += f" \n Subject ID: {subject_id}\n Label: {label[i]}\n Questionnaire: {questionnaire_batch[i]}\n View: {j+1}/{num_views}"
                    list_of_properties.append(properties_text)
                save_img_with_info_views(list_of_views, list_of_properties, os.path.join(out_folder_this_batch, f"sample_{i}.png"))
            if n_batches >= batches_to_show:
                break
    elif params['selected_problem'] == "PD":
        n_batches=0
        for batch_idx, batch in enumerate(train_loader):
            n_batches += 1
            out_folder_this_batch=os.path.join(out_folder,f"batch_{n_batches}")
            os.makedirs(out_folder_this_batch, exist_ok=True)
            if no_text:
                show_metadata=False
            else:
                show_metadata=True
            debug_images_PD(mean,std, batch, out_folder_this_batch, input_is_batch=True,meta_fn=get_meta, show_metadata=show_metadata)

            if n_batches >= batches_to_show:
                break
def interactive_random_samples(args, params, batches_to_show=3,mean=None,std=None,pre_filtered_csv=None):
    #this functions shows images and properties of a random sample of images from the dataloader
    train_loader, expected_shape = get_dataloader(args, params, pre_filtered_csv)

    if pre_filtered_csv is None:
        pre_filtered_csv = pd.read_parquet(params['list_of_ids_paths'])
        print("Loaded pre_filtered_csv, for interactive visualization, from file:", params['list_of_ids_paths'])

    def get_meta(subject_id, cols=params['interactive_properties']):
        """Look up a single row by unique_id and return a subset of columns."""
        match = pre_filtered_csv.loc[pre_filtered_csv["unique_id"] == subject_id]
        if match.empty:
            return f"(no row for unique_id={subject_id})"
        row = match.iloc[0]
        return {c: row[c] for c in cols if c in row.index}

    # in a notebook, first:  %matplotlib widget   (needs ipympl)
    launch_interactive_PD(train_loader, mean, std,
                        get_meta=get_meta, max_batches=batches_to_show)

def grids_of_random_samples(args,out_folder,batches_to_show=3,mean=None,std=None):
    if os.path.exists(out_folder):
        #delete all files in the folder
        for entry in os.listdir(out_folder):
            path = os.path.join(out_folder, entry)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    else:
        os.makedirs(out_folder, exist_ok=True)
    #this functions shows images and properties of a random sample of images from the dataloader
    train_loader, expected_shape = get_dataloader(args)

    def take_random_batches(loader, n_batches, max_batches=100):
        # take n_batches (exclusive) random numbers from the 0-max_batches range
        selected_idxs = random.sample(range(max_batches), n_batches)
        i=0
        for batch in loader:
            if i in selected_idxs:
                yield batch
            i+=1

    
    for b_idx, batch in enumerate(take_random_batches(train_loader, 3, max_batches=100)):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch  # (B, n, C, H, W)
        if len(expected_shape) == 3:
            n = 1
            imgs = imgs.unsqueeze(1)  # Add a modality dimension if only one view
        else:
            n = imgs.shape[1]
        mean = torch.as_tensor(mean).expand(n, -1)   # (n, C), no copy
        std  = torch.as_tensor(std).expand(n, -1)
        show_batch_grid(
            imgs, mean, std,
            nrow=8, max_imgs=32,
            title=f"batch {b_idx}",
            path=os.path.join(out_folder,f"batch_{b_idx}.png"),     # one file per batch; each has a row per view
            stacked=True,
        )
#Reading from shards
def debug_from_shards(params,args,out_folder, augmentation_transform, transform, invert_color=True):
    #clear the files in the folder if it already exists
    if os.path.exists(out_folder):
        for entry in os.listdir(out_folder):
            path = os.path.join(out_folder, entry)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    else:
        os.makedirs(out_folder, exist_ok=True)
    shard_path = os.path.join(params['data_folder'],"train")
    csv_data = pd.read_csv(CSV_LOAD_PATH)
    csv_data = csv_data[csv_data['split'] == 'train']
    #for the rows in which modality_type==x map it to X
    csv_data['modality_type'] = csv_data['modality_type'].replace({'x': 'X'})

    def get_images_w_low_high_num(n_images,modality,questionnaire):
        filtered_data = csv_data.copy()
        if modality:
            #if modality is not none filter for the specified modality
            filtered_data = csv_data[csv_data['modality_type'] == modality]
        if questionnaire:
            #if questionnaire is not none filter for the specified questionnaire
            filtered_data = filtered_data[filtered_data['questionnaire'].isin(questionnaire)]
        #keep only rows with num>1
        filtered_data = filtered_data[filtered_data['num'] >= 1]
        #select the rows with num value in the top 25% percentile
        num_threshold_high = filtered_data['num'].quantile(0.75)
        high_num_data = filtered_data[filtered_data['num'] >= num_threshold_high]
        #select in the bottom 25% percentile
        num_threshold_low = filtered_data['num'].quantile(0.25)
        low_num_data = filtered_data[filtered_data['num'] <= num_threshold_low]
        #randomly sample n_images from each of the two dataframes
        high_num_sample = high_num_data.sample(n=n_images, random_state=params['seed'])
        low_num_sample = low_num_data.sample(n=n_images, random_state=params['seed'])
        #for each return the list of names assembled in this way: shard_name/subject_id.questionnaire.modality_type.png
        high_num_names = [f"{row['shard_name'].split('.')[0]}/{row['subject_id']}.{row['questionnaire']}.{row['modality_type']}.png" for index, row in high_num_sample.iterrows()]
        print("High num names:", high_num_names)
        low_num_names = [f"{row['shard_name'].split('.')[0]}/{row['subject_id']}.{row['questionnaire']}.{row['modality_type']}.png" for index, row in low_num_sample.iterrows()]
        print("Low num names:", low_num_names)
        return high_num_names, low_num_names
    def specific_samples_from_shards(list_of_ids,modality,questionnaire):
        #this function enables selecting specific ids and showing the images from the shards directly
        filtered_data = csv_data.copy()
        try:
            filtered_data = filtered_data[filtered_data['subject_id'].isin(list_of_ids)]
            if modality:
                #if modality is not none filter for the specified modality
                filtered_data = csv_data[csv_data['modality_type'] == modality]
            if questionnaire:
                #if questionnaire is not none filter for the specified questionnaire
                filtered_data = filtered_data[filtered_data['questionnaire'].isin(questionnaire)]
            images_per_id={}
            unique_ids = filtered_data['subject_id'].unique()
            for subject_id in unique_ids:
                subset = filtered_data[filtered_data['subject_id'] == subject_id]
                images_per_id[subject_id] = [f"{row['shard_name'].split('.')[0]}/{row['subject_id']}.{row['questionnaire']}.{row['modality_type']}.png" for index, row in subset.iterrows()]
        except Exception as e:
            print(f"Error filtering data: {e}")
            images_per_id = {}
        return images_per_id

    def grid_augmentations(n_samples,modality,questionnaire):
        #this function enables selecting specific ids and showing the images from the shards directly
        filtered_data = csv_data.copy()
        if modality:
            #if modality is not none filter for the specified modality
            filtered_data = csv_data[csv_data['modality_type'] == modality]
        if questionnaire:
            #if questionnaire is not none filter for the specified questionnaire
            filtered_data = filtered_data[filtered_data['questionnaire'].isin(questionnaire)]
        #sample n_samples from the filtered data
        sampled = filtered_data.sample(n=n_samples, random_state=params['seed'])
        names = [f"{row['shard_name'].split('.')[0]}/{row['subject_id']}.{row['questionnaire']}.{row['modality_type']}.png" for index, row in sampled.iterrows()]
        return names
    
    def save_image_list(images, path, nrow=8, size=(224, 224),
                        mean=None, std=None, title="images", stacked=True):
        """Save a list of images via show_batch_grid.

        images : list — either
                * flat list of (C, H, W) tensors  -> treated as 1 view, B=len
                * list of (n, C, H, W) tensors     -> n views per sample
                Sizes may differ; each is resized to `size`.
        mean,std : per-view (n, C) constants. If None, assumes images are already
                in [0,1] and uses identity denorm (mean=0, std=1).
        """
        def to_chw(t):
            t = torch.as_tensor(t).float()
            if t.ndim == 2:                      # H,W -> 1,H,W
                t = t.unsqueeze(0)
            return t

        # normalize each element to (n, C, H, W)
        fixed = []
        for im in images:
            im = to_chw(im)
            if im.ndim == 3:                     # (C,H,W) single view -> (1,C,H,W)
                im = im.unsqueeze(0)
            fixed.append(im)

        batch = torch.stack(fixed)               # (B, n, C, H, W)
        n = batch.shape[1]

        if mean is None or std is None:          # already-displayable images
            mean = torch.zeros(n, batch.shape[2])
            std  = torch.ones(n, batch.shape[2])

        show_batch_grid(
            batch, mean, std,
            nrow=nrow, max_imgs=len(batch),
            title=title, path=os.path.join(path,title), stacked=stacked,
        )
    def save_to_grid(list_of_paths, path, title):
        img_list = []
        for member in list_of_paths:
            shard=os.path.join(shard_path,member.split('/')[0]+".tar")
            subject_id = member.split('/')[1].split('.')[0]
            #print(shard)
            '''with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print("Corresponding rows in CSV:")
                print(csv_data[csv_data['subject_id']==subject_id])'''
            with tarfile.open(shard) as tar:
                '''for m in tar.getmembers():
                    if subject_id in m.name:
                        print(f"Extracting {m.name} from {shard}")'''
                f = tar.extractfile(member.split('/')[1])
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                img = augmentation_transform(img)
                if invert_color:
                    img = ImageOps.invert(img)
                img_tensor = transform(img)
                img_list.append(img_tensor)
        #convert to tensor
        save_image_list(img_list, path, nrow=8, size=(224, 224),
                        mean=None, std=None, title=title, stacked=True)
    def save_augmentations(image_name, path, title, n_augmentations=6):
        img_list = []
        shard=os.path.join(shard_path,image_name.split('/')[0]+".tar")
        with tarfile.open(shard) as tar:
            f = tar.extractfile(image_name.split('/')[1])
            source_img = Image.open(io.BytesIO(f.read())).convert("RGB")
            for i in range(n_augmentations):
                img = augmentation_transform(source_img)
                if invert_color:
                    img = ImageOps.invert(img)
                img_tensor = transform(img)
                img_list.append(img_tensor)
        #convert to tensor
        save_image_list(img_list, path, nrow=3, size=(224, 224),
                        mean=None, std=None, title=title, stacked=True)
    def process_path(path):
        in_tar_path = path.split('/')[1]
        questionnaire = in_tar_path.split('.')[1]
        modality = in_tar_path.split('.')[2]
        subject_id = in_tar_path.split('.')[0]
        return subject_id,questionnaire, modality

    
    #key values you can use: 'hand', 'number_random', 'X', 'hand_sentences_full', 'all' 
    #name in the shard is the same as name here
    high_num_names, low_num_names = get_images_w_low_high_num(n_images=6, modality='hand', questionnaire=['q5','q10'])
    images_per_id = specific_samples_from_shards(list_of_ids=['D3I3J0N6','A0P9N2Q7'], modality=None, questionnaire=None)
    images_to_augment = grid_augmentations(n_samples=3, modality=None, questionnaire=None)
    
    save_to_grid(high_num_names, out_folder, 'high_num_grid.png')
    save_to_grid(low_num_names, out_folder, 'low_num_grid.png')

    for subject_id in images_per_id:
        specific_names = images_per_id[subject_id]
        print("#"*50)
        print(f"Saving grid for subject {subject_id} with {len(specific_names)} images.")
        print(f"Specific names: {specific_names}")
        for q in ['q'+str(i) for i in range(1,14)]:
            this_list=[]
            for name in specific_names:
                if q==process_path(name)[1]:
                    this_list.append(name)
            this_out_folder=os.path.join(out_folder,f"{subject_id}")
            os.makedirs(this_out_folder, exist_ok=True)
            if len(this_list)>0:
                save_to_grid(this_list, this_out_folder, f'{q}_grid.png')

    for image_name in images_to_augment:
        subject_id=image_name.split('/')[1].split('.')[0]
        save_augmentations(image_name, out_folder, f'{subject_id}_augment_grid.png')

    return

def explore_files(filename='worker94_shard-000000/D3I3J0N6.q1.hand.png'):
    #this function explores the files in the shards and prints some properties of them
    shard_path = os.path.join(params['data_folder'],"train",filename.split('/')[0]+".tar")
    subject=filename.split('/')[1].split('.')[0]
    with tarfile.open(shard_path) as tar:
        print("Elements for subject ",subject)
        for member in tar.getmembers():
            if subject in member.name:
                print(f"- {member.name}")
################################################################################

if __name__ == "__main__":
    main(params)
    