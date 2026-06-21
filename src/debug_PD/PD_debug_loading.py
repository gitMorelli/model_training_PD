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

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset, melt_df
from src.utils.data_loading_utils import prepare_handedness_dataset, prepare_handedness_dataset_all, generate_exclusion_set_val, test_handedness_dataset_all
from src.utils.image_processing import ResizeLongestSide
from src.utils.visualization import debug_images_dataset

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"
CSV_LOAD_PATH = "/home/a_morelli/vscode_projects/model_training/data/inspect_statistics/merged_statistics_w_predictions_w_original.csv"

#LIST_OF_IDS_HANDEDNESS_PATH = os.path.join(SOURCE_PATH,"handedness_model_ids.csv")
LIST_OF_IDS_HANDEDNESS_PATH = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv"

#data_folder = "png_resized_padded_whitebg", "all_png_resized_padded", "all_png_whitebg" , "all_no_grids_png_whitebg" 
data_folder = "all_no_grids_png_whitebg" #"all_no_grids_png_resized_half_whitebg"
SOURCE_PATTERN = os.path.join(SOURCE_PATH,data_folder)

SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")

#QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(q) for q in range(1,14)]

SAVE_PATH = "/home/a_morelli/vscode_projects/model_training/data/dataset_info"
input_size = 224
DEBUG_IMGS = True
SEED=42
DATA_MODALITY = 'digit' #'all' # 'X', 'text', 'digit', 'all'

BALANCED_DATA = True
BALANCING_FACTOR = 1
MAJORITY_CLASS_ID = 0
THRESHOLD_NUM = 1
NUM_tiles = 3

MEAN = [0.06040578708052635, 0.06040578708052635, 0.06040578708052635]
STD = [0.23823712766170502, 0.23823712766170502, 0.23823712766170502]


def main(run_random_samples_from_loader=False, run_study_loader = False, show_grids=True, run_compute_time=True, run_debug_from_shards=False,
         run_explore_files=False):
    args = get_args()
    random.seed(SEED)

    if run_random_samples_from_loader:
        random_samples_from_dataloader(args, out_folder=os.path.join(SAVE_PATH, "random_samples"), batches_to_show=3)
    
    #to update
    if show_grids:
        grids_of_random_samples(args, out_folder=os.path.join(SAVE_PATH, "grids"), batches_to_show=3)
    
    #to update
    if run_compute_time:
        compute_time_to_iterate_on_dataloader(args)
    
    if run_study_loader:
        study_dataloader(args)
    
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
            T.Resize((224, 224)),
            T.ToTensor(),          # Scales pixels to [0, 1]
            T.Normalize(mean=MEAN, 
                            std=STD),
        ])
        debug_from_shards(args, out_folder=os.path.join(SAVE_PATH, "debug_from_shards"), 
                          augmentation_transform=augmentation_transform, transform=transform, invert_color=True)
        #add some other grids (eg larger than higher, bad predictions, ...)
    
    if run_explore_files:
        explore_files(filename='worker94_shard-000000/D3I3J0N6.q1.hand.png')



########### Tensor visualization utils ############
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
        axs[i][0].imshow(image_data)
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

def tensor_debug_info(t, name="tensor", norm_mean=None, norm_std=None):
    """Build a human-readable diagnostic string from a raw image tensor (C×H×W or H×W).

    If norm_mean/norm_std are provided, also reports whether denormalizing
    recovers a valid [0,1] display range — turning the 'guess' into a check.
    """
    t_cpu = t.detach().cpu().float()
    lines = []

    # --- shape & dtype ---
    lines.append(f"{name}")
    lines.append(f"{'shape':<14}: {tuple(t.shape)}")
    lines.append(f"{'dtype':<14}: {t.dtype}")
    lines.append(f"{'device':<14}: {t.device}")

    # --- value range (the key normalization clue) ---
    vmin, vmax = t_cpu.min().item(), t_cpu.max().item()
    lines.append(f"{'min':<14}: {vmin:+.4f}")
    lines.append(f"{'max':<14}: {vmax:+.4f}")
    lines.append(f"{'mean':<14}: {t_cpu.mean().item():+.4f}")
    lines.append(f"{'std':<14}: {t_cpu.std().item():+.4f}")

    # --- per-channel stats (catches uneven normalization) ---
    if t_cpu.ndim == 3 and t_cpu.shape[0] in (1, 3, 4):
        for c in range(t_cpu.shape[0]):
            ch = t_cpu[c]
            lines.append(
                f"  ch{c} mean/std : {ch.mean().item():+.3f} / {ch.std().item():+.3f}"
                f"  [{ch.min().item():+.3f}, {ch.max().item():+.3f}]"
            )

    # --- interpretation heuristics ---
    if vmin < -0.01 and vmax > 1.01:
        guess = "likely NORMALIZED (mean/std) — denorm before display"
    elif 0.0 <= vmin and vmax <= 1.01:
        guess = "looks like [0,1] float — ToPILImage OK"
    elif vmax > 1.5 and vmax <= 255.5:
        guess = "looks like [0,255] range"
    else:
        guess = "unusual range — inspect manually"
    lines.append(f"{'guess':<14}: {guess}")

    # --- denorm check (confirmation when mean/std are known) ---
    if norm_mean is not None and norm_std is not None and t_cpu.ndim == 3:
        mean = torch.as_tensor(norm_mean, dtype=torch.float32).view(-1, 1, 1)
        std  = torch.as_tensor(norm_std,  dtype=torch.float32).view(-1, 1, 1)
        if mean.shape[0] == t_cpu.shape[0]:        # channel count must match
            denorm = t_cpu * std + mean
            dmin, dmax = denorm.min().item(), denorm.max().item()
            lines.append(f"{'denorm range':<14}: [{dmin:+.4f}, {dmax:+.4f}]")
            # small tolerance for float drift / mild clipping at edges
            if -0.02 <= dmin and dmax <= 1.02:
                verdict = "OK — denorm recovers [0,1]"
            else:
                spill_lo = max(0.0, -dmin)
                spill_hi = max(0.0, dmax - 1.0)
                verdict = (f"OUT OF RANGE by ({spill_lo:.3f} low, {spill_hi:.3f} high) "
                           f"— wrong mean/std, or tensor isn't normalized")
            lines.append(f"{'denorm check':<14}: {verdict}")
        else:
            lines.append(f"{'denorm check':<14}: SKIPPED — mean/std has "
                         f"{mean.shape[0]} ch, tensor has {t_cpu.shape[0]}")

    # --- health flags ---
    flags = []
    if torch.isnan(t_cpu).any(): flags.append("NaN present!")
    if torch.isinf(t_cpu).any(): flags.append("Inf present!")
    if vmin == vmax:             flags.append("constant tensor (all same value)")
    if flags:
        lines.append(f"{'flags':<14}: " + ", ".join(flags))

    return "\n".join(lines)

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

def get_dataloader(args,normalized=True):
    worker = args.num_workers
    batch_size = 32
    prefetch_factor = 2
    decode_approach = 'pil'
    load_in_memory = False
    split_workers = True
    transform = None
    apply_augmentation = False
    invert_color=True
    use_grid = True

    grid_dict = None
    if use_grid:
        with open("/mnt/beegfs02/scratch/a_morelli/datasets/rr_data_h5.pkl", "rb") as f:
            grid_dict = pickle.load(f)
        print("Grid dictionary loaded successfully. Number of entries:", len(grid_dict))

    if apply_augmentation:
        #add a random crop transform without resizing
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
    else:
        augmentation_transform = None
    if normalized:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),          # Scales pixels to [0, 1]
            T.Normalize(mean=MEAN, 
                            std=STD),
        ])
    else:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),          # Scales pixels to [0, 1]
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
                                            invert_color=invert_color, grid_dict=grid_dict)

    train_loader = DataLoader(
        train_dataset, 
        num_workers=worker, 
        batch_size=None, 
        prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=True
    )

    if DATA_MODALITY == 'all':
        example_input_array = torch.randn(3,3, 224, 224)  # For visualizing the graph in TensorBoard
    elif NUM_tiles > 1 or use_grid:
        example_input_array = torch.randn(NUM_tiles, 3, 224, 224)
    else:
        example_input_array = torch.randn(3, 224, 224)
    expected_shape = example_input_array.shape

    return train_loader, expected_shape
###########
def study_dataloader(args):
    train_loader, expected_shape = get_dataloader(args)
    train_loader_un_normalized, _ = get_dataloader(args,normalized=False)
    compute_per_modality_norm(train_loader_un_normalized, num_modalities=expected_shape[1], max_batches=None)
    dataset_overview(train_loader, num_modalities=expected_shape[1], num_classes=2, max_batches=50)
def compute_time_to_iterate_on_dataloader(args):
    import time
    start = time.time()
    train_loader, expected_shape = get_dataloader(args)
    for batch_idx, batch in enumerate(train_loader):
        img_tensor, *_ = batch
        #print(f"Batch {batch_idx}: img_tensor shape: {img_tensor.shape}, label shape: {label.shape}, subject_id_batch shape: {subject_id_batch.shape}, questionnaire_batch shape: {questionnaire_batch.shape}")
    end = time.time()
    elapsed_time = end - start
    print(f"Time taken to iterate over the dataloader: {elapsed_time:.2f} seconds")
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
                
                properties_text = tensor_debug_info(single_img, name=f"view {len(list_of_views)}", norm_mean=MEAN, norm_std=STD)

                tensor_debug_info(single_img, name=f"view {len(list_of_views)}")

                # Convert the single 3D tensor to PIL
                img_pil = T.ToPILImage()(denorm(single_img, MEAN, STD))
                image_data = np.array(img_pil)
                list_of_views.append(image_data)
                
                properties_text += f" \n Subject ID: {subject_id}\n Label: {label[i]}\n Questionnaire: {questionnaire_batch[i]}\n View: {j+1}/{num_views}"
                list_of_properties.append(properties_text)
            save_img_with_info_views(list_of_views, list_of_properties, os.path.join(out_folder_this_batch, f"sample_{i}.png"))
        if n_batches >= batches_to_show:
            break
def grids_of_random_samples(args,out_folder,batches_to_show=3):
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
        mean = torch.as_tensor(MEAN).expand(n, -1)   # (n, C), no copy
        std  = torch.as_tensor(STD).expand(n, -1)
        show_batch_grid(
            imgs, mean, std,
            nrow=8, max_imgs=32,
            title=f"batch {b_idx}",
            path=os.path.join(out_folder,f"batch_{b_idx}.png"),     # one file per batch; each has a row per view
            stacked=True,
        )
def debug_from_shards(args,out_folder, augmentation_transform, transform, invert_color=True):
    #clear the files in the folder if it already exists
    if os.path.exists(out_folder):
        for entry in os.listdir(out_folder):
            path = os.path.join(out_folder, entry)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    else:
        os.makedirs(out_folder, exist_ok=True)
    shard_path = os.path.join(SOURCE_PATTERN,"train")
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
        high_num_sample = high_num_data.sample(n=n_images, random_state=SEED)
        low_num_sample = low_num_data.sample(n=n_images, random_state=SEED)
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
        sampled = filtered_data.sample(n=n_samples, random_state=SEED)
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
    shard_path = os.path.join(SOURCE_PATTERN,"train",filename.split('/')[0]+".tar")
    subject=filename.split('/')[1].split('.')[0]
    with tarfile.open(shard_path) as tar:
        print("Elements for subject ",subject)
        for member in tar.getmembers():
            if subject in member.name:
                print(f"- {member.name}")



if __name__ == "__main__":
    main()
    