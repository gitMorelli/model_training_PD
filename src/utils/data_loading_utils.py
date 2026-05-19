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

#Datasets and dataloaders for speed tests
class InMemoryWdsDataset(torch.utils.data.Dataset):
    def __init__(self, shard_files, decode_approach, transform, seq_length=39):
        self.samples = []
        self.keys = []

        # 1. Build the unbatched pipeline strictly for loading data into RAM
        loading_pipeline = (
            wds.WebDataset(shard_files)  # No shardshuffle needed for the initial cache load
            .decode(decode_approach)
            .map(lambda sample: process_wds_sample(sample, transform, seq_length))
        )

        print(f"🧠 Loading all shards into RAM (Decode: {decode_approach}, Transform: {transform})...")
        
        # 2. Iterate through the pipeline and store samples
        for img_tensor, key in tqdm(loading_pipeline, desc="Caching to RAM"):
            # CRITICAL OPTIMIZATION: Convert float32 [0.0, 1.0] to uint8 [0, 255]
            # This makes the memory footprint 4x smaller!
            img_tensor_uint8 = (img_tensor * 255).to(torch.uint8)
            
            self.samples.append(img_tensor_uint8)
            self.keys.append(key)

        print(f"✅ Successfully cached {len(self.samples)} sequences in RAM!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 3. Restore to float32 dynamically right before it goes to the model
        img_tensor = self.samples[idx].to(torch.float32) / 255.0
        key = self.keys[idx]
        return img_tensor, key

class MultiTarSequenceDataset(Dataset):
    def __init__(self, sequence_list, max_open_tars=20, transform=None):
        """
        sequence_list: A list of tuples -> (tar_file_path, [list_of_6_png_paths])
        max_open_tars: How many tar files to keep open per worker to prevent OS file limit errors.
        """
        self.sequence_list = sequence_list
        self.max_open_tars = max_open_tars
        self.transform = transform or T.ToTensor()
        self.blank_image = torch.zeros(3, 224, 224)
        
        # Dictionary to cache open tarfile objects locally per worker
        self.open_tars = {} 

    def __len__(self):
        return len(self.sequence_list)

    def _get_tar_obj(self, tar_path):
        """Manages an LRU-style cache of open tar files."""
        if tar_path not in self.open_tars:
            # If cache is full, close and remove the oldest opened tar file
            if len(self.open_tars) >= self.max_open_tars:
                oldest_tar_path = next(iter(self.open_tars))
                self.open_tars[oldest_tar_path].close()
                del self.open_tars[oldest_tar_path]
            
            # Open the new tar file and add to cache
            self.open_tars[tar_path] = tarfile.open(tar_path, 'r')
            
        return self.open_tars[tar_path]

    def __getitem__(self, idx):
        tar_path, internal_filenames = self.sequence_list[idx]
        images = []
        
        tar_obj = self._get_tar_obj(tar_path)
        
        for fname in internal_filenames:
            try:
                member = tar_obj.getmember(fname)
                f = tar_obj.extractfile(member)
                img = Image.open(f).convert('RGB')
                img = self.transform(img)
                images.append(img)
            except KeyError:
                #print(f"Warning: {fname} not found in {tar_path}.")
                # Handle missing data or pad with zeros as needed
                images.append(self.blank_image)
                
        # Stack the 6 images: [6, Channels, Height, Width]
        return torch.stack(images)


# Datasets and dataloaders for handedness model
