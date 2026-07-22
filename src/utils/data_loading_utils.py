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
from PIL import ImageOps
import random
import json
from pathlib import Path
import numpy as np
import h5py
import sys
from functools import partial
import pickle
import cv2
import gc

from src.utils.image_processing import get_augmentation_transform, ink_density, sharpness, is_uniform_image, SyntheticTransform

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
def prepare_handedness_dataset(shard_pattern, decode_approach='pil',load_in_memory=False, split_workers=True, 
                               batch_size=4, transform = None, modality='X',rate=1, balanced_data=False, exclusion_set=set()):
    def filter(sample):
        # 'sample' is a dictionary. 
        # Assumes you have decoded the class label (e.g., via .cls or custom key)
        label = sample[1].item()  # Adjust this if your label is stored differently
        subject_id = sample[2]  # Assuming the subject ID is stored in the third position of the tuple returned by select_single_modality
        
        if label == -1:
            return False  # Filter out samples with missing labels
        elif subject_id in exclusion_set:
            return False  # Filter out samples whose subject ID is in the exclusion set
        '''elif label == MAJORITY_CLASS_ID and balanced_data:
            # Keep only 20% of the majority class samples
            return random.random() < BALANCING_FACTOR*rate  # Adjust the rate as needed (e.g., 0.2 for 20%)'''
        # Always keep minority classes
        return True
    def select_single_modality(sample, transform=None, modality='X'):
        if modality == 'X':
            modality_string = 'X'
        elif modality == 'text':
            modality_string = 'hand'
        elif modality == 'digit':
            modality_string = 'number_random'

        img_tensor = None
        label = None
        blank_image = torch.zeros(3, 224, 224)  # Assuming 3 channels and 224x224 size for ResNet18
        
        for key, value in sample.items():
            #print(f"Processing key: {key} with value type: {type(value)}")
            if key.endswith((".png", ".jpg", ".jpeg")):
                parts = key.split('.')
                #example key: q5.number_random.png

                #print(f"Processing key: {key} with parts: {parts}")
                #raise Exception("Debugging: Stopping after processing the first image key to check the key structure and modality matching.")

                # If it matches our target modality, process it
                if len(parts) == 3 and parts[1].lower() == modality_string.lower():
                    try:
                        if transform is not None:
                            img_tensor = transform(value) 
                        elif isinstance(value, torch.Tensor):
                            img_tensor = value
                        else:
                            img_tensor = T.ToTensor()(value)
                    except Exception as e:
                        print(f"Skipping corrupted image {key}: {e}")
                            
            elif key.endswith("json"):
                label = torch.tensor(value.get("label", -1), dtype=torch.long)
                subject_id = value.get("subject", "unknown") 
        #raise Exception("Debugging: Stopping after processing the first sample to check the key structure, modality matching, and label extraction.")
                
        # If we are missing either the image or the label, return the filter flag (-1)
        if img_tensor is None or label is None:
            return blank_image, torch.tensor(-1, dtype=torch.long) , subject_id,0,0
        
        # Return the image directly. 
        # Shape will be (Channels, Height, Width) instead of (1, Channels, Height, Width)
        return img_tensor, label , subject_id,0,0
    
    # 1. Use glob to find all files matching the pattern
    shard_files = glob.glob(shard_pattern)
    # Sort them just to be safe so they load in order
    shard_files.sort()


    if load_in_memory:
        return 0
    else:
        # 1. Define the base WDS Pipeline
        dataset = wds.WebDataset(shard_files, shardshuffle=100)

        # 2. Conditionally apply worker splitting
        if split_workers:
            dataset = dataset.select(wds.split_by_worker)

        # 3. Apply the remaining transformations
        dataset = (dataset
            .decode(decode_approach)
            .map(lambda sample: select_single_modality(sample, transform,modality=modality)) 
            .select(filter) # to filter missing data (labelled as -1)
            .batched(batch_size,partial=False) 
        )
    return dataset

def prepare_handedness_dataset_all(shard_pattern, decode_approach='pil', load_in_memory=False, 
                               split_workers=True, batch_size=4, transform=None, exclusion_set=set(), modality='X',huggingface_transform=False,
                               augmentation_transform=None, invert_color=False,n_views=1, grid_dict = None):
    if modality == 'X':
        modality_string = 'X'
    elif modality == 'text':
        modality_string = 'hand'
    elif modality == 'digit':
        modality_string = 'number_random'
    elif modality == 'sent':
        modality_string = 'hand_sentences_full'
    else:
        modality_string = 'all'
    
    # 2. File gathering
    shard_files = glob.glob(shard_pattern)
    shard_files.sort()

    if load_in_memory:
        return 0 # Handle your in-memory logic here
        
    # 3. Build the WebDataset Pipeline
    dataset = wds.WebDataset(shard_files, shardshuffle=100)

    if split_workers:
        dataset = dataset.select(wds.split_by_worker)

    # 4. Apply the flattening and batching
    if modality=='all':
        dataset = (dataset
            #.decode(decode_approach)
            .compose(create_flattener_handedness_multimode(transform,augmentation_transform, modalities_list = ('hand', 'number_random', 'X'),
                                                exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color)) # This replaces .map() and .select()
            .batched(batch_size,partial=False)
        )
    elif grid_dict:
        dataset = (dataset
            #.decode(decode_approach)
            .compose(create_flattener_handedness_grid(transform,augmentation_transform, n_views=n_views ,modality_string=modality_string,
                                                exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color,
                                                grid_dict=grid_dict)) # This replaces .map() and .select()
            .batched(batch_size,partial=False)
        )
    elif n_views > 1:
        dataset = (dataset
            #.decode(decode_approach)
            .compose(create_flattener_handedness_multiview(transform,augmentation_transform, n_views=n_views ,modality_string=modality_string,
                                                exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color)) # This replaces .map() and .select()
            .batched(batch_size,partial=False)
        )
    else:
        dataset = (dataset
            #.decode(decode_approach)
            .compose(create_flattener_handedness(transform,augmentation_transform, modality_string=modality_string,
                                                exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color)) # This replaces .map() and .select()
            .batched(batch_size,partial=False)
        )
    
    return dataset

def create_flattener_handedness(transform_func,augmentation_transform, modality_string, exclusion_set, huggingface_transform=False, invert_color=False):
        def flatten_samples(src):
            """Takes an iterator of samples and yields multiple individual images."""
            for sample in src:
                label = None
                subject_id = "unknown"
                
                # Step A: Find the JSON file first to get the label and subject
                for key, value in sample.items():
                    if key.endswith("json"):
                        try:
                            json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                            json_data = json.loads(json_str)
                            label = torch.tensor(json_data.get("label", -1), dtype=torch.long)
                            subject_id = json_data.get("subject", "unknown")
                        except Exception as e:
                            print(f"Error parsing JSON for sample {key}: {e}")
                        break
                        
                # Step B: Filter at the sample level
                # If the sample is missing a label or has a -1 label, skip the whole sample
                if label is None or label.item() == -1:
                    continue
                    
                # Step C: Process and YIELD images one by one
                for key, value in sample.items():
                    if key.endswith((".png", ".jpg", ".jpeg")):
                        parts = key.split('.')
                        
                        # Expected format: q5.number_random.png
                        if len(parts) == 3 and parts[1].lower() == modality_string.lower():
                            questionnaire = parts[0]
                            modality_type = parts[1]

                            complete_example_id = f"{subject_id}_{questionnaire[1:]}"
                            if complete_example_id in exclusion_set: #the exclusion set ids are the subjects_id + the questionnaire number
                                #since the exclusion is specific for the questionnaire, no tfor the subject 
                                continue
                            
                            try:
                                if isinstance(value, bytes):
                                    img = Image.open(io.BytesIO(value))
                                    img = img.convert('RGB') # Standardize to RGB just in case
                                else:
                                    img = value # Fallback in case it somehow got decoded

                                if augmentation_transform is not None:
                                    img = augmentation_transform(img)
                                if invert_color:
                                    img = ImageOps.invert(img)
                                # Apply transformations
                                if transform_func is not None:
                                    if huggingface_transform:
                                        inputs = transform_func(images=img, return_tensors="pt")
                                        img_tensor = inputs['pixel_values'][0]
                                    else:
                                        img_tensor = transform_func(img) 
                                elif isinstance(img, torch.Tensor):
                                    img_tensor = img
                                else:
                                    img_tensor = T.ToTensor()(img)
                                
                                # YIELD ONE ROW AT A TIME
                                yield img_tensor, label, subject_id, questionnaire, modality_type
                                
                            except Exception as e:
                                print(f"Skipping corrupted image {key}: {e}")
                                
        return flatten_samples

def create_flattener_handedness_multiview(transform_func,augmentation_transform, n_views,modality_string, exclusion_set, 
                                          huggingface_transform=False, invert_color=False):
        def flatten_samples(src):
            """Takes an iterator of samples and yields multiple individual images."""
            for sample in src:
                label = None
                subject_id = "unknown"
                
                # Step A: Find the JSON file first to get the label and subject
                for key, value in sample.items():
                    if key.endswith("json"):
                        try:
                            json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                            json_data = json.loads(json_str)
                            label = torch.tensor(json_data.get("label", -1), dtype=torch.long)
                            subject_id = json_data.get("subject", "unknown")
                        except Exception as e:
                            print(f"Error parsing JSON for sample {key}: {e}")
                        break
                        
                # Step B: Filter at the sample level
                # If the sample is missing a label or has a -1 label, skip the whole sample
                if label is None or label.item() == -1:
                    continue
                    
                # Step C: Process and YIELD images one by one
                for key, value in sample.items():
                    if key.endswith((".png", ".jpg", ".jpeg")):
                        parts = key.split('.')
                        
                        # Expected format: q5.number_random.png
                        if len(parts) == 3 and parts[1].lower() == modality_string.lower():
                            questionnaire = parts[0]
                            modality_type = parts[1]

                            complete_example_id = f"{subject_id}_{questionnaire[1:]}"
                            if complete_example_id in exclusion_set: #the exclusion set ids are the subjects_id + the questionnaire number
                                #since the exclusion is specific for the questionnaire, no tfor the subject 
                                continue
                            
                            try:
                                if isinstance(value, bytes):
                                    img = Image.open(io.BytesIO(value))
                                    img = img.convert('RGB') # Standardize to RGB just in case
                                else:
                                    img = value # Fallback in case it somehow got decoded

                                list_of_views = []
                                for _ in range(n_views):
                                    if augmentation_transform is not None:
                                        img_view = augmentation_transform(img)
                                    else:
                                        raise Exception("Augmentation transform must be provided for multiview flattener to create different views.")
                                    
                                    if invert_color:
                                        img_view = ImageOps.invert(img_view)
                                    
                                    # Apply transformations
                                    if transform_func is not None:
                                        if huggingface_transform:
                                            inputs = transform_func(images=img_view, return_tensors="pt")
                                            img_tensor = inputs['pixel_values'][0]
                                        else:
                                            img_tensor = transform_func(img_view) 
                                    elif isinstance(img, torch.Tensor):
                                        img_tensor = img
                                    else:
                                        img_tensor = T.ToTensor()(img)
                                    
                                    list_of_views.append(img_tensor)
                                
                                stacked_views = torch.stack(list_of_views, dim=0)
                                yield stacked_views, label, subject_id, questionnaire, modality_string
                                
                            except Exception as e:
                                print(f"Skipping corrupted image {key}: {e}")
                                
        return flatten_samples

def create_flattener_handedness_multimode(
    transform_func,
    augmentation_transform, 
    modalities_list=('hand', 'number_random', 'X'), 
    exclusion_set=set(), 
    huggingface_transform=False, 
    invert_color=False
):
    def flatten_samples(src):
        reference_size = (224, 224) 
        for sample in src:
            label = None
            subject_id = "unknown"
            
            # Step A: Extract label and subject from JSON
            for key, value in sample.items():
                if key.endswith("json"):
                    try:
                        json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                        json_data = json.loads(json_str)
                        label = torch.tensor(json_data.get("label", -1), dtype=torch.long)
                        subject_id = json_data.get("subject", "unknown")
                    except Exception as e:
                        print(f"Error parsing JSON for sample {key}: {e}")
                    break
                    
            if label is None or label.item() == -1:
                continue
                
            # Step B: Pre-index all available images in this sample by (q_number, modality)
            # This handles 'subject.qX.modality.png' transparently
            image_pool = {}
            
            for key, value in sample.items():
                if key.endswith((".png", ".jpg", ".jpeg")):
                    parts = key.split('.')
                    q_num = int(parts[0][1:])
                    mod_name = parts[1].lower()
                    image_pool[(q_num, mod_name)] = value

            # Step C: Loop explicitly through questionnaires 1 to 13
            mods_lower = [m.lower() for m in modalities_list]
            
            for X in range(1, 14):
                questionnaire = f"q{X}"
                complete_example_id = f"{subject_id}_{X}"
                
                # Exclusion check per specific questionnaire
                if complete_example_id in exclusion_set:
                    continue
                
                # Check if this questionnaire even has ANY data in the sample before processing blanks
                if not any((X, m) in image_pool for m in mods_lower):
                    continue

                final_tensors = []
                for m in mods_lower:
                    raw_image_data = image_pool.get((X, m))
                    
                    if raw_image_data is not None:
                        if isinstance(raw_image_data, bytes):
                            img = Image.open(io.BytesIO(raw_image_data)).convert('RGB')
                        else:
                            img = raw_image_data
                    else:
                        img = Image.new('RGB', reference_size, color=(255,255,255))  # Blank image if missing
                        #print(f"Warning: Missing modality {m} for questionnaire {questionnaire} in sample {complete_example_id}. Using blank image.")
                            
                    if augmentation_transform is not None:
                        img = augmentation_transform(img)
                    if invert_color:
                        img = ImageOps.invert(img)
                        
                    if transform_func is not None:
                        if huggingface_transform:
                            img_tensor = transform_func(images=img, return_tensors="pt")['pixel_values'][0]
                        else:
                            img_tensor = transform_func(img)
                    else:
                        img_tensor = T.ToTensor()(img)
                        
                    final_tensors.append(img_tensor)
                
                stacked_modalities = torch.stack(final_tensors, dim=0)
                #check that it is 3,3,224,224 
                #if not print the shape and the key to debug
                '''if stacked_modalities.shape != (len(mods_lower), 3, 224, 224):
                    print(f"Unexpected shape for {complete_example_id} in questionnaire {questionnaire}: {stacked_modalities.shape}")
                    continue'''

                yield stacked_modalities, label, subject_id, questionnaire, modalities_list

                
    return flatten_samples

def create_flattener_handedness_grid(transform_func,augmentation_transform, n_views,modality_string, exclusion_set, 
                                          huggingface_transform=False, invert_color=False, grid_dict=None):
        def flatten_samples(src):
            """Takes an iterator of samples and yields multiple individual images."""
            for sample in src:
                label = None
                subject_id = "unknown"
                
                # Step A: Find the JSON file first to get the label and subject
                for key, value in sample.items():
                    if key.endswith("json"):
                        try:
                            json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                            json_data = json.loads(json_str)
                            label = torch.tensor(json_data.get("label", -1), dtype=torch.long)
                            subject_id = json_data.get("subject", "unknown")
                        except Exception as e:
                            print(f"Error parsing JSON for sample {key}: {e}")
                        break
                        
                # Step B: Filter at the sample level
                # If the sample is missing a label or has a -1 label, skip the whole sample
                if label is None or label.item() == -1:
                    continue
                    
                # Step C: Process and YIELD images one by one
                for key, value in sample.items():
                    if key.endswith((".png", ".jpg", ".jpeg")):
                        parts = key.split('.')
                        
                        # Expected format: q5.number_random.png
                        if len(parts) == 3 and parts[1].lower() == modality_string.lower():
                            questionnaire = parts[0]
                            modality_type = parts[1]
                            

                            complete_example_id = f"{subject_id}_{questionnaire[1:]}"
                            if complete_example_id in exclusion_set: #the exclusion set ids are the subjects_id + the questionnaire number
                                #since the exclusion is specific for the questionnaire, no tfor the subject 
                                continue
                            
                            try:
                                if isinstance(value, bytes):
                                    img = Image.open(io.BytesIO(value))
                                    img = img.convert('RGB') # Standardize to RGB just in case
                                else:
                                    img = value # Fallback in case it somehow got decoded
                                
                                num,grid = grid_lookup(grid_dict, subject_id, questionnaire, modality_type)
                                x_coords = [0]+sorted(list(grid[0, :]))+[img.width]
                                n_x = len(x_coords) -1
                                y_coords = [0]+sorted(list(grid[1, :]))+[img.height]
                                #n_y = len(y_coords) -1

                                list_of_views = []
                                for _ in range(n_views):
                                    #sample a random number between 1 and num included
                                    rand_num = random.randint(1,num)
                                    coordinates = (x_coords[(rand_num-1)%n_x], y_coords[(rand_num-1)//n_x], 
                                                   x_coords[(rand_num-1)%n_x+1], y_coords[(rand_num-1)//n_x+1])
                                    chunk = img.crop(coordinates)

                                    if augmentation_transform is not None:
                                        img_view = augmentation_transform(chunk)
                                    else:
                                        img_view = chunk
                                    if invert_color:
                                        img_view = ImageOps.invert(img_view)
                                    # Apply transformations
                                    if transform_func is not None:
                                        if huggingface_transform:
                                            inputs = transform_func(images=img_view, return_tensors="pt")
                                            img_tensor = inputs['pixel_values'][0]
                                        else:
                                            img_tensor = transform_func(img_view) 
                                    elif isinstance(img, torch.Tensor):
                                        img_tensor = img
                                    else:
                                        img_tensor = T.ToTensor()(img)
                                    
                                    list_of_views.append(img_tensor)
                                
                                stacked_views = torch.stack(list_of_views, dim=0)
                                yield stacked_views, label, subject_id, questionnaire, modality_string
                                
                            except Exception as e:
                                print(f"Skipping corrupted image {key}: {e}")
                                
        return flatten_samples

def test_flattener_handedness(transform_func,augmentation_transform, modality_string, exclusion_set, huggingface_transform=False, invert_color=False):
        def flatten_samples(src):
            """Takes an iterator of samples and yields multiple individual images."""
            for sample in src:
                label = None
                subject_id = "unknown"
                
                # Step A: Find the JSON file first to get the label and subject
                for key, value in sample.items():
                    if key.endswith("json"):
                        label = torch.tensor(0) #torch.tensor(value.get("label", -1), dtype=torch.long)
                        subject_id = 'A' # value.get("subject", "unknown")
                        break # Found it, no need to keep checking other keys for json
                        
                # Step B: Filter at the sample level
                # If the sample is missing a label or has a -1 label, skip the whole sample
                if label is None or label.item() == -1:
                    continue
                    
                # Step C: Process and YIELD images one by one
                for key, value in sample.items():
                    if key.endswith((".png", ".jpg", ".jpeg")):
                        parts = key.split('.')
                        
                        # Expected format: q5.number_random.png
                        if len(parts) == 3 and parts[1].lower() == modality_string.lower():
                            questionnaire = parts[0]
                            modality_type = parts[1]

                            complete_example_id = f"{subject_id}_{questionnaire[1:]}"
                            if complete_example_id in exclusion_set: #the exclusion set ids are the subjects_id + the questionnaire number
                                #since the exclusion is specific for the questionnaire, no tfor the subject 
                                continue
                            
                            try:
                                if augmentation_transform is not None:
                                    value = augmentation_transform(value)
                                if invert_color:
                                    value = ImageOps.invert(value)
                                # Apply transformations
                                if transform_func is not None:
                                    if huggingface_transform:
                                        inputs = transform_func(images=value, return_tensors="pt")
                                        img_tensor = inputs['pixel_values'][0]
                                    else:
                                        img_tensor = transform_func(value) 
                                elif isinstance(value, torch.Tensor):
                                    img_tensor = value
                                else:
                                    img_tensor = T.ToTensor()(value)
                                
                                # YIELD ONE ROW AT A TIME
                                yield img_tensor, label, subject_id, questionnaire, modality_type
                                
                            except Exception as e:
                                print(f"Skipping corrupted image {key}: {e}")
                                
        return flatten_samples

def test_handedness_dataset_all(shard_pattern, decode_approach='pil', load_in_memory=False, 
                               split_workers=True, batch_size=4, transform=None, exclusion_set=set(), modality='X',huggingface_transform=False,
                               augmentation_transform=None, invert_color=False):
    if modality == 'X':
        modality_string = 'X'
    elif modality == 'text':
        modality_string = 'hand'
    elif modality == 'digit':
        modality_string = 'number_random'
    
    # 2. File gathering
    shard_files = glob.glob(shard_pattern)
    shard_files.sort()

    if load_in_memory:
        return 0 # Handle your in-memory logic here
        
    # 3. Build the WebDataset Pipeline
    dataset = wds.WebDataset(shard_files, shardshuffle=100)

    if split_workers:
        dataset = dataset.select(wds.split_by_worker)

    # 4. Apply the flattening and batching
    dataset = (dataset
        .decode(decode_approach)
        .compose(test_flattener_handedness(transform,augmentation_transform, modality_string=modality_string,
                                             exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color)) # This replaces .map() and .select()
        .batched(batch_size,partial=False)
    )
    
    return dataset

def load_representations_handedness(output_path, as_dataframe=False, to_torch=False):
    """
    Load what run_inference_to_dataframe(..., output_path=...) wrote, and split
    the representation array along axis 1 into separately named arrays.
 
    Parameters
    ----------
    output_path : str | Path
        The same stem passed to the saver (e.g. "out/run01"). Reads
        <stem>_metadata.parquet, <stem>_representations.npy, <stem>_layout.json.
    as_dataframe : bool
        If True, return a single DataFrame: the metadata plus one column per
        modality/tile (each cell a length-N vector). Otherwise return
        (metadata, reps_by_name).
    to_torch : bool
        If True, the representation arrays are returned as torch.Tensor.
        (Ignored when as_dataframe=True.)
 
    Returns
    -------
    (metadata, reps_by_name)            if as_dataframe is False
        metadata     : pd.DataFrame, row i aligned with each array's row i
        reps_by_name : dict[str, np.ndarray|torch.Tensor], each (num_rows, N)
            keys are the descriptive names recovered from the layout:
              data_modality == 'all'      -> 'text', 'digit', 'cX'
              single modality, tiles == 1 -> '<data_modality>'
              single modality, tiles == k -> '<data_modality>_tile0' ...
    df : pd.DataFrame                    if as_dataframe is True
    """
    base_name = os.path.basename(output_path)
    meta_file = os.path.join(output_path,base_name + '_metadata.parquet')
    repr_file = os.path.join(output_path,base_name + '_representations.npy')
    layout_file = os.path.join(output_path,base_name + '_layout.json')
 
 
    metadata = pd.read_parquet(meta_file)
    representations = np.load(repr_file)
    with open(layout_file) as fh:
        layout = json.load(fh)
 
    axis1_labels = layout.get('axis1_labels')
    data_modality = layout.get('data_modality')
 
    if axis1_labels is None:
        # (num_rows, N): a single representation, named after the modality.
        reps_by_name = {data_modality: representations}
    else:
        # (num_rows, M, N): fan axis 1 out into named (num_rows, N) arrays.
        if representations.shape[1] != len(axis1_labels):
            raise ValueError(
                f"axis-1 size {representations.shape[1]} does not match the "
                f"{len(axis1_labels)} labels in the layout ({axis1_labels})."
            )
        reps_by_name = {
            label: np.ascontiguousarray(representations[:, i, :])
            for i, label in enumerate(axis1_labels)
        }
 
    if as_dataframe:
        df = metadata.copy()
        for name, arr in reps_by_name.items():
            df[f'repr_{name}'] = list(arr)  # one length-N vector per cell
        return df, None
 
    if to_torch:
        reps_by_name = {k: torch.from_numpy(v) for k, v in reps_by_name.items()}
 
    return metadata, reps_by_name
###################################################################################################################

############### Datasets and dataloaders for PD model ########################
##############################################################################
#change the exclusion criteria to use only the id, not id+questionnaire
#modify to manage masked elements of the sequence (for control subjects they are masked i think, or maybe I have removed them, should check!)

ALL_MODALITIES = ['hand', 'number_random', 'X']
# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def all_arrays(obj):
    if isinstance(obj, list):
        return all(all_arrays(x) for x in obj)
    return isinstance(obj, np.ndarray)

def _parse_sample_json(sample):
    """Find and parse the JSON metadata blob in a webdataset sample."""
    for key, value in sample.items():
        if key.endswith("json"):
            try:
                json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                return json.loads(json_str)
            except Exception as e:
                print(f"Error parsing JSON for sample {key}: {e}")
                return None
    return None

def _index_images(sample, grouped):
    """Index images by q_number/modality (flat) or subject/q_number/modality (grouped).

    Keys look like 'qX.modality.png' (flat) or 'subject.qX.modality.png' (grouped).
    """
    pool = {}
    #print("Debug indexing function ..")
    for key, value in sample.items():
        if not key.endswith((".png", ".jpg", ".jpeg")):
            continue
        parts = key.split('.')
        if grouped:
            subj = parts[0].upper()
            q_num = int(parts[1][1:])
            mod_name = parts[2].lower()
            #print(f"Indexing grouped image: subject={subj}, questionnaire={q_num}, modality={mod_name}", flush=True)
            pool.setdefault(subj, {}).setdefault(q_num, {})[mod_name] = value
        else:
            q_num = int(parts[0][1:])
            mod_name = parts[1].lower()
            pool.setdefault(q_num, {})[mod_name] = value
    return pool

def _make_subject_sequence_builder(transform_func, augmentation_transform_list,
                                   modality_string_list, original_modality_names,
                                   grid_dict, censor_time, filter_modality,
                                   huggingface_transform, invert_color, debug, train_df, exp_params=None, is_synthetic = False):
    """Return a function that builds the full sequence for one subject.

    All the static configuration is captured in the closure, so the returned
    builder only needs the per-subject data.
    """

    def build_questionnaire_views(X, images_for_q, original_id, questionnaire_info, synth_transform=None):
        """Build the list of augmented views for one questionnaire."""
        list_of_views = []
        list_of_modality_names = []
        # if the image was rescaled, the grid coordinates must be rescaled too
        rescale_factor = questionnaire_info[str(X)]['rescale_factor']

        for current_mode in ALL_MODALITIES:
            num, grid = grid_lookup(grid_dict, original_id, 'q' + str(X), current_mode)

            # indexes of modality_string_list matching the current mode
            mask = [j for j, m in enumerate(modality_string_list) if m == current_mode]
            if len(mask) == 0:
                continue
            selected_transforms = [augmentation_transform_list[j] for j in mask]
            selected_modality_names = [original_modality_names[j] for j in mask]

            raw_image_data = images_for_q.get(current_mode.lower())

            imputed = False
            if isinstance(raw_image_data, bytes):
                img = Image.open(io.BytesIO(raw_image_data)).convert('RGB')
            elif num <= 0 or raw_image_data is None:
                # modality not present -> impute a blank image
                img = Image.new('RGB', (224, 224), color=(255, 255, 255))
                imputed = True
            else:
                img = raw_image_data
            
            if synth_transform: #i am creating synthetic time-series
                img = synth_transform(img, X-1)
            

            if debug:
                selected_transforms = [('original',None)] + selected_transforms
                selected_modality_names = ([selected_modality_names[0].split('_')[0] + '_original']
                                           + selected_modality_names)

            for k, augmentation_transform in enumerate(selected_transforms):
                if augmentation_transform[0] == 'original':
                    img_view = img.copy()
                elif augmentation_transform[0] == 'grid':
                    # select a random grid crop
                    if num > 0:
                        x_coords = [0] + sorted(list(grid[0, :] * rescale_factor[0])) + [img.width]
                        n_x = len(x_coords) - 1
                        y_coords = [0] + sorted(list(grid[1, :] * rescale_factor[1])) + [img.height]
                        rand_num = random.randint(1, num)
                        coordinates = (x_coords[(rand_num - 1) % n_x],
                                       y_coords[(rand_num - 1) // n_x],
                                       x_coords[(rand_num - 1) % n_x + 1],
                                       y_coords[(rand_num - 1) // n_x + 1])
                        img_view = img.crop(coordinates)
                    else:
                        img_view = img.copy()
                    img_view = augmentation_transform[1](img_view)  # apply the callable transform
                elif augmentation_transform[0] is None:
                    img_view = img.copy()
                else:
                    # a callable transform
                    img_view = augmentation_transform[1](img)

                if invert_color and not debug:
                    img_view = ImageOps.invert(img_view)
                
                to_grayscale = exp_params.get('to_grayscale', False) if exp_params else False
                if to_grayscale:
                    img_view = T.Grayscale(num_output_channels=1)(img_view)

                if transform_func is not None:
                    if huggingface_transform:
                        img_tensor = transform_func(images=img_view, return_tensors="pt")['pixel_values'][0]
                    else:
                        img_tensor = transform_func(img_view)
                else:
                    img_tensor = T.ToTensor()(img_view)

                if debug:
                    if augmentation_transform[0] == 'original':
                        metadata = debug_image_properties(img_view)
                    else:
                        metadata = debug_image_properties(img_tensor)
                    metadata['selected_transform'] = selected_transforms[k]
                    metadata['imputed'] = imputed
                    list_of_views.append(metadata)
                else:
                    list_of_views.append(img_tensor)
                list_of_modality_names.append(selected_modality_names[k])

        return list_of_views, list_of_modality_names, rescale_factor

    def build_subject_sequence(subject_id, last_q, questionnaire_info, subject_images, case_grid_pattern, rempli_seulq12):
        """Loop questionnaires 1..13 and build the sequence for one subject.

        Returns (sequence, questionnaires, modalities, resized_list) or None
        if no usable frames were found.
        """
        if is_synthetic:
            persona_seed = random.randint(0, 2**10 - 1)
            synth_transform = SyntheticTransform(exp_params,subject_id,train_df, persona_seed, n_steps=13, jitter=0.05)
        else:
            synth_transform = None

        original_id = subject_id.split('_')[0]

        frames = []          # each entry -> (n_views, C, H, W)
        questionnaires = []  # per-frame metadata
        resized_list = []
        modalities = []

        q_to_keep=questionnaires_to_keep(last_q, censor_time, questionnaire_info, train_df,subject_id, subject_images, case_grid_pattern, rempli_seulq12)

        for X in range(1, 14):
            if X not in q_to_keep:  # questionnaire not to keep based on censoring
                continue
            if X not in subject_images:  # questionnaire not in the shard file
                continue

            num_filter_modality, _ = grid_lookup(grid_dict, original_id, 'q' + str(X), filter_modality)
            if num_filter_modality <= 0:
                # filter modality missing -> skip questionnaire
                # (other missing modalities are imputed instead)
                continue

            views, names, rescale_factor = build_questionnaire_views(
                X, subject_images[X], original_id, questionnaire_info, synth_transform=synth_transform)

            if len(views) == 0:
                continue

            frames.append(views if debug else torch.stack(views, dim=0))
            questionnaires.append(X)
            resized_list.append(rescale_factor)
            modalities.append(names)

        if not frames:
            return None

        sequence = frames if debug else torch.stack(frames, dim=0)  # (T_i, n_views, C, H, W)
        return sequence, questionnaires, modalities, resized_list

    return build_subject_sequence

def keep_questionnaire(last_q,censor_time, questionnaire_info, questionnaire_number):
    '''return true if the quesitonnaire can be keeped, return false if i have to discard it'''
    if censor_time == -1: #keep all questionnaires
        return True
    elif censor_time == 0:
        return questionnaire_number <= last_q
    else:
        questionnaire_dt=questionnaire_info[questionnaire_number]['case_dt_dateq']
        return questionnaire_dt <= -censor_time #eg if censor time=1 -> i keep
        #all questionnaires that are at least 1 year before the case date, so dt_q<=-1

def questionnaires_to_keep(last_q, censor_time, questionnaire_info, train_df,subject_id,subject_images, case_grid_pattern, rempli_seulq12):
    '''take case_grid_pattern, the case_dt_q for each questionnaire, the training_df and the filtering modality to decide
    which questionnaires to sample for training
    case_grid_pattern is between 1 and 13
    case_dt can be missing (NaN) or numeric (remember is for the case only)'''
    if train_df is None and censor_time != 'pre_diagnosis':
        raise ValueError("train_df must be provided for censor_time other than 'pre_diagnosis' ")
    else:
        grid_pattern = train_df.loc[train_df['unique_id']==subject_id,'grid_pattern'].values[0]

    def filter_grid_pattern(case_grid_pattern, selected_questionnaires):
        new_list = []
        for q in selected_questionnaires:
            if case_grid_pattern[q-1]=='1':
                new_list.append(q)
        return new_list

    if censor_time=='all':
        list_questionnaires = list(range(1,14))
    elif censor_time=='pre_diagnosis':
        list_questionnaires = [q for q in range(1,last_q+1)]
        list_questionnaires = filter_grid_pattern(case_grid_pattern, list_questionnaires)
    elif censor_time=='pre_diagnosis_1y':
        list_questionnaires = [q for q in range(1,last_q+1) if questionnaire_info[q]['case_dt_dateq']<=-1]
        list_questionnaires = filter_grid_pattern(case_grid_pattern, list_questionnaires)
    elif censor_time=='last_and_previous':
        list_questionnaires = [last_q]
        for q in range(last_q-1,0,-1):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
                break 
    elif censor_time=='last_and_successive':
        list_questionnaires = [last_q]
        for q in range(last_q+1,14):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
                break
    elif censor_time=='last_successive_and_previous':
        list_questionnaires = [last_q]
        for q in range(last_q+1,14):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
                break
        for q in range(last_q-1,0,-1):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
                break 
    elif censor_time=='successive':
        list_questionnaires = []
        for q in range(last_q+1,14):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
        list_questionnaires = filter_grid_pattern(case_grid_pattern, list_questionnaires)
    elif censor_time=='long':
        def count_ones(s):
            return s.count('1')
        if count_ones(grid_pattern)>=8:
            list_questionnaires = [q for q in range(1,14)]
        else:
            list_questionnaires = []
    
    if rempli_seulq12==0: 
        #remove questionnaires>=12 if present and q12 was not filled alone
        list_questionnaires = [q for q in list_questionnaires if q<12]
    #print(f"List {subject_id}: ",list_questionnaires)
    return list_questionnaires

#---------- yield smaples
def create_sequence_flattener_PD_multiview(transform_func, augmentation_transform_list, n_views,
                                           modality_string_list, original_modality_names,
                                           exclusion_set, exp_params=None,huggingface_transform=False,
                                           invert_color=False, grid_dict=None, censor_time='pre_diagnosis',
                                           filter_modality='number_random', debug=False, train_df=None):
    #check if you have a column called 'synth_label' in the train_df
    if train_df is not None and 'synth_label' in train_df.columns:
        is_synthetic = True
    else: 
        is_synthetic = False
    #print(is_synthetic, "is_synthetic", flush=True)
    
    build_sequence = _make_subject_sequence_builder(
        transform_func, augmentation_transform_list, modality_string_list,
        original_modality_names, grid_dict, censor_time, filter_modality,
        huggingface_transform, invert_color, debug, train_df, exp_params=exp_params, is_synthetic=is_synthetic)

    def flatten_samples(src):
        for sample in src:
            json_data = _parse_sample_json(sample)
            if json_data is None:
                continue
            
            subject_id = json_data.get("subject", "unknown")  # XXX_YYY format
            if is_synthetic:
                #get the label from the train_df
                synth_label = train_df.loc[train_df['unique_id']==subject_id,'synth_label'].values
                label = torch.tensor(synth_label[0], dtype=torch.long) if len(synth_label)>0 else torch.tensor(-1, dtype=torch.long)
                #print("synthetic label for subject", subject_id, "is", label.item(),flush=True)
            elif train_df is not None and 'fake_label' in train_df.columns:
                fake_label = train_df.loc[train_df['unique_id']==subject_id,'fake_label'].values
                label = torch.tensor(fake_label[0], dtype=torch.long) if len(fake_label)>0 else torch.tensor(-1, dtype=torch.long)
            else:
                label = torch.tensor(json_data.get("label", -1), dtype=torch.long)
            questionnaire_info = json_data.get("questionnaire_info", {})
            last_q = json_data.get("last_q", None)
            case_grid_pattern = json_data.get("case_grid_pattern", None)
            rempli_seulq12 = json_data.get("rempli_seulq12", 1)

            if label.item() == -1:
                continue
            if subject_id in exclusion_set:
                continue

            image_pool = _index_images(sample, grouped=False)

            result = build_sequence(subject_id, last_q, questionnaire_info, image_pool,case_grid_pattern, rempli_seulq12)
            if result is None:
                continue

            sequence, questionnaires, modalities, resized_list = result
            yield sequence, label, subject_id, questionnaires, modalities, resized_list

    return flatten_samples

def create_sequence_group_flattener_PD_multiview(transform_func, augmentation_transform_list,
                                    modality_string_list, original_modality_names,
                                    exclusion_set, grid_dict, censor_time,
                                    filter_modality, exp_params=None,huggingface_transform=False,
                                    invert_color=False, debug=False, train_df=None, **kw):
    build_sequence = _make_subject_sequence_builder(
        transform_func, augmentation_transform_list, modality_string_list,
        original_modality_names, grid_dict, censor_time, filter_modality,
        huggingface_transform, invert_color, debug, train_df,exp_params=exp_params)

    def flatten_groups(src):
        for sample in src:
            json_data = _parse_sample_json(sample)
            if json_data is None:
                continue

            image_pool = _index_images(sample, grouped=True)  # subj -> q -> mod
            group_id = json_data["group_id"]

            '''print("Start debugging dataloader ...")
            print(image_pool.keys())
            print("length exclusion set:", len(exclusion_set))
            print("json -> \n", json_data["subjects"], flush=True)
            assert 1==0, "debug"'''

            group = []
            for subject_id, meta in json_data["subjects"].items():
                if subject_id in exclusion_set:
                    continue
                result = build_sequence(subject_id, meta["last_q"],
                                        meta["questionnaire_info"],
                                        image_pool.get(subject_id, {}),meta['case_grid_pattern'])
                if result is None:
                    continue
                sequence, questionnaires, modalities, resized = result
                label = torch.tensor(meta["label"], dtype=torch.long)
                group.append((sequence, label, subject_id,
                              questionnaires, modalities, resized))

            # only yield valid groups: exactly one positive, ≥1 control
            labels = [g[1].item() for g in group]
            if sum(labels) == 1 and len(group) >= 2:
                yield group_id, group

    return flatten_groups

def create_sequence_flattener_PD_grid(transform_func, augmentation_transform, n_views, modality_string,
                                              exclusion_set, huggingface_transform=False, invert_color=False,
                                              grid_dict=None, censor_time='pre_diagnosis'):
    def flatten_samples(src):
        for sample in src:
            label = None
            subject_id = "unknown"

            # Step A: JSON → label + subject (unchanged)
            for key, value in sample.items(): 
                if key.endswith("json"):
                    try:
                        json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                        json_data = json.loads(json_str)
                        label = torch.tensor(json_data.get("label", -1), dtype=torch.long)
                        subject_id = json_data.get("subject", "unknown") #is in the XXX_YYY format
                        questionnaire_info = json_data.get("questionnaire_info", {})
                        last_q = json_data.get("last_q", None)
                    except Exception as e:
                        print(f"Error parsing JSON for sample {key}: {e}")
                    break

            # Step B: sample-level filter (unchanged)
            if label is None or label.item() == -1:
                continue
            if subject_id in exclusion_set: #for PD i exclude full subject data (subjects are in the format XX_YY for exclusion)
                continue

            # Pre-index all available images in this sample by (q_number, modality)
            # This handles 'subject.qX.modality.png' transparently
            image_pool = {}
            
            for key, value in sample.items():
                if key.endswith((".png", ".jpg", ".jpeg")):
                    parts = key.split('.')
                    q_num = int(parts[0][1:])
                    mod_name = parts[1].lower()
                    if q_num not in image_pool:
                        image_pool[q_num] = {}
                    image_pool[q_num][mod_name] = value

            # Loop explicitly through questionnaires 1 to 13
            # build the SEQUENCE instead of yielding per image
            frames = []            # each entry -> (n_views, C, H, W)
            questionnaires = []    # per-frame metadata, kept as a list
            resized_list = []
            modalities = []
            for X in range(1, 14):
                if not keep_questionnaire(last_q,censor_time, questionnaire_info, X):
                    continue
                if X not in image_pool: #i skip questionnaires that were not put in the shard file
                    continue
                #i skip the subject if the modality is not present (0 chunks)
                original_id = subject_id.split('_')[0] #the original subject id is the first part of the subject_id
                num, grid = grid_lookup(grid_dict, original_id, 'q' + str(X), modality_string)
                if num <=0:
                    continue
                raw_image_data = image_pool[X].get(modality_string)

                try:
                    if isinstance(raw_image_data, bytes):
                        img = Image.open(io.BytesIO(raw_image_data)).convert('RGB')
                    else:
                        img = raw_image_data

                    rescale_factor= questionnaire_info[str(X)]['rescale_factor']#if the image was rescaled i have to rescale the grid coordinates too
                    grid[0,:] = grid[0,:]*rescale_factor[0]
                    grid[1,:] = grid[1,:]*rescale_factor[1]
                    x_coords = [0] + sorted(list(grid[0, :])) + [img.width]
                    n_x = len(x_coords) - 1
                    y_coords = [0] + sorted(list(grid[1, :])) + [img.height]

                    list_of_views = []
                    list_of_modality_names = []
                    for i in range(n_views):

                        rand_num = random.randint(1, num)
                        coordinates = (x_coords[(rand_num - 1) % n_x],     y_coords[(rand_num - 1) // n_x],
                                        x_coords[(rand_num - 1) % n_x + 1], y_coords[(rand_num - 1) // n_x + 1])
                        chunk = img.crop(coordinates)

                        img_view = augmentation_transform(chunk) if augmentation_transform is not None else chunk
                        if invert_color:
                            img_view = ImageOps.invert(img_view)

                        if transform_func is not None:
                            if huggingface_transform:
                                img_tensor = transform_func(images=img_view, return_tensors="pt")['pixel_values'][0]
                            else:
                                img_tensor = transform_func(img_view)
                        else:
                            img_tensor = T.ToTensor()(img_view)

                        list_of_views.append(img_tensor)
                        list_of_modality_names.append(modality_string+f"_{i}")

                    frames.append(torch.stack(list_of_views, dim=0))   # (n_views, C, H, W)
                    questionnaires.append(X)
                    resized_list.append(rescale_factor)  # or to_rescale if you want True/False
                    modalities.append(list_of_modality_names)

                except Exception as e:
                    print(f"Skipping corrupted image {key}: {e}")

            # Step D: yield the whole sequence ONCE
            if not frames:
                continue

            sequence = torch.stack(frames, dim=0)   # (T_i, n_views, C, H, W)
            yield sequence, label, subject_id, questionnaires, modalities, resized_list

    return flatten_samples

#------ collate functions
def collate_variable_sequences_PD_grouped(samples, debug=False):
    """Collate for the grouped loader.
 
    Each sample is one GROUP:
        (sequences, labels, subjects, questionnaires, modalities, resized, group_id)
    where sequences is (G, T, k, C, H, W) if stacked, or a list of per-subject
    sequences (debug mode, or if you switched to list-yield for variable T).
 
    Groups are flattened into individual subjects, then collated with the same
    core as the flat version. Two extra outputs are appended:
        group_ids     : list of len N, the group_id string for each subject
        group_seq_ids : tensor (N,), index of the group within this batch
                        (lets you regroup subjects downstream)
    """
    sequences, labels_parts, subject_ids = [], [], []
    questionnaires, modalities_list, resized_list = [], [], []
    group_ids, group_seq_ids = [], []
 
    for g, s in enumerate(samples):
        (grp_seqs, grp_labels, grp_subjects,
         grp_qs, grp_mods, grp_res, group_id) = s
 
        # unstack (G, T, k, C, H, W) -> list of G tensors (T, k, C, H, W)
        if isinstance(grp_seqs, torch.Tensor):
            grp_seqs = list(torch.unbind(grp_seqs, dim=0))
 
        sequences.extend(grp_seqs)
        labels_parts.append(grp_labels)                    # tensor (G,)
        subject_ids.extend(grp_subjects)
        questionnaires.extend(grp_qs)
        modalities_list.extend(grp_mods)
        resized_list.extend(grp_res)
        group_ids.extend([group_id] * len(grp_subjects))
        group_seq_ids.extend([g] * len(grp_subjects))
 
    labels = torch.cat(labels_parts)
 
    base = _collate_subject_level(sequences, labels, subject_ids, questionnaires,
                                  modalities_list, resized_list, debug=debug)
 
    return (*base, group_ids, torch.tensor(group_seq_ids))

def _collate_subject_level(sequences, labels, subject_ids, questionnaires,
                           modalities_list, resized_list, debug=False):
    """Shared core: takes per-subject lists and builds the flat batch.
 
    sequences      : list of (T_i, k, C, H, W) tensors (or nested lists if debug)
    labels         : tensor (N,)
    subject_ids    : list of N strings
    questionnaires : list of N lists of questionnaire numbers
    modalities_list: list of N lists (per timestep, list of modality names)
    resized_list   : list of N lists of rescale factors
    """
    if debug:
        lengths = torch.tensor([len(seq) for seq in sequences])
    else:
        lengths = torch.tensor([seq.shape[0] for seq in sequences])
 
    seq_ids, slot_ids = [], []
    resized, modalities = [], []
    if debug:
        frames = []
    else:
        frames = torch.cat(sequences, dim=0)               # (sum T_i, k, C, H, W)
 
    for b, qs in enumerate(questionnaires):
        for i, q in enumerate(qs):
            seq_ids.append(b)
            slot_ids.append(int(q) - 1)                    # slots 0..12
            resized.append(resized_list[b][i])
            modalities.append(modalities_list[b][i])
            if debug:
                frames.append(sequences[b][i])
    
    # if the batch has three subjects then frames, seq_ids, slot_ids have dimension sum(T_i) where T_i is the number of time-steps for subject i
    return (frames,  # list of dicts if debug
            torch.tensor(seq_ids), #which subject the frame is from
            torch.tensor(slot_ids), #which questionnaire/time-step if the frame
            lengths, 
            labels,
            resized,
            subject_ids,
            modalities)
 
def collate_variable_sequences_PD(samples, debug=False):
    """Collate for the flat (per-subject) loader. Same output as before."""
    sequences       = [s[0] for s in samples]              # each (T_i, k, C, H, W)
    labels          = torch.stack([s[1] for s in samples])
    subject_ids     = [s[2] for s in samples]
    questionnaires  = [s[3] for s in samples]
    modalities_list = [s[4] for s in samples]
    resized_list    = [s[5] for s in samples]
 
    return _collate_subject_level(sequences, labels, subject_ids, questionnaires,
                                  modalities_list, resized_list, debug=debug)

def collate_groups_PD(batch_of_groups, debug=False):
    sequences, labels, subject_ids = [], [], []
    questionnaires, modalities_list, resized_list = [], [], []
    group_ids = []

    for g_idx, (gid, group) in enumerate(batch_of_groups):
        for seq, lab, sid, qs, mods, res in group:
            sequences.append(seq)
            labels.append(lab)
            subject_ids.append(sid)
            questionnaires.append(qs)
            modalities_list.append(mods)
            resized_list.append(res)
            group_ids.append(g_idx)

    labels = torch.stack(labels)
    out = _collate_subject_level(sequences, labels, subject_ids, questionnaires,
                                 modalities_list, resized_list, debug=debug)
    return (*out, torch.tensor(group_ids))

#------ Build dataset --------
def prepare_PD_dataset(shard_pattern, split_workers=True, batch_size=4, transform=None, exclusion_set=set(), modality='X',
                       huggingface_transform=False,augmentation_transform=None, invert_color=False,n_views=1, grid_dict = None,
                       censor_time='pre_diagnosis', filter_modality='digit', debug=False, grouped=False, train_df=None, exp_params=None):
    '''
    if n_views is fractional i sample a fraction n_view of the patches for each image; else i use the same n_view for all iamges
    '''
    modalities_map = {
        'X': 'X',
        'text': 'hand',
        'digit': 'number_random',
        'digitOrdered': 'number',
        'sent': 'hand_sentences_full'
    }
    if isinstance(modality, list):
        modality_string = [m.split('_')[0] for m in modality]  # Normalize to base modality names (keep original name for debugging)
        modality_string = [modalities_map[m] for m in modality_string if m in modalities_map]
    elif modality in modalities_map:
        modality_string = modalities_map[modality]
    else:
        modality_string = 'all'
    filter_modality = modalities_map.get(filter_modality)  # Normalize filter modality

    # 2. File gathering
    shard_files = glob.glob(shard_pattern)
    shard_files.sort()
        
    # 3. Build the WebDataset Pipeline
    dataset = wds.WebDataset(shard_files, shardshuffle=100, 
                             workersplitter=wds.split_by_worker if split_workers else None) #shuffle the shards, but not the samples inside the shards 
    
    
    #measure the size of the first 50 samples in the dataset to estimate the mean and max sample size (before decoding)
    #and determine the impact of retaining samples for shuffling
    sizes = []
    for i, sample in enumerate(dataset):   # raw wds dataset, before .shuffle/.compose
        sizes.append(sum(len(v) for v in sample.values() if isinstance(v, bytes)))
        if i == 50: break
    print(f"MEMORY TEST --------------> mean sample: {np.mean(sizes)/1e6:.1f} MB, max: {np.max(sizes)/1e6:.1f} MB")
    

    shuffle_buffer = exp_params.get('shuffle_buffer', 100) if exp_params else 100
    if shuffle_buffer: 
        dataset = dataset.shuffle(shuffle_buffer) #shuffle the samples

    if grid_dict: 
        if modality_string != 'all' and not isinstance(modality, list) and n_views >= 1:
            dataset = (dataset
                #.decode(decode_approach)
                .compose(create_sequence_flattener_PD_grid(transform,augmentation_transform, n_views=n_views ,modality_string=modality_string,
                                                    exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color,
                                                    grid_dict=grid_dict, censor_time=censor_time)) # This replaces .map() and .select()
                .batched(batch_size,
                        collation_fn=partial(collate_variable_sequences_PD, questionnaire_to_slot=None),
                        partial=False)
            )
        elif isinstance(modality, list):
            compose_fn = create_sequence_group_flattener_PD_multiview if grouped else create_sequence_flattener_PD_multiview
            collate_fn = collate_groups_PD if grouped else collate_variable_sequences_PD

            dataset = (dataset
                #.decode(decode_approach)
                .compose(compose_fn(transform, augmentation_transform, n_views=n_views, modality_string_list=modality_string,
                                    exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color,
                                    grid_dict=grid_dict, censor_time=censor_time,
                                    original_modality_names=modality, filter_modality=filter_modality, debug=debug, train_df=train_df,
                                    exp_params=exp_params)) # This replaces .map() and .select()
                .batched(batch_size,
                        collation_fn=partial(collate_fn, debug=debug),
                        partial=False)
            )
        else:
            raise ValueError("grid_dict must be provided for PD dataset preparation.")
        
    
    return dataset

#pipelines
def prepare_loaders_PD(worker,prefetch_factor,exp_params,exclusion_set,val_exclusion_set, grid_dict,transform, 
                       SHARD_PATTERN_train, SHARD_PATTERN_val, train_df=None):
    def worker_init_fn(worker_id):
        # Force OpenCV to use a single thread per DataLoader worker process
        cv2.setNumThreads(0)
        gc.collect()
        gc.freeze()   # move inherited objects to permanent generation; GC won't touch them

    print(f"Reading from {SHARD_PATTERN_train} and {SHARD_PATTERN_val}..")
    augmentation_transform = get_augmentation_transform(exp_params)
    print("Augmentation transform:", augmentation_transform)
    #assert 1==0 , "STOPPED HERE TO CHECK AUGMENTATION TRANSFORM"

    common_kwargs = dict(
        split_workers=True,
        batch_size=exp_params['batch_size'],
        transform=transform,
        modality=exp_params['data_modality'],
        huggingface_transform=exp_params['huggingface_transform'],
        augmentation_transform=augmentation_transform,
        invert_color=exp_params['invert_color'],
        n_views=exp_params['num_tiles'],
        grid_dict=grid_dict,
        censor_time=exp_params['censor_time'],
        filter_modality=exp_params['filter_modality'],
        debug=exp_params.get('debug', False),
        grouped=exp_params.get('grouped', False),
        #persistent_workers=worker > 0,

    )

    train_dataset = prepare_PD_dataset(SHARD_PATTERN_train, exclusion_set=exclusion_set,train_df=train_df, exp_params=exp_params, **common_kwargs)
    val_dataset   = prepare_PD_dataset(SHARD_PATTERN_val, exclusion_set=val_exclusion_set,train_df=train_df,exp_params=exp_params, **common_kwargs)
    
    train_loader = DataLoader(
        train_dataset, 
        num_workers=worker, 
        batch_size=None, 
        prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    ) #add collate_fn=lambda x: x,  if you want to bypass thedefault converter (default converter converts numpy to tensors)
    val_loader = DataLoader(
        val_dataset, 
        num_workers=2, 
        batch_size=None, 
        prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=False, #creates a stall when true and variable size batches (eg custom collate)
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, train_dataset, val_dataset
####################################################################################################################

# Functions and datasets for extracting debug information

def debug_image_properties(img_source):
    #compute mean intensity
    img = img_source.convert('L') #convert to grayscale
    arr=np.array(img)
    #i convert to float to avoid the values baing converted to torch tensors via the default_collate in the data loader
    
    #compute ink density
    threshold = 128                                # pixels darker than this = ink

    # cast once, in float64, to avoid uint8 overflow in the squared sum
    arr_f = arr.astype(np.float64) / 255.0   # drop the /255.0 if you want 0-255 stats
    

    img_properties = {
        'format': img_source.format, #img format theimage was loaded from
        'num_channels_original': len(img_source.getbands()), #number of channels in the original image before conversion
        'mode': img_source.mode, #the color mode (e.g., RGB, RGBA, L)
        'size': img.size, #width, height
        'width': img.size[0],
        'height': img.size[1],
        'memory_size_bytes': arr.nbytes,
        #'area': img.size[0] * img.size[1],
        #'ratio': img.size[0] / img.size[1] if img.size[1] != 0 else None, #aspect ratio
        'ink_density': ink_density(arr,threshold), #fraction of pixels that are ink (binary threshold)
        #'sharpness': sharpness(arr), #sharpness of the image
        'is_uniform': is_uniform_image(img_source, tol=3),

        # accumulators for dataset-level mean/std
        'pixel_sum': float(arr_f.sum()),
        'pixel_sq_sum': float((arr_f ** 2).sum()),
        'num_pixels': int(arr_f.size),
    }
    return img_properties


def explore_data(shard_pattern, load_in_memory=False, 
                               split_workers=True, batch_size=4):
    def create_df():
        def flatten_samples(src):
            """Takes an iterator of samples and yields multiple individual images."""
            for sample in src:
                label = None
                subject_id = "unknown"
                
                # Step A: Find the JSON file first to get the label and subject
                for key, value in sample.items():
                    if key.endswith("json"):
                        try:
                            json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                            json_data = json.loads(json_str)
                            label = torch.tensor(json_data.get("label", -1), dtype=torch.long)
                            subject_id = json_data.get("subject", "unknown")
                            shard_name = json_data.get("shard_name", "unknown") 
                        except Exception as e:
                            print(f"Error parsing JSON for sample {key}: {e}")
                        break
                        
                # Step B: Filter at the sample level
                # If the sample is missing a label or has a -1 label, skip the whole sample
                if label is None or label.item() == -1:
                    continue
                    
                # Step C: Process and YIELD images one by one
                for key, value in sample.items():
                    if key.endswith((".png", ".jpg", ".jpeg")):
                        parts = key.split('.')
                        
                        # Expected format: q5.number_random.png
                        if len(parts) == 3:
                            questionnaire = parts[0]
                            modality_type = parts[1]
                            
                            try:
                                if isinstance(value, bytes):
                                    img_source = Image.open(io.BytesIO(value))
                                    #convert to grayscale
                                    img = img_source.convert('L')
                                else:
                                    img = value # Fallback in case it somehow got decoded
                                #compute mean intensity
                                arr=np.array(img)
                                mean_intensity = arr.mean()
                                #compute std
                                std_intensity = arr.std()
                                #compute ink density
                                threshold = 128                                # pixels darker than this = ink
                                ink_pixels = (arr < threshold).sum()
                                ink_density_binary = ink_pixels / arr.size    # fraction of inked pixels

                                img_properties = {
                                    'format': img_source.format, #img format theimage was loaded from
                                    'num_channels_original': len(img_source.getbands()), #number of channels in the original image before conversion
                                    'mode': img_source.mode, #the color mode (e.g., RGB, RGBA, L)
                                    'size': img.size, #width, height
                                    'ratio': img.size[0] / img.size[1] if img.size[1] != 0 else None, #aspect ratio
                                    'mean_intensity': mean_intensity, #mean pixel intensity
                                    'std_intensity': std_intensity, #std of pixel intensity
                                    'ink_density_binary': ink_density_binary, #fraction of pixels that are ink (binary threshold)
                                }

                                #get the filename from the shard_name path
                                shard_filename = os.path.basename(shard_name)

                                yield subject_id, questionnaire, modality_type, label, shard_filename, img_properties
                                
                            except Exception as e:
                                print(f"Skipping corrupted image {key}: {e}")
                                
        return flatten_samples

    
    # 2. File gathering
    shard_files = glob.glob(shard_pattern)
    shard_files.sort()

    if load_in_memory:
        return 0 # Handle your in-memory logic here
        
    # 3. Build the WebDataset Pipeline
    dataset = wds.WebDataset(shard_files, shardshuffle=False)

    if split_workers:
        dataset = dataset.select(wds.split_by_worker)

    # 4. Apply the flattening and batching
    dataset = (dataset
        #.decode(decode_approach)
        .compose(create_df()) # This replaces .map() and .select()
    )
    
    return dataset

#Training samples selection
def melt_df(df,modality,threshold=1, questionnaires_to_include=None):
    if questionnaires_to_include is None:
        questionnaires_to_include = [str(q) for q in range(1,14)]
    exclusion_set = set()
    avail_columns=[f'q_{q}_num_{modality}' for q in questionnaires_to_include]
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


# Class rebalancing / exclusion
def generate_exclusion_set_val(csv_data, data_modality, majority_class_id, balancing_factor, label_col='lateralite',id_col='ident_projet',split='train'):
    train_data = csv_data[csv_data['split'] == split]
    class_counts = train_data[label_col].value_counts()
    print(f"Class distribution in val set (after filtering for modality {data_modality}):\n{class_counts}")
    num_0 = class_counts.get(0.0, 0)  # Count of class 0 (e.g., right-handed)
    num_1 = class_counts.get(1.0, 0)  # Count of class 1 (e.g., left-handed)
    majority_class_ids = train_data[train_data[label_col] == majority_class_id][id_col].unique()
    # randomly select a fraction of those ids based on the balancing factor
    majority_ids_to_include = random.sample(list(majority_class_ids), min(int(num_1*balancing_factor), len(majority_class_ids)))
    exclusion_set = set(majority_class_ids) - set(majority_ids_to_include)
    return exclusion_set

def generate_exclusion_set_PD(csv_source,exp_params,split='train', original_data=None):
    '''
    this funciton computes, for a specific split, all the ids that should be ignored during loading, accounting for all exclusion/inclusion
    conditions.

    If original_data=None we assume csv_source contains all the train/val ids and we 
    remove the ids for which we don't have data and optionally some controls to reduce the case/control ratio

    If original_data=None -> the dataset was pre-filtered externally -> we perform the same exclusion but also exclude
    all ids that were present in the full dataset but not in the pre-filtered one
    '''

    if original_data is None: #for compatibility with old code
        original_data = csv_source.copy()
    
    csv_data = csv_source.copy()
    csv_data = csv_data[csv_data['split'] == split]
    all_original_ids = set(original_data[original_data['split']==split]['unique_id'].unique())

    if exp_params['grouped']: #if grouped modality i want only one case in each group -> have to deal with the cases that are
        #also selected as controls
        #remove all rows with case_control==0 and diag_park_final1_quest==1
        print("Removing all rows with case_control==0 and diag_park_final1_quest==1 to avoid having cases also as controls in grouped modality")
        print(f"Initial number of samples in {split} set: {len(csv_data)}")
        csv_data = csv_data[~((csv_data['case_control']==0) & (csv_data['diag_park_final1_quest']==1))]
        print(f"Number of samples in {split} set after removing cases also as controls: {len(csv_data)}")
        print("-" * 50)

    #exclude the subjects for which grid_pattern or case_grid_pattern has a certain pattern (eg all 0 or 0 before last avail q or ...)
    if exp_params['filter_missing'] == 'all':
        csv_data['last_avail_q'] = 13 #set last avail q to 13 for all subjects to reuse the same code
    print(f"Initial number of samples in {split} set: {len(csv_data)}")
    if 'case_control' in csv_data.columns:
        print(f"Initial number of unique subjects with case_control==1 in {split} set: {csv_data[csv_data['case_control']==1]['unique_id'].nunique()}")
    def prefix_has_one(pattern, n):
        return '1' in pattern[:int(n)]
    mask = csv_data.apply(
        lambda r: prefix_has_one(r['grid_pattern'], r['last_avail_q'])
                and prefix_has_one(r['case_grid_pattern'], r['last_avail_q']),
        axis=1
    )
    #subjects_with_no_pre_avail_q_data = csv_data[~mask]['unique_id'].unique()
    csv_data = csv_data[mask]
    print(f"Number of samples in {split} set after filtering for 000.. string: {len(csv_data)}")
    if 'case_control' in csv_data.columns:
        print(f"Number of unique subjects with case_control==1 in {split} set after filtering for 000.. string: {csv_data[csv_data['case_control']==1]['unique_id'].nunique()}")

    if (split=='train' and exp_params['balanced_data']) or (split=='val' and exp_params['balance_validation']):
        csv_data['group_id'] = csv_data['unique_id'].str.split('_').str[1].astype(int)
        print(f"Balancing the dataset for {split} set with balancing factor {exp_params['balancing_factor']})")
        #keep only the controls
        csv_data_controls = csv_data[csv_data['case_control'] == 0]
        csv_data_cases = csv_data[csv_data['case_control'] == 1]

        #keep balancing factor ids in each group for each case, sampling with priority from the at_least_warning==0 controls
        n_target = int(exp_params['balancing_factor'])
        seed = exp_params['seed']
        def pick(group):
            n = min(len(group), n_target)
            priority = group.loc[group['at_least_warning'] == 0, 'unique_id']
            rest     = group.loc[group['at_least_warning'] != 0, 'unique_id']

            if len(priority) >= n:
                chosen = priority.sample(n, random_state=seed)
            else:
                chosen = pd.concat([
                    priority,  # take all priority ids
                    rest.sample(n - len(priority), random_state=seed)  # fill remainder
                ])
            return chosen
        selected_ids = set(
            csv_data_controls.groupby('group_id', group_keys=False).apply(pick)
        )

        #keep only the selected controls
        csv_data_controls = csv_data_controls[csv_data_controls['unique_id'].isin(selected_ids)]
        #re-join cases and controls
        csv_data = pd.concat([csv_data_cases, csv_data_controls])
    
    ids_to_exclude = all_original_ids - set(csv_data['unique_id'].unique()) #i exclude all the ids that are not still in the df
    
    return ids_to_exclude

def prepare_exclusion_sets_PD(exp_params,verbose=True,class_col='', pre_computed_csv=None):
    '''returns the num_0 and num_1 in the training set after considering the exclusion -> the true class numerosity'''
    exclusion_set = set()
    val_exclusion_set = set()
    
    if pre_computed_csv is not None:
        original_data = pd.read_parquet(exp_params['list_of_ids_paths'])
        csv_data = pre_computed_csv.copy()
    else:
        original_data = pd.read_parquet(exp_params['list_of_ids_paths'])
        csv_data = original_data.copy()
    if verbose:
        print("Initial CSV data loaded. First row example:")
        for col in csv_data.columns:
            print(f"{col}: {csv_data[col].iloc[0]}")
        print('#' * 50)
        print("Unique subjects in the dataset:", csv_data['unique_id'].nunique())
        print("Unique subjects in training set:", csv_data[csv_data['split'] == 'train']['unique_id'].nunique())
        print("Unique subjects in validation set:", csv_data[csv_data['split'] == 'val']['unique_id'].nunique())
        print('#' * 50)
        if class_col in csv_data.columns:
            print("Class distribution in the entire dataset:\n", csv_data[class_col].value_counts())
            print("Class distribution in the training set:\n", csv_data[csv_data['split'] == 'train'][class_col].value_counts())
            print('#' * 50)

    
    exclusion_set = generate_exclusion_set_PD(csv_data,exp_params, split='train',original_data=original_data) 
    val_exclusion_set = generate_exclusion_set_PD(csv_data,exp_params, split='val', original_data=original_data)
    
    if verbose:
        print(len(exclusion_set), "samples will be excluded from the training set to achieve balancing.")
        print('#' * 50)
    
    filtered_csv_data_train = csv_data[~csv_data['unique_id'].isin(exclusion_set)]
    train = filtered_csv_data_train[filtered_csv_data_train['split'] == 'train']
    #compute the number of samples for each class in the training set
    if class_col in train.columns:
        counts = (
            train[class_col]
            .value_counts()
            .reindex(range(train[class_col].max() + 1), fill_value=0)
            .sort_index()
        )
    else:
        counts = [0,0]  # Default counts if class_col is not present (eg when i iterate in debug mode on pre_training dataset)

    if verbose:
        print("After applying the exclusion set, the training set has:")
        print(f"Class 0: {counts[0]} samples")
        print(f"Class 1: {counts[1]} samples")
        print(f"Ratio of Class 1 to Class 0: {counts[1] / counts[0] if counts[0] > 0 else 'undefined'}")

    return exclusion_set, val_exclusion_set, counts

#Merging data from the full dataset 
def merge_properties_from_full_dataset_PD(exp_params, csv_data, properties_to_add, verbose=True):
    if 'full_dataset' not in exp_params:
        raise ValueError("The 'full_dataset' key is missing in exp_params. Specify it to load properties from that dataset.")
    full_dataset_path = exp_params['full_dataset']
    full_df = pd.read_csv(full_dataset_path, encoding='cp1252')

    if csv_data is None:
        csv_data = pd.read_parquet(exp_params['list_of_ids_paths'])
    
    if verbose:
        print("Lenght before merging:", len(csv_data))  
    # Extract ident_projet from "{ident_projet}_XXXX" by stripping the trailing suffix
    csv_data['_ident_projet'] = csv_data['unique_id'].str.rsplit('_', n=1).str[0]

    # Merge in the desired columns
    csv_data = csv_data.merge(
        full_df[['ident_projet'] + properties_to_add],
        left_on='_ident_projet',
        right_on='ident_projet',
        how='left'
    )

    # Clean up helper columns
    csv_data = csv_data.drop(columns=['_ident_projet'])
    if 'ident_projet' not in csv_data.columns:  # only drop if merge created a duplicate
        pass
    else:
        csv_data = csv_data.drop(columns=['ident_projet'])
    
    if verbose:
        print("Lenght after merging:", len(csv_data))  
        print("#" * 50)
    
    return csv_data

#Preparing synthetic data dataframe
def synthetic_data_override(exp_params, verbose=True):
    '''this function generate a new training set with the synthetic classes, save it to file and override the id_list_path
    -> the other functions go read that'''
    if exp_params['synthetic'] is None:
        return exp_params
    train_data = pd.read_parquet(exp_params['list_of_ids_paths'])
    train_data['synth_label'] = 0
    synthetic_modes = exp_params['synthetic']
    synthetic_proportions = exp_params['synthetic_proportions']
    num_synthetic_modes = len(synthetic_modes)
    #assign randomly the synthetic labels to the training set based on the proportions
    #shuffle the train_data
    rng = np.random.default_rng(exp_params['seed'])          # seed for reproducibility
    train_data['synth_label'] = rng.choice(num_synthetic_modes, size=len(train_data), p=synthetic_proportions)
    synthetic_path = exp_params['list_of_ids_paths'].replace('.parquet', '_synthetic.parquet')
    train_data.to_parquet(synthetic_path, index=False)
    if verbose:
        print(f"Synthetic data generated and saved to {synthetic_path} !!!!!!!!!!!!")
        print("Now overriding experiment settings...  ")
    exp_params['list_of_ids_paths'] = synthetic_path
    exp_params['num_classes'] = num_synthetic_modes
    exp_params['class_col'] = 'synth_label'
    exp_params['filter_missing'] = 'all'

    return exp_params


#Loading the grid_file data
def pre_load_grid_data(h5_filepath,csv_data):
    '''
    this function takes a csv with the samples to consider and for each 
    saves the grid file in a csv (load grids for all modalities)
    '''
    import time
    start = time.time()
    unique_subjects = csv_data['ident_projet'].unique()
    
    full_dict=get_ids_data_from_h5_file_list(h5_filepath, unique_subjects)
        
    end = time.time()
    print(f"Preloaded grid data for {len(unique_subjects)} subjects in {end - start:.2f} seconds.")
    #get the memory used by the full_dict in MBs
    total_size = sys.getsizeof(full_dict) / (1024 * 1024)
    print(f"Total memory used by preloaded grid data: {total_size:.2f} MB")
    return full_dict
def get_ids_data_from_h5_file_list(file_path, target_ids):
    """
    Retrieves data for a list of IDs in a single file open.
    Returns: {id: {q_name: {class_key: (scalar, array)}}}
    IDs not found in the file are omitted (and reported).
    """
    target_ids = [str(i) for i in target_ids]
    results = {}

    with h5py.File(file_path, 'r') as f:
        for i,target_id in enumerate(target_ids):
            if target_id not in f:
                print(f"ID {target_id} not found in {file_path}")
                continue

            id_grp = f[target_id]
            id_data = {}

            for q_name in id_grp.keys():
                q_grp = id_grp[q_name]
                id_data[q_name] = {
                    class_key: (
                        q_grp[class_key].attrs.get('scalar_value'),
                        q_grp[class_key][()],          # [()] reads the full array
                    )
                    for class_key in q_grp.keys()
                }

            results[target_id] = id_data
            if i%100 == 0:
                print(f"Processed {i+1}/{len(target_ids)} IDs from {file_path}", flush=True)
                total_size = sys.getsizeof(results) / (1024 * 1024)
                print(f"Total memory used until now: {total_size:.2f} MB", flush=True)

    return results
def load_grid_dict_old(exp_params):
    if exp_params['use_grid']:
        """Load the grid dictionary from a pickle file."""
        with open(exp_params['grid_dict_path'], "rb") as f:
            grid_dict = pickle.load(f)
        print("Grid dictionary loaded successfully. Number of entries:", len(grid_dict))
        return grid_dict
    else:
        print("Grid usage is disabled. No grid dictionary will be loaded.")
        return None
def load_grid_dict(exp_params):
    if exp_params['use_grid']:
        base = exp_params['grid_dict_path']  # wherever you saved step 1
        packed = {
            "data":          np.load(os.path.join(base, "grid_data.npy"), mmap_mode='r'),
            "offsets":       np.load(os.path.join(base, "grid_offsets.npy")),
            "scalars":       np.load(os.path.join(base, "grid_scalars.npy")),
            "entry_of_code": np.load(os.path.join(base, "grid_entry_of_code.npy"), mmap_mode='r'),
        }
        with open(os.path.join(base, "grid_vocabs.pkl"), "rb") as f:
            packed["vocabs"] = pickle.load(f)
        return packed
    else:
        print("Grid usage is disabled. No grid dictionary will be loaded.")
        return None
def grid_lookup(packed, id_, q_name, class_key):
    v = packed["vocabs"]
    code = (v["id"][id_] * v["NQ"] + v["q"][q_name]) * v["NC"] + v["c"][class_key]
    i = packed["entry_of_code"][code]
    if i < 0:
        raise KeyError((id_, q_name, class_key))
    lo, hi = packed["offsets"][i], packed["offsets"][i + 1]
    return packed["scalars"][i], packed["data"][:, lo:hi]

#preparing the pre-training dataset (csv)
def prepare_pre_training(df, data_selected_source):
    data_selected = data_selected_source.copy()
    def combine_avail_columns(df):
        cols = [f"q_{i}_avail" for i in range(1, 14)]
        df['avail_pattern'] = df[cols].astype(int).astype(str).agg(''.join, axis=1)
        df = df.drop(columns=cols)
        return df
    def combine_grid_columns(df):
        cols = [f"q_{i}_grid_file_avail" for i in range(1, 14)]
        df['grid_pattern'] = df[cols].astype(int).astype(str).agg(''.join, axis=1)
        df = df.drop(columns=cols)
        return df
    df = combine_avail_columns(df)
    df = combine_grid_columns(df)
    df = df[['grid_pattern', 'avail_pattern', 'ident_projet', 'split','rempli_seulq11','rempli_seulq12']]

    #create a ident_projet column from the unique_id column of data_selected; unique_id is in the form XXXX_YYYYY with XXXX the ident_projet
    data_selected['ident_projet'] = data_selected['unique_id'].str.split('_').str[0]
    #get the unique ident_projet from data_selected
    unique_ident_projet = data_selected['ident_projet'].unique()
    #exclude the rows in df that are in unique_ident_projet
    df = df[~df['ident_projet'].isin(unique_ident_projet)]
    #change the name of ident_projet to unique_id in df
    df = df.rename(columns={'ident_projet': 'unique_id'})

    return df


# Path selection
def return_file_paths(problem,grouped,pre_training):
    if problem == 'PD' and not grouped and not pre_training:
        list_of_ids_paths = "/home/a_morelli/datasets/id_lists/PD_training_set_20_07_26.parquet" 
        #"/home/a_morelli/datasets/id_lists/PD_training_set_13_7_26.parquet"
        data_folder = '/mnt/beegfs02/scratch/a_morelli/model_training/shards/PD/final_png_whitebg_21_07_26'
        #"/mnt/beegfs02/scratch/a_morelli/model_training/PD/final_png_whitebg"
        #"/home/a_morelli/datasets/shards/PD/final_png_whitebg_21_07_26"
        grid_dict_path = "/home/a_morelli/datasets/id_lists/h5/PD_data_h5.pkl"
    elif problem == 'PD' and pre_training:
        list_of_ids_paths = "/home/a_morelli/datasets/id_lists/final_table_for_matching_splitted_13_7_26_pre_training.parquet"
        data_folder = "/mnt/beegfs02/scratch/a_morelli/model_training/shards/PDpretraining/final_png_whitebg_21_07_26"
        grid_dict_path = "/home/a_morelli/datasets/id_lists/h5/pre_training_data_h5_21_07_26"
    return list_of_ids_paths, data_folder, grid_dict_path