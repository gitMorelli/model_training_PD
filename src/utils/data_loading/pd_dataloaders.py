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
import glob
from typing import NamedTuple, Optional, List

from src.utils.image_processing import get_augmentation_transform, ink_density, sharpness, is_uniform_image, SyntheticTransform, place_patches, get_grid_sample
from src.utils.image_processing import debug_image_properties
from src.utils.handwriting_features import extract_image_properties
from src.utils.image_processing import augment_frame

from src.utils.data_loading.grid_loading import grid_lookup
from src.utils.data_loading.pd_shared_helpers import _make_subject_sequence_builder, _parse_sample_json, _index_images,_collate_subject_level
from src.utils.data_loading.grouped.datasets import collate_groups_PD, create_sequence_group_flattener_PD_multiview


OPTIONAL_FIELD_SPECS = {
    "global_properties": (6, lambda vals: torch.stack(vals, dim=0)),   # (N, m1)
    "local_properties":  (7, lambda vals: torch.cat(vals, dim=0)),     # (sum T_i, m2)
    "view_mask":         (8, lambda vals: torch.cat(vals, dim=0)),   # (sum T_i, k)
}
modalities_map = {
        'X': 'X',
        'text': 'hand',
        'digit': 'number_random',
        'digitOrdered': 'number',
        'sent': 'hand_sentences_full'
    }

def _maybe_aggregate(samples, field_name):
    idx, agg = OPTIONAL_FIELD_SPECS[field_name]
    if all(len(s) > idx and s[idx] is not None for s in samples):
        return agg([s[idx] for s in samples])
    return None

def _get_unsupervised_batch(sequence, questionnaires, modalities, n,replace=False, generator=None):
    """
    sequence:       tensor (T, n_views, C, H, W)
    questionnaires: list of length T
    modalities:     list of T lists, each of length n_views
    n:              number of (C, H, W) images to sample

    Returns:
        images:  tensor (n, C, H, W)
        mods:    list of n modality names
        quests:  list of n questionnaire numbers
    """
    T, V = sequence.shape[:2]
    total = T * V

    if replace:
        flat_idx = torch.randint(total, (n,), generator=generator)
    else:
        if n > total:
            raise ValueError(f"Cannot sample {n} images without replacement from {total}")
        flat_idx = torch.randperm(total, generator=generator)[:n]

    t_idx = flat_idx // V   # time step
    v_idx = flat_idx % V    # view within that step

    images = sequence[t_idx.to(sequence.device), v_idx.to(sequence.device)]  # (n, C, H, W)

    t_list, v_list = t_idx.tolist(), v_idx.tolist()
    mods = [modalities[t][v] for t, v in zip(t_list, v_list)]
    quests = [questionnaires[t] for t in t_list]

    return images, mods, quests

# MARK: Dataset for time series stream
def create_sequence_flattener_PD_multiview(transform_func, augmentation_transform_list, n_views,
                                           modality_string_list, original_modality_names,
                                           exclusion_set, exp_params=None,huggingface_transform=False,
                                           invert_color=False, grid_dict=None, censor_time='pre_diagnosis',
                                           filter_modality='number_random', debug=False, train_df=None, is_val=False):
    #get values from exp_params if not provided
    unsupervised_pretrain=exp_params.get('unsupervised_pretrain', False) if exp_params else False
    unsup_kwarks=exp_params.get('unsupervised_kwargs', {'n_im_per_subj': 10}) if exp_params else {}

    #check if you have a column called 'synth_label' in the train_df
    if train_df is not None and 'synth_label' in train_df.columns:
        is_synthetic = True
    else: 
        is_synthetic = False
    #print(is_synthetic, "is_synthetic", flush=True)
    
    build_sequence = _make_subject_sequence_builder(
        transform_func, augmentation_transform_list, modality_string_list,
        original_modality_names, grid_dict, censor_time, filter_modality,
        huggingface_transform, invert_color, debug, train_df, exp_params=exp_params, is_synthetic=is_synthetic,
        is_val=is_val)

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
            elif exp_params['problem']=='handedness':
                #get the label from the train_df
                handedness_label = train_df.loc[train_df['unique_id']==subject_id,exp_params['class_col']].values
                label = torch.tensor(handedness_label[0], dtype=torch.long) if len(handedness_label)>0 else torch.tensor(-1, dtype=torch.long)
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

            sequence, questionnaires, modalities, resized_list, global_properties, local_properties, view_mask  = result

            if unsupervised_pretrain:
                n_im_per_subj = unsup_kwarks.get('n_im_per_subj', 10)
                sequence, questionnaires, modalities = _get_unsupervised_batch(sequence, questionnaires, modalities, n_im_per_subj, generator=None)
                yield sequence, questionnaires, modalities
            else:
                yield sequence, label, subject_id, questionnaires, modalities, resized_list, global_properties, local_properties, view_mask 

    return flatten_samples

def collate_variable_sequences_PD(samples, debug=False):
    sequences       = [s[0] for s in samples]
    labels          = torch.stack([s[1] for s in samples])
    subject_ids     = [s[2] for s in samples]
    questionnaires  = [s[3] for s in samples]
    modalities_list = [s[4] for s in samples]
    resized_list    = [s[5] for s in samples]

    global_properties = _maybe_aggregate(samples, "global_properties") #if global_properties is None for any sample, it will return None
    local_properties  = _maybe_aggregate(samples, "local_properties")
    view_mask         = _maybe_aggregate(samples, "view_mask") #if view_mask is None for any sample, it will return None

    return _collate_subject_level(sequences, labels, subject_ids, questionnaires,
                                   modalities_list, resized_list, debug=debug,
                                   global_properties=global_properties,
                                   local_properties=local_properties, view_mask=view_mask)

def _get_compose(grouped=False, unsupervised_pretrain=False):
    if grouped:
        return create_sequence_group_flattener_PD_multiview
    elif unsupervised_pretrain:
        return create_sequence_flattener_PD_multiview
    else:
        return create_sequence_flattener_PD_multiview

def _get_collate(grouped=False, unsupervised_pretrain=False):
    if grouped:
        return collate_groups_PD
    elif unsupervised_pretrain:
        return lambda samples, debug=False: samples  # No collation for unsupervised pretraining
    else:
        return collate_variable_sequences_PD

def prepare_PD_dataset(shard_pattern, split_workers=True, batch_size=4, transform=None, exclusion_set=set(), modality='X',
                       huggingface_transform=False,augmentation_transform=None, invert_color=False,n_views=1, grid_dict = None,
                       censor_time='pre_diagnosis', filter_modality='digit', debug=False, grouped=False, train_df=None, exp_params=None, 
                       partial_batch=False, is_val=False):
    '''
    if n_views is fractional i sample a fraction n_view of the patches for each image; else i use the same n_view for all iamges
    '''
    #get args from exp_params if not provided
    unsupervised_pretrain = exp_params.get('unsupervised_pretrain', False) 
    
    if isinstance(modality, list):
        modality_string = [m.split('_')[0] for m in modality]  # Normalize to base modality names (keep original name for debugging)
        modality_string = [modalities_map[m] for m in modality_string if m in modalities_map]
    elif modality in modalities_map:
        modality_string = modalities_map[modality]
    else:
        modality_string = 'all'
    filter_modality = modalities_map.get(filter_modality)  # Normalize filter modality

    # 2. File gathering
    if isinstance(shard_pattern, str):
        shard_files = glob.glob(shard_pattern)
        shard_files.sort()
    else:
        shard_files = shard_pattern  # assume it's already a list of files
        
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

    if not grid_dict: 
        raise ValueError("grid_dict must be provided for PD dataset preparation.")
    if modality_string != 'all' and not isinstance(modality, list) and n_views >= 1:
        raise ValueError("Not implemented anymore, see version before 15/10/26")
    elif isinstance(modality, list):
        compose_fn = _get_compose(grouped=grouped, unsupervised_pretrain=unsupervised_pretrain)
        collate_fn = _get_collate(grouped=grouped, unsupervised_pretrain=unsupervised_pretrain)

        dataset = (dataset
            #.decode(decode_approach)
            .compose(compose_fn(transform, augmentation_transform, n_views=n_views, modality_string_list=modality_string,
                                exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color,
                                grid_dict=grid_dict, censor_time=censor_time,
                                original_modality_names=modality, filter_modality=filter_modality, debug=debug, train_df=train_df,
                                exp_params=exp_params, is_val=is_val)) # This replaces .map() and .select()
            .batched(batch_size,
                    collation_fn=partial(collate_fn, debug=debug),
                    partial=partial_batch)
        )
        
    
    return dataset

# MARK: Dataset for unsupervised pre-training


# MARK: - Prepare loaders
def prepare_loaders_PD(worker,prefetch_factor,exp_params,exclusion_set,val_exclusion_set, grid_dict,transform, 
                       SHARD_PATTERN_train, SHARD_PATTERN_val, train_df=None, cross_val=False, one_only=False, persistent_workers=True,
                       running_eval=False, partial_batch=False, merge_shards=False):
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

    if cross_val or merge_shards:
        print("[DEBUG] Merging train and val shards, cross_val:", cross_val, "merge_shards:", merge_shards)
        train_shards = sorted(glob.glob(SHARD_PATTERN_train))
        val_shards   = sorted(glob.glob(SHARD_PATTERN_val))
        all_shards   = train_shards + val_shards
        train_input = all_shards
        val_input   = all_shards
    else:
        train_input = SHARD_PATTERN_train
        val_input   = SHARD_PATTERN_val

    train_dataset = prepare_PD_dataset(train_input, exclusion_set=exclusion_set,train_df=train_df, exp_params=exp_params, 
                                       partial_batch=partial_batch, **common_kwargs) 
    train_loader = DataLoader(
        train_dataset, 
        num_workers=worker, 
        batch_size=None, 
        prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers
    ) #add collate_fn=lambda x: x,  if you want to bypass thedefault converter (default converter converts numpy to tensors)


    if not one_only: #if i use this funtion for building a single dataloader for training, i don't need to build the validation dataloader
        val_dataset   = prepare_PD_dataset(val_input, exclusion_set=val_exclusion_set,train_df=train_df,exp_params=exp_params, 
                                        partial_batch=True, is_val=True, **common_kwargs)
        
        if running_eval:
            val_workers = worker
        else:
            val_workers = worker//2
            prefetch_factor = min(2, prefetch_factor) if val_workers > 0 else None

        val_loader = DataLoader(
            val_dataset, 
            num_workers=val_workers, 
            batch_size=None, 
            prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
            pin_memory=False, #creates a stall when true and variable size batches (eg custom collate)
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers
        )
    else:
        val_loader = None
        val_dataset = None
    return train_loader, val_loader, train_dataset, val_dataset





