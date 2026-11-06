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
import zlib
import warnings

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
 
# How to map the transform's output range to [-1, 1] with ink = +1.
_RANGE_CONVERSIONS = {
    "ink_pm1": lambda x: x,                # already [-1, 1], ink = +1: no change
    "paper_pm1": lambda x: -x,             # [-1, 1] but paper = +1 (white = 1)
    "paper_unit": lambda x: 1.0 - 2.0 * x,  # [0, 1], white paper = 1, dark ink = 0
    "ink_unit": lambda x: 2.0 * x - 1.0,   # [0, 1], ink = 1 (inverted images)
}
 
_checked = False  # range/polarity sanity check runs once per process (per worker)

def _maybe_aggregate(samples, field_name):
    idx, agg = OPTIONAL_FIELD_SPECS[field_name]
    if all(len(s) > idx and s[idx] is not None for s in samples):
        return agg([s[idx] for s in samples])
    return None

def _stable_seed(key: str, base_seed: int = 0) -> int:
    # Python's hash() is randomised per interpreter, so it can differ between runs
    # (and between workers with the 'spawn' start method). crc32 is stable everywhere.
    return (base_seed * 1_000_003 + zlib.crc32(key.encode())) % (2**63)

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

    return images, quests, mods

# MARK: Dataset for time series stream
def create_sequence_flattener_PD_multiview(transform_func, augmentation_transform_list, n_views,
                                           modality_string_list, original_modality_names,
                                           exclusion_set, exp_params=None,huggingface_transform=False,
                                           invert_color=False, grid_dict=None, censor_time='pre_diagnosis',
                                           filter_modality='number_random', debug=False, train_df=None, is_val=False):
    #get values from exp_params if not provided
    unsupervised_pretrain=exp_params.get('unsupervised_pretrain', False) if exp_params else False
    unsup_kwarks=exp_params.get('unsupervised_kwargs', {'n_im_per_subj': 10, 'replace':True}) if exp_params else {}
    #print(f"[DEBUG] unsupervised_pretrain: {unsupervised_pretrain}, unsup_kwarks: {unsup_kwarks}", flush=True) #REMOVE

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

            if label.item() == -1 and not unsupervised_pretrain:
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
                replace = unsup_kwarks.get('replace', True)
                gen = (torch.Generator().manual_seed(_stable_seed(subject_id, unsup_kwarks.get('val_seed', 0)))
                    if is_val else None)   # training stays random
                sequence, questionnaires, modalities = _get_unsupervised_batch(sequence, questionnaires, modalities, 
                                                                               n_im_per_subj, generator=gen,replace=replace)
                #print(sequence.shape, questionnaires, modalities, "unsupervised batch", flush=True) #[REMOVE]
                yield sequence, questionnaires, modalities
            else:
                yield sequence, label, subject_id, questionnaires, modalities, resized_list, global_properties, local_properties, view_mask 

    return flatten_samples

def collate_variable_sequences_PD(samples, debug=False, **kwargs):
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

# MARK: Unsupervised training
def _to_single_channel(images: torch.Tensor) -> torch.Tensor:
    c = images.shape[1]
    if c == 1:
        return images
    if c == 3:  # e.g. RGB output of a Hugging Face transform: average to grayscale
        return images.mean(dim=1, keepdim=True)
    raise ValueError(f"expected 1 or 3 channels, got images of shape {tuple(images.shape)}")
 
 
def _sanity_check(x: torch.Tensor) -> None:
    """Warn (once) if values fall outside [-1, 1] or polarity looks inverted.
 
    Handwriting is mostly paper, so the median pixel should be background, i.e.
    close to -1. A positive median almost always means ink/paper are swapped.
    """
    global _checked
    if _checked:
        return
    _checked = True
    lo, hi, med = x.min().item(), x.max().item(), x.median().item()
    if lo < -1.01 or hi > 1.01:
        warnings.warn(f"collate_unsupervised: image values span [{lo:.2f}, {hi:.2f}], expected [-1, 1]. "
                      f"Check `input_range` or the transform's normalisation.")
    if med > 0:
        warnings.warn(f"collate_unsupervised: median pixel = {med:.2f} > 0, so the background looks like "
                      f"ink. Polarity is probably inverted (expected paper = -1, ink = +1); "
                      f"check `input_range` and `invert_color`.")
 
def collate_unsupervised(samples, debug: bool = False, **kwargs) -> dict:
    mod_to_id = kwargs.get('mod_to_id', None)
    input_range = kwargs.get('input_range', "ink_pm1")
    check = kwargs.get('check', True)

    if len(samples) == 0:
        raise ValueError("collate_unsupervised received an empty batch")
    if input_range not in _RANGE_CONVERSIONS:
        raise ValueError(f"input_range must be one of {list(_RANGE_CONVERSIONS)}, got {input_range!r}")
 
    images, quests, mods, subjects = [], [], [], []
    for i, s in enumerate(samples):
        imgs, q, m = s[0], s[1], s[2]
        if imgs.ndim != 4:
            raise ValueError(f"sample {i}: expected images (n, C, H, W), got {tuple(imgs.shape)}")
        n = imgs.shape[0]
        if len(q) != n or len(m) != n:
            raise ValueError(f"sample {i}: {n} images but {len(q)} questionnaires / {len(m)} modalities")
        images.append(imgs)
        quests.extend(q)
        mods.extend(m)
        if len(s) > 3:
            subjects.extend([s[3]] * n)
 
    shapes = {tuple(t.shape[1:]) for t in images}
    if len(shapes) > 1:
        raise ValueError(f"cannot concatenate images of different sizes in one batch: {sorted(shapes)}")
 
    x = torch.cat(images).float()
    x = _to_single_channel(x)
    x = _RANGE_CONVERSIONS[input_range](x)
    if check:
        _sanity_check(x)
    x = x.clamp(-1.0, 1.0)
 
    unknown = sorted({str(m) for m in mods if m not in mod_to_id})
    if unknown:
        raise KeyError(f"modality names not in mod_to_id: {unknown}. Known: {list(mod_to_id)}")
    modality = torch.tensor([mod_to_id[m] for m in mods], dtype=torch.long)
 
    try:
        questionnaire = torch.tensor([int(q) for q in quests], dtype=torch.long)
    except (TypeError, ValueError):
        questionnaire = list(quests)
 
    batch = {"image": x.contiguous(), "modality": modality, "questionnaire": questionnaire}
    if subjects:
        batch["subject_id"] = subjects
 
    if debug:
        print(f"[collate_unsupervised] {len(samples)} subjects -> image {tuple(x.shape)}, "
              f"range [{x.min():.2f}, {x.max():.2f}], modality counts {torch.bincount(modality, minlength=3).tolist()}",
              flush=True)
    return batch

# MARK: build datasets and dataloaders
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
        return collate_unsupervised
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
    if shuffle_buffer and not is_val:  # Only shuffle training data, not validation
        dataset = dataset.shuffle(shuffle_buffer) #shuffle the samples

    if not grid_dict: 
        raise ValueError("grid_dict must be provided for PD dataset preparation.")
    if modality_string != 'all' and not isinstance(modality, list) and n_views >= 1:
        raise ValueError("Not implemented anymore, see version before 15/10/26")
    elif isinstance(modality, list):
        compose_fn = _get_compose(grouped=grouped, unsupervised_pretrain=unsupervised_pretrain)
        collate_fn = _get_collate(grouped=grouped, unsupervised_pretrain=unsupervised_pretrain)
        collate_kwargs = dict(mod_to_id=exp_params.get('mod_to_id', None), 
                              input_range=exp_params.get('unsupervised_kwargs', {}).get('input_range', "ink_pm1"), 
                              check=True)

        dataset = (dataset
            #.decode(decode_approach)
            .compose(compose_fn(transform, augmentation_transform, n_views=n_views, modality_string_list=modality_string,
                                exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color,
                                grid_dict=grid_dict, censor_time=censor_time,
                                original_modality_names=modality, filter_modality=filter_modality, debug=debug, train_df=train_df,
                                exp_params=exp_params, is_val=is_val)) # This replaces .map() and .select()
            .batched(batch_size,
                    collation_fn=partial(collate_fn, debug=debug, **collate_kwargs),
                    partial=partial_batch)
        )
        
    
    return dataset


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





