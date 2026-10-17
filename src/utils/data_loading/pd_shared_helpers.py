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
import io

from src.utils.image_processing import get_augmentation_transform, ink_density, sharpness, is_uniform_image, SyntheticTransform, place_patches, get_grid_sample
from src.utils.image_processing import debug_image_properties
from src.utils.handwriting_features import extract_image_properties
from src.utils.image_processing import augment_frame

from src.utils.data_loading.grid_loading import grid_lookup



# Globals
ALL_MODALITIES = ['hand', 'number_random', 'X']

window_size_dict = {
        'hand': 100,
        'number_random': 60,
        'X': 60
    } #average of the median size of patches in the modality

window_n_elements_window_dict = {
    'hand': 9,
    'number_random': 9,
    'X': 16
}

# MARK: - Shared helpers
# ---------------------------------------------------------------------------

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

def _decode_image(raw_image_data, num):
    imputed = False
    if isinstance(raw_image_data, bytes):
        img = Image.open(io.BytesIO(raw_image_data)).convert('RGB')
    elif num <= 0 or raw_image_data is None:
        # modality not present -> impute a blank image
        img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        imputed = True
    else:
        img = raw_image_data
    return img, imputed

def _update_transforms(selected_transforms, selected_modality_names, debug, feature_extraction):
    if debug and feature_extraction==False:
        #if feature_extraction is true i don't want the original modality
        selected_transforms = [('original',None)] + selected_transforms
        selected_modality_names = ([selected_modality_names[0].split('_')[0] + '_original']
                                    + selected_modality_names)
    elif debug and feature_extraction==True: #for feature extraction i want only the original modality, no augmentation
        selected_transforms = [('original',None)] 
        selected_modality_names = ([selected_modality_names[0].split('_')[0] + '_original'])
    return selected_transforms, selected_modality_names

def _set_coords(num, grid, rescale_factor, img):
    if num>0:
        x_coords = [0] + sorted(list(grid[0, :] * rescale_factor[0])) + [img.width]
        n_x = len(x_coords) - 1
        y_coords = [0] + sorted(list(grid[1, :] * rescale_factor[1])) + [img.height]
    else:
        x_coords = [0, img.width]
        n_x = 1
        y_coords = [0, img.height]
    return x_coords, n_x, y_coords

def _initialize_sampling(num, current_mode, selected_transforms, is_val):
    #if i have a window modality i need to prepare some data
    count_windows = 0
    sampled_windows = []
    n_elements_window = window_n_elements_window_dict.get(current_mode, 9)
    sample_types = [t[0] for t in selected_transforms]
    has_any = any(t in sample_types for t in ['window', 'patch'])
    has_windows = any(t in sample_types for t in ['window'])
    has_patches = any(t in sample_types for t in ['patch'])
    if has_windows and has_patches:
        raise ValueError("Cannot have both window and patch modalities in the same set of transforms.")
    
    if num>0 and has_any:
        #count the number of window views you have in selected_transforms
        if has_windows:
            n_window_views = sum(1 for t in selected_transforms if t[0] == 'window')
        elif has_patches:
            n_window_views = sum(1 for t in selected_transforms if t[0] == 'patch')
            n_elements_window = 1 #for patch modality i only need one patch per view

        indices = [i for i in range(1, num + 1)]
        if is_val:
            sampled_windows_no_rep = indices[:] #if i am in validation -> no randomicity
        else:
            sampled_windows_no_rep = random.sample(indices, len(indices))
        
        remaining = n_elements_window * n_window_views - len(sampled_windows_no_rep)
        if remaining > 0:
            if is_val:
                sampled_windows = sampled_windows_no_rep[:]
                for i in range(remaining):
                    sampled_windows.append(indices[i % len(indices)])
            else:
                sampled_windows = sampled_windows_no_rep + random.choices(indices, k=remaining)
                sampled_windows = random.sample(sampled_windows, len(sampled_windows))
        else:
            sampled_windows = sampled_windows_no_rep[:n_elements_window * n_window_views]

    return count_windows, n_elements_window, sampled_windows

def get_properties(sid,train_df,params, q_to_keep):
    """Get the properties for a specific step of a subject from the training DataFrame.
    if selected_properties is None -> we don't use covariates in trainig
    if only per_step or global is None -> the corresponding property list will be empty

    local properties is a tensor of shape T_i,n (T_i number of questionnaires for subject i, n number of properties extracted per step)
    global  properties is a tensor of shape m (m number of properties extracted globally for the subject)
    """
    selected_properties={}
    properties = params.get('selected_properties', None)
    if not properties:   # True for None, [], (), '', etc.
        return None, None
    
    selected_properties['per_step'] = [prop for prop in properties if '*' in prop]
    selected_properties['global'] = [prop for prop in properties if '*' not in prop]

    global_properties = []
    local_properties = []
    row = train_df[train_df['unique_id'] == sid].iloc[0]
    all_cols = train_df.columns.tolist()

    if selected_properties['global'] is not None:
        for prop in selected_properties['global']: 
            #select all the properties that are in the columns of the dataframe and start with '_'+prop
            prefix = '_' + prop
            result = [s for s in all_cols if s.startswith(prefix)]
            for r in result:
                global_properties.append(row[r])
            '''
            note available global properties
            Column: etudegp
            Column: profq2
            Column: lateralite
            Column: relative_age
            '''
    if selected_properties['per_step'] is not None:
        for q in q_to_keep:
            step_properties = []
            for prop in selected_properties['per_step']:
                #substitute the '*' with q in prop
                prefix = '_' + prop.replace('*', str(q)) + '_'
                result = [s for s in all_cols if s.startswith(prefix)]
                for r in result:
                    step_properties.append(row[r])
            local_properties.append(torch.tensor(step_properties, dtype=torch.float32))
            '''
            note available per-step properties
            Column: case_dt_dateq* -> case_dt_dateq1, case_dt_dateq2, ..
            ''' 
    #transform the properties to torch tensors
    global_properties = torch.tensor(global_properties, dtype=torch.float32) if global_properties else torch.tensor([], dtype=torch.float32)
    local_properties = torch.stack(local_properties) if local_properties else torch.tensor([], dtype=torch.float32)
    return global_properties, local_properties

def _augment(rng, pixel_space_augmentations):
    sigma = 0.0
    #print("----------------> Augmenting")
    img_view, sigma = augment_frame(img_view, rng, pixel_space_augmentations)

    '''orig = img_view.copy()
    aug, sigma = augment_frame(orig, rng, pixel_space_augmentations)

    a = np.asarray(orig.convert("L"), dtype=np.int16)
    b = np.asarray(aug.convert("L"), dtype=np.int16)
    print("mean abs diff:", np.abs(a - b).mean(), " max:", np.abs(a - b).max())
    print("-"*50)'''
    #sigma has no effect, i should add it as gaussian noise after to_tensor and before normalization
    '''
    t = TF.to_tensor(f) 
    if sigma:
        t = (t + torch.randn_like(t) * sigma).clamp(0, 1)
    out.append(TF.normalize(t, self.mean, self.std))
    ''' 
    return img_view

def _append_new_views(img_view, img_tensor, debug, feature_extraction, augmentation_transform, selected_transforms, 
                      imputed, k, selected_modality_names, list_of_views, x_coords=None, y_coords=None):
    if debug and feature_extraction==False:
        if augmentation_transform[0] == 'original':
            metadata = debug_image_properties(img_view)
        else:
            metadata = debug_image_properties(img_tensor)
        metadata['selected_transform'] = selected_transforms[k]
        metadata['imputed'] = imputed
        list_of_views.append(metadata)
    elif debug and feature_extraction:
        metadata = extract_image_properties(img_view,
                    x_coords,
                    y_coords,
                    threshold=128,
                    ink_is_bright=True,
                    use_otsu=False,
                    min_ink_pixels=12,
                    min_component_area=8,
                    compute_slant=True,
                    trend_keys=("bbox_height", "stroke_width_mean"),  #"ink_area", "fill_ratio"
                    reductions=("mean", "std"))
        list_of_views.append(metadata)
    elif '_diff' in selected_modality_names[k]:
        orig_tensor=list_of_views[-1]  #the previous one is the original
        augment_tensor=list_of_views[-2]  #the one before is the augmented
        diff_tensor = torch.abs(orig_tensor - augment_tensor)
        list_of_views.append(diff_tensor)
    else:
        list_of_views.append(img_tensor)
    return list_of_views

# MARK: Sampling functions
def _grid_sampling(x_coords, y_coords, img, augmentation_transform, n_x, num):
    if num > 0:
            rand_num = random.randint(1, num)
            img_view = get_grid_sample(x_coords, y_coords, img, rand_num, n_x)
    else:
        img_view = img.copy()
    img_view = augmentation_transform[1](img_view)  # apply the callable transform
    return img_view
def _window_sampling(x_coords, y_coords, img, augmentation_transform, n_x, imputed, current_mode, selected_modality_names, 
                     sampled_windows, count_windows, n_elements_window, k):
    # scale the size based on the number of elements in the window (if 9 -> 3xexpected size of a chunk)
    size = int(window_size_dict.get(current_mode, 60)*np.sqrt(n_elements_window))  
    #create a sizexsizex3 blank image
    img_view = Image.new('RGB', (size, size), color=(255, 255, 255))
    if imputed:
        return img_view, count_windows
    if '_noaug' in selected_modality_names[k] or '_diff' in selected_modality_names[k]:
        count_windows -= 1 #i should use the same index as that of the previous augmented slide
    this_window_indexes = sampled_windows[(count_windows) * n_elements_window: (count_windows+1) * n_elements_window]
    count_windows += 1
    patches=[]
    for idx in this_window_indexes:
        patch = get_grid_sample(x_coords, y_coords, img, idx, n_x)
        patches.append(patch) 
    img_view = place_patches(patches, size, 255)
    img_view = augmentation_transform[1](img_view)  # apply the callable transform
    return img_view, count_windows
def _patch_sampling(x_coords, y_coords, img, augmentation_transform, n_x, imputed, selected_modality_names, 
                     sampled_windows, count_windows, k):
    size = 32
    #create a sizexsizex3 blank image
    img_view = Image.new('RGB', (size, size), color=(255, 255, 255))
    if imputed:
        return img_view, count_windows
    if '_noaug' in selected_modality_names[k] or '_diff' in selected_modality_names[k]:
        count_windows -= 1 #i should use the same index as that of the previous augmented slide
    this_patch_index = sampled_windows[count_windows]
    count_windows += 1
    img_view = get_grid_sample(x_coords, y_coords, img, this_patch_index, n_x)
    img_view = augmentation_transform[1](img_view)  # apply the callable transform
    return img_view, count_windows

def _make_subject_sequence_builder(transform_func, augmentation_transform_list,
                                   modality_string_list, original_modality_names,
                                   grid_dict, censor_time, filter_modality,
                                   huggingface_transform, invert_color, debug, train_df, 
                                   exp_params=None, is_synthetic = False,
                                   is_val=False):
    """Return a function that builds the full sequence for one subject.

    All the static configuration is captured in the closure, so the returned
    builder only needs the per-subject data.
    """

    rng = np.random.default_rng() #for the random sampling of pixel_level augmentations

    def build_questionnaire_views(X, images_for_q, original_id, questionnaire_info, synth_transform=None):
        """Build the list of augmented views for one questionnaire."""
        list_of_views = []
        list_of_modality_names = []
        list_of_masked_views = []
        # if the image was rescaled, the grid coordinates must be rescaled too
        rescale_factor = questionnaire_info[str(X)]['rescale_factor']

        to_grayscale = exp_params.get('to_grayscale', False) if exp_params else False
        num_channels = 1 if to_grayscale else 3

        feature_extraction = (exp_params or {}).get('feature_extraction', False)

        for current_mode in ALL_MODALITIES:
            num, grid = grid_lookup(grid_dict, original_id, 'q' + str(X), current_mode)

            # indexes of modality_string_list matching the current mode
            mask = [j for j, m in enumerate(modality_string_list) if m == current_mode]
            if len(mask) == 0:
                continue
            selected_transforms = [augmentation_transform_list[j] for j in mask]
            selected_modality_names = [original_modality_names[j] for j in mask]

            raw_image_data = images_for_q.get(current_mode.lower())

            img,imputed = _decode_image(raw_image_data, num)
            
            
            if synth_transform: #i am creating synthetic time-series
                img = synth_transform(img, X-1)
            
            selected_transforms, selected_modality_names = _update_transforms(selected_transforms, selected_modality_names, 
                                                                              debug, feature_extraction)
            

            x_coords,n_x,y_coords = _set_coords(num, grid, rescale_factor, img)
            
            count_windows, n_elements_window, sampled_windows = _initialize_sampling(num, current_mode, selected_transforms, is_val)
            
            debug_pixel_space_augmentations = exp_params.get('debug_pixel_space_augmentations', False) 
            if debug_pixel_space_augmentations:
                selected_transforms = selected_transforms + selected_transforms +selected_transforms
                selected_modality_names = [n for name in selected_modality_names for n in (name, name + '_noaug',name + '_diff')]
            '''key_random = random.randint(0, 2**32 - 1) #unique key for the worker'''
            
            for k, augmentation_transform in enumerate(selected_transforms):
                #print(key_random, selected_modality_names[k],count_windows, flush=True)
                if augmentation_transform[0] == 'original':
                    img_view = img.copy()
                elif augmentation_transform[0] == 'grid':
                    # select a random grid crop
                    img_view = _grid_sampling(x_coords, y_coords, img, augmentation_transform, n_x, num)
                elif augmentation_transform[0] == 'window':
                    img_view, count_windows = _window_sampling(x_coords, y_coords, img, augmentation_transform, n_x, imputed, current_mode, selected_modality_names, 
                                             sampled_windows, count_windows, n_elements_window, k)
                elif augmentation_transform[0] == 'patch':
                    img_view, count_windows = _patch_sampling(x_coords, y_coords, img, augmentation_transform, n_x, imputed, selected_modality_names, 
                                             sampled_windows, count_windows, k)
                elif augmentation_transform[0] is None:
                    img_view = img.copy()
                else:
                    # a callable transform
                    img_view = augmentation_transform[1](img)
                
                pixel_space_augmentations=exp_params.get('image_pixel_space_augmentations', None)
                if (pixel_space_augmentations is not None) and (not is_val) and (not '_noaug' in selected_modality_names[k]): #apply augmentation in pixel space (only during training)
                    img_view = _augment(rng, pixel_space_augmentations)

                if invert_color:
                    img_view = ImageOps.invert(img_view)
                
                if to_grayscale:
                    img_view = T.Grayscale(num_output_channels=1)(img_view)

                if transform_func is not None:
                    if huggingface_transform: 
                        img_tensor = transform_func(images=img_view, return_tensors="pt")['pixel_values'][0]
                    else:
                        img_tensor = transform_func(img_view)
                else:
                    img_tensor = T.ToTensor()(img_view)

                list_of_views = _append_new_views(img_view, img_tensor, debug, feature_extraction, augmentation_transform, selected_transforms,
                                                    imputed, k, selected_modality_names, list_of_views, x_coords, y_coords)
                list_of_modality_names.append(selected_modality_names[k])
                list_of_masked_views.append(not imputed) #the views with true will be kept in training, the others masked

        return list_of_views, list_of_modality_names, rescale_factor, list_of_masked_views

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
        masks = []
        questionnaires = []  # per-frame metadata
        resized_list = []
        modalities = []
        global_properties = []
        local_properties = []

        q_to_keep=questionnaires_to_keep(last_q, censor_time, questionnaire_info, train_df,subject_id, 
                                         subject_images, case_grid_pattern, rempli_seulq12)

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

            views, names, rescale_factor, masked_views = build_questionnaire_views(
                X, subject_images[X], original_id, questionnaire_info, synth_transform=synth_transform)

            if len(views) == 0:
                continue

            frames.append(views if debug else torch.stack(views, dim=0))
            masks.append(torch.tensor(masked_views, dtype=torch.bool))
            questionnaires.append(X)
            resized_list.append(rescale_factor)
            modalities.append(names)
        global_properties, local_properties = get_properties(subject_id,train_df,exp_params, questionnaires)

        if not frames:
            return None

        sequence = frames if debug else torch.stack(frames, dim=0)  # (T_i, n_views, C, H, W)
        return sequence, questionnaires, modalities, resized_list, global_properties, local_properties, masks

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
    case_grid_pattern is between 1 and 13 ,
    case_dt can be missing (NaN) or numeric (remember is for the case only)'''
    if train_df is None and censor_time != 'pre_diagnosis':
        raise ValueError("train_df must be provided for censor_time other than 'pre_diagnosis' ")
    else:
        grid_pattern = train_df.loc[train_df['unique_id']==subject_id,'grid_pattern'].values[0]
        if {'rempli_pattern', 'case_pattern'}.issubset(train_df.columns):
            # if the questionnaire is marked as unavailable but was actually extracted ->
            # consider the document unreliable and exclude it
            rempli_pattern = train_df.loc[train_df['unique_id'] == subject_id, 'rempli_pattern'].values[0]
            case_pattern = train_df.loc[train_df['unique_id'] == subject_id, 'case_pattern'].values[0]

            case_chars = list(case_grid_pattern)
            grid_chars = list(grid_pattern)

            for q in range(13):
                if case_chars[q] == '1' and case_pattern[q] == '0':
                    case_chars[q] = '0'
                if grid_chars[q] == '1' and rempli_pattern[q] == '0':
                    grid_chars[q] = '0'

            case_grid_pattern = ''.join(case_chars)
            grid_pattern = ''.join(grid_chars)
    
    def filter_grid_pattern(case_grid_pattern, selected_questionnaires):
        new_list = []
        for q in selected_questionnaires:
            if case_grid_pattern[q-1]=='1':
                new_list.append(q)
        return new_list
    
    if rempli_seulq12==0: 
        #remove questionnaires>=12 if present and q12 was not filled alone
        last_valid=11
    else:
        last_valid=13
    
    if last_q is None:
        last_q=13

    if last_q>last_valid:
        last_q=last_valid    

    if censor_time=='all':
        list_questionnaires = list(range(1,last_valid+1))
    elif censor_time=='all_matched':
        list_questionnaires = list(range(1,last_valid+1))
        list_questionnaires = filter_grid_pattern(case_grid_pattern, list_questionnaires)
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
        for q in range(last_q+1,last_valid+1):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
                break
    elif censor_time=='last_successive_and_previous':
        list_questionnaires = [last_q]
        for q in range(last_q+1,last_valid+1):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
                break
        for q in range(last_q-1,0,-1):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
                break 
    elif censor_time=='successive':
        list_questionnaires = []
        for q in range(last_q+1,last_valid+1):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
        list_questionnaires = filter_grid_pattern(case_grid_pattern, list_questionnaires)
    elif censor_time=='long':
        def count_ones(s):
            return s.count('1')
        if count_ones(grid_pattern)>=8:
            list_questionnaires = [q for q in range(1,last_valid+1)]
        else:
            list_questionnaires = []
    elif censor_time=='first_and_last':
        list_questionnaires = []
        for q in range(1,last_valid+1):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
                break
        for q in range(last_valid,0,-1):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
                break
    elif censor_time=='last_abs':
        list_questionnaires = []
        for q in range(last_valid,0,-1):
            if grid_pattern[q-1]=='1' and q in subject_images:
                list_questionnaires.append(q)
                break
    
    #print(f"List {subject_id}: ",list_questionnaires)
    return list_questionnaires

# MARK: collate functions

