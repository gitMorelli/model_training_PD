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

from src.utils.data_loading.grouped.datasets import collate_groups_PD, collate_variable_sequences_PD_grouped

# Globals
ALL_MODALITIES = ['hand', 'number_random', 'X']

class SubjectBatch(NamedTuple):
    frames: torch.Tensor            # or list of dicts if debug
    seq_ids: torch.Tensor
    slot_ids: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor
    resized: list
    subject_ids: list
    modalities: list
    global_properties: Optional[torch.Tensor] = None   # (N, m1)
    local_properties: Optional[torch.Tensor] = None     # (sum T_i, m2)
OPTIONAL_FIELD_SPECS = {
    "global_properties": (6, lambda vals: torch.stack(vals, dim=0)),   # (N, m1)
    "local_properties":  (7, lambda vals: torch.cat(vals, dim=0)),     # (sum T_i, m2)
}


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

    size_dict = {
        'hand': 100,
        'number_random': 60,
        'X': 60
    } #average of the median size of patches in the modality

    n_elements_window_dict = {
        'hand': 9,
        'number_random': 9,
        'X': 16
    }

    rng = np.random.default_rng() #for the random sampling of pixel_level augmentations

    def build_questionnaire_views(X, images_for_q, original_id, questionnaire_info, synth_transform=None):
        """Build the list of augmented views for one questionnaire."""
        list_of_views = []
        list_of_modality_names = []
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
            

            if debug and feature_extraction==False:
                #if feature_extraction is true i don't want the original modality
                selected_transforms = [('original',None)] + selected_transforms
                selected_modality_names = ([selected_modality_names[0].split('_')[0] + '_original']
                                           + selected_modality_names)
            elif debug and feature_extraction==True: #for feature extraction i want only the original modality, no augmentation
                selected_transforms = [('original',None)] 
                selected_modality_names = ([selected_modality_names[0].split('_')[0] + '_original'])

            if num>0:
                x_coords = [0] + sorted(list(grid[0, :] * rescale_factor[0])) + [img.width]
                n_x = len(x_coords) - 1
                y_coords = [0] + sorted(list(grid[1, :] * rescale_factor[1])) + [img.height]
            else:
                x_coords = [0, img.width]
                n_x = 1
                y_coords = [0, img.height]
            
            #if i have a window modality i need to prepare some data
            n_elements_window = n_elements_window_dict.get(current_mode, 9)
            if num>0 and 'window' in [t[0] for t in selected_transforms]:
                #count the number of window views you have in selected_transforms
                n_window_views = sum(1 for t in selected_transforms if t[0] == 'window')
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
                count_windows = 0
            
            debug_pixel_space_augmentations = exp_params.get('debug_pixel_space_augmentations', False) 
            if debug_pixel_space_augmentations:
                selected_transforms = selected_transforms + selected_transforms +selected_transforms
                selected_modality_names = [n for name in selected_modality_names for n in (name, name + '_noaug',name + '_diff')]
            '''key_random = random.randint(0, 2**32 - 1) #unique key for the worker
            print(key_random, selected_modality_names, flush=True)
            print(key_random, selected_transforms, flush=True)'''
            for k, augmentation_transform in enumerate(selected_transforms):
                #print(key_random, selected_modality_names[k],count_windows, flush=True)
                if augmentation_transform[0] == 'original':
                    img_view = img.copy()
                elif augmentation_transform[0] == 'grid':
                    # select a random grid crop
                    if num > 0:
                        rand_num = random.randint(1, num)
                        img_view = get_grid_sample(x_coords, y_coords, img, rand_num, n_x)
                    else:
                        img_view = img.copy()
                    img_view = augmentation_transform[1](img_view)  # apply the callable transform
                elif augmentation_transform[0] == 'window':
                    # scale the size based on the number of elements in the window (if 9 -> 3xexpected size of a chunk)
                    size = int(size_dict.get(current_mode, 60)*np.sqrt(n_elements_window))  
                    #create a sizexsizex3 blank image
                    img_view = Image.new('RGB', (size, size), color=(255, 255, 255))
                    if num>0:
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
                elif augmentation_transform[0] is None:
                    img_view = img.copy()
                else:
                    # a callable transform
                    img_view = augmentation_transform[1](img)
                
                pixel_space_augmentations=exp_params.get('image_pixel_space_augmentations', None)
                if (pixel_space_augmentations is not None) and (not is_val) and (not '_noaug' in selected_modality_names[k]): #apply augmentation in pixel space (only during training)
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

            views, names, rescale_factor = build_questionnaire_views(
                X, subject_images[X], original_id, questionnaire_info, synth_transform=synth_transform)

            if len(views) == 0:
                continue

            frames.append(views if debug else torch.stack(views, dim=0))
            questionnaires.append(X)
            resized_list.append(rescale_factor)
            modalities.append(names)
        global_properties, local_properties = get_properties(subject_id,train_df,exp_params, questionnaires)

        if not frames:
            return None

        sequence = frames if debug else torch.stack(frames, dim=0)  # (T_i, n_views, C, H, W)
        return sequence, questionnaires, modalities, resized_list, global_properties, local_properties

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

#---------- yield smaples
def create_sequence_flattener_PD_multiview(transform_func, augmentation_transform_list, n_views,
                                           modality_string_list, original_modality_names,
                                           exclusion_set, exp_params=None,huggingface_transform=False,
                                           invert_color=False, grid_dict=None, censor_time='pre_diagnosis',
                                           filter_modality='number_random', debug=False, train_df=None, is_val=False):
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

            sequence, questionnaires, modalities, resized_list, global_properties, local_properties = result
            yield sequence, label, subject_id, questionnaires, modalities, resized_list, global_properties, local_properties

    return flatten_samples

def create_sequence_group_flattener_PD_multiview(transform_func, augmentation_transform_list,
                                    modality_string_list, original_modality_names,
                                    exclusion_set, grid_dict, censor_time,
                                    filter_modality, exp_params=None,huggingface_transform=False,
                                    invert_color=False, debug=False, train_df=None, is_val=False, **kw):
    build_sequence = _make_subject_sequence_builder(
        transform_func, augmentation_transform_list, modality_string_list,
        original_modality_names, grid_dict, censor_time, filter_modality,
        huggingface_transform, invert_color, debug, train_df,exp_params=exp_params,
        is_val=is_val)

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

                rempli_seulq12 = 1  #i ignore the rempli_seul problem for now
                result = build_sequence(subject_id, meta["last_q"],
                                        meta["questionnaire_info"],
                                        image_pool.get(subject_id, {}),meta['case_grid_pattern'], rempli_seulq12)
                if result is None:
                    continue
                sequence, questionnaires, modalities, resized, _, _ = result
                label = torch.tensor(meta["label"], dtype=torch.long)
                group.append((sequence, label, subject_id,
                              questionnaires, modalities, resized))

            # only yield valid groups: exactly one positive, ≥1 control
            labels = [g[1].item() for g in group]
            if sum(labels) == 1 and len(group) >= 2:
                yield group_id, group

    return flatten_groups


#------ collate functions
def _maybe_aggregate(samples, field_name):
    idx, agg = OPTIONAL_FIELD_SPECS[field_name]
    if all(len(s) > idx and s[idx] is not None for s in samples):
        return agg([s[idx] for s in samples])
    return None

def collate_variable_sequences_PD(samples, debug=False):
    sequences       = [s[0] for s in samples]
    labels          = torch.stack([s[1] for s in samples])
    subject_ids     = [s[2] for s in samples]
    questionnaires  = [s[3] for s in samples]
    modalities_list = [s[4] for s in samples]
    resized_list    = [s[5] for s in samples]

    global_properties = _maybe_aggregate(samples, "global_properties") #if global_properties is None for any sample, it will return None
    local_properties  = _maybe_aggregate(samples, "local_properties")

    return _collate_subject_level(sequences, labels, subject_ids, questionnaires,
                                   modalities_list, resized_list, debug=debug,
                                   global_properties=global_properties,
                                   local_properties=local_properties)

def _collate_subject_level(sequences, labels, subject_ids, questionnaires,
                            modalities_list, resized_list, debug=False,
                            global_properties=None, local_properties=None):
    if debug:
        lengths = torch.tensor([len(seq) for seq in sequences])
    else:
        lengths = torch.tensor([seq.shape[0] for seq in sequences])

    seq_ids, slot_ids = [], []
    resized, modalities = [], []
    if debug:
        frames = []
    else:
        frames = torch.cat(sequences, dim=0)

    for b, qs in enumerate(questionnaires):
        for i, q in enumerate(qs):
            seq_ids.append(b)
            slot_ids.append(int(q) - 1)
            resized.append(resized_list[b][i])
            modalities.append(modalities_list[b][i])
            if debug:
                frames.append(sequences[b][i])

    return SubjectBatch(
        frames=frames,
        seq_ids=torch.tensor(seq_ids),
        slot_ids=torch.tensor(slot_ids),
        lengths=lengths,
        labels=labels,
        resized=resized,
        subject_ids=subject_ids,
        modalities=modalities,
        global_properties=global_properties,
        local_properties=local_properties,
    )


#------ Build dataset --------
def prepare_PD_dataset(shard_pattern, split_workers=True, batch_size=4, transform=None, exclusion_set=set(), modality='X',
                       huggingface_transform=False,augmentation_transform=None, invert_color=False,n_views=1, grid_dict = None,
                       censor_time='pre_diagnosis', filter_modality='digit', debug=False, grouped=False, train_df=None, exp_params=None, 
                       partial_batch=False, is_val=False):
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

    if grid_dict: 
        if modality_string != 'all' and not isinstance(modality, list) and n_views >= 1:
            dataset = (dataset
                #.decode(decode_approach)
                .compose(create_sequence_flattener_PD_grid(transform,augmentation_transform, n_views=n_views ,modality_string=modality_string,
                                                    exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color,
                                                    grid_dict=grid_dict, censor_time=censor_time)) # This replaces .map() and .select()
                .batched(batch_size,
                        collation_fn=partial(collate_variable_sequences_PD, questionnaire_to_slot=None),
                        partial=partial_batch)
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
                                    exp_params=exp_params, is_val=is_val)) # This replaces .map() and .select()
                .batched(batch_size,
                        collation_fn=partial(collate_fn, debug=debug),
                        partial=partial_batch)
            )
        else:
            raise ValueError("grid_dict must be provided for PD dataset preparation.")
        
    
    return dataset

#pipelines
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
####################################################################################################################




