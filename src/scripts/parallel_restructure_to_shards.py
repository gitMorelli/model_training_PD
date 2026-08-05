import tarfile
import webdataset as wds
import glob
import os
import json
import pandas as pd
import torchvision.transforms as T
from PIL import Image
import io
import math
import multiprocessing as mp
import concurrent.futures
import numpy as np
import cv2
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Iterable

from src.utils.image_processing import convert_background_to_white, get_tiles, recolor_border_via_profiles
from src.utils.file_utils import get_id_data_from_h5_file 
from src.utils.data_loading_utils import prepare_pre_training

# from src.utils.file_utils import recreate_dir 
# NOTE: Avoid clearing directories programmatically in a distributed environment 
# to prevent race conditions. Clear the output directory manually before running the Slurm array.
params={
    'convert_to_jpg': False,
    'resize': None,
    'padded': False,
    'convert_to_white': True,
    'grouped': True,
    'scale_tolerance': 0.1,
    'hd5_FILE_PATH': "/mnt/beegfs01/scratch/a_morelli/extraction/final/results_aggregated/final_aggregated_data.h5",
    'questionnaire_templates_PATH':"/home/a_morelli/datasets/others/template_sizes.json",
    'SOURCE_folder': "/mnt/beegfs01/scratch/a_morelli/extraction/final/data",
    'LIST_OF_IDS_PD_PATH': "/home/a_morelli/datasets/id_lists/PD_training_set_20_07_26.parquet",
    'LIST_OF_IDS_PD_PRE_MATCHING_PATH': "/home/a_morelli/datasets/id_lists/final_table_for_matching_splitted_13_7_26.csv",
    'pre_computed_lists': "/mnt/beegfs02/scratch/a_morelli/model_training/shards/pre_computed_lists_03_08_26.json",
}

LIST_OF_IDS_PD_TEST_PATH = "/mnt/beegfs01/scratch/a_morelli/extraction/progress.csv"
QUESTIONNAIRES = [str(i) for i in range(1,14)] # q1 to q12, inclusive. Adjust as needed.
LIST_OF_IDS_HANDEDNESS_PATH = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs_w_sentences.csv"
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(i) for i in range(1,14)] # q1 to q12, inclusive. Adjust as needed.
ALL_MODALITIES_TO_INCLUDE = ["hand_sentences_full","hand", "number_random", "number", "X"] # ["hand", "number_random", "X","sentence"] # Adjust as needed based on which modalities you want to include in the WDS samples


CODE_TO_RUN = "for_PD" #"for_PD" #"for_handedness" #for_PD or for_handedness or for_PD_test, "for_PDpretraining"
#OUTPUT_PATH = "/mnt/beegfs01/scratch/a_morelli/model_training/sharded_test_parallel"

current_date = pd.Timestamp.now().strftime("%d_%m_%y")
def generate_output_name(source,code_to_run, params,current_date):
    folder_name = code_to_run.split('_', 1)[1]
    output_path = os.path.join(source, folder_name)
    name = "final"
    
    if params['convert_to_jpg']:
        name = name+"_jpeg"
    else:
        name = name+"_png"
        
    if params['resize'] is not None:
        name += "_resized_"+str(params['resize'])
    if params['padded']:
        name += "_padded"
    if params['convert_to_white']:
        name += "_whitebg"

    if params['grouped']:
        name += "_grouped"
    name += "_"+current_date
    
    output_path = os.path.join(output_path,name)
    
    return output_path,folder_name,name
OUTPUT_PATH, folder_name, name = generate_output_name("/mnt/beegfs02/scratch/a_morelli/model_training/shards", CODE_TO_RUN, params, current_date)


MAX_SHARD_SIZE = 1e9 # 1e9 ~1 GB per shard
MAX_SHARD_COUNT = 1000 # Max items per shard

def main():
    pre_computed_lists = json.load(open(params['pre_computed_lists']))
    assert pre_computed_lists['params']['LIST_OF_IDS_PD_PATH'] == params['LIST_OF_IDS_PD_PATH'], "Mismatch in LIST_OF_IDS_PD_PATH between pre-computed lists and current parameters."   
    assert pre_computed_lists['params']['LIST_OF_IDS_PD_PRE_MATCHING_PATH'] == params['LIST_OF_IDS_PD_PRE_MATCHING_PATH'], "Mismatch in LIST_OF_IDS_PD_PRE_MATCHING_PATH between pre-computed lists and current parameters."
    
    if CODE_TO_RUN == "for_PD_test":
        id_list = pd.read_csv(LIST_OF_IDS_PD_TEST_PATH)["id"].tolist()
        # Remove num_test_ids limits to process the whole dataset
        convert_to_wds_parallel(OUTPUT_PATH,id_list,function=process_chunk_PD_test,num_test_ids=5000,data=None) 
        # Run checks only on Task 0 to avoid messy logs
        if int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) == 0:
            run_checks(OUTPUT_PATH)
    if CODE_TO_RUN == "for_handedness":
        data = pd.read_csv(LIST_OF_IDS_HANDEDNESS_PATH)
        # Remove num_test_ids limits to process the whole dataset
        # Remove ids with q_{questionnaire_to_use}_num_X<1 
        #data = data[data[f'q_{QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS[0]}_num_X'] >= 1]
        for train_split in ['train','val','test']:
            split_data = data[data['split'] == train_split]
            id_list = split_data['ident_projet'].tolist()[:]
            output_path = os.path.join(OUTPUT_PATH,train_split)
            convert_to_wds_parallel(output_path,id_list,function=process_chunk_handedness,num_test_ids=None,data=split_data) 
            # Run checks only on Task 0 to avoid messy logs
            if int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) == 0:
                run_checks(output_path)
    if CODE_TO_RUN == "for_PD":
        modalities_to_shard = ALL_MODALITIES_TO_INCLUDE
        #read from parquet file 
        data = pd.read_parquet(params['LIST_OF_IDS_PD_PATH'])
        print(data.head())
        for train_split in ['train','val','test']:
            split_data = data[data['split'] == train_split]
            split_data['group_id'] = split_data['unique_id'].apply(lambda x: x.split('_')[1]) #extract the group id from the unique_id 
            if params['grouped']: #if grouping i have to split the jobs by case-control groups
                id_list = pre_computed_lists['for_PD_grouped'][train_split] 
            else:
                id_list = pre_computed_lists['for_PD'][train_split] #these are pre_shuffled
            #while ident_projet is just XXXXX
            output_path = os.path.join(OUTPUT_PATH,train_split)
            if params['grouped']:
                convert_to_wds_parallel(output_path,id_list,function=process_chunk_PD_grouped,modalities_to_shard=modalities_to_shard,
                                        num_test_ids=None,data=split_data)
            else:
                convert_to_wds_parallel(output_path,id_list,function=process_chunk_PD,modalities_to_shard=modalities_to_shard,
                                        num_test_ids=None,data=split_data) 
            # Run checks only on Task 0 to avoid messy logs
            if int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) == 0:
                run_checks(output_path)
    if CODE_TO_RUN == "for_PDpretraining":
        #read from parquet file 
        modalities_to_shard = ["hand", "number_random", "X"]
        data_selected = pd.read_parquet(params['LIST_OF_IDS_PD_PATH']) #has the ids selected for training
        
        data = pd.read_csv(params['LIST_OF_IDS_PD_PRE_MATCHING_PATH']) #has all the ids
        data = prepare_pre_training(data, data_selected) #remove all the ids selected for training and prepare grid_pattern and avail_pattern

        print(data.head())
        for train_split in ['train','val']:
            split_data = data[data['split'] == train_split]
            id_list = pre_computed_lists['for_PDpretraining'][train_split] #pre shuffled data
            output_path = os.path.join(OUTPUT_PATH,train_split)
            
            convert_to_wds_parallel(output_path,id_list,function=process_chunk_PD_pretraining,modalities_to_shard=modalities_to_shard,
                                    num_test_ids=None,data=split_data) 
            # Run checks only on Task 0 to avoid messy logs
            if int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) == 0:
                run_checks(output_path)
                #save the parameters used for the sharding in a json file in the output_path
                
    if int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) == 0:
        save_path = os.path.join(OUTPUT_PATH, "metadata.json")
        with open(save_path, "w") as f:
            json.dump(params, f)


# ---------------------------------------------------------------- helpers

def scan_tar(old_tar):
    """Group tar members into {questionnaire: [png members]} and {questionnaire: json member}."""
    sequences, json_files = {}, {}
    for m in old_tar.getmembers():
        if not m.isfile():
            continue
        questionnaire = m.name.split('/')[1][1:]   # 'qX' -> 'X'
        if m.name.endswith('.png'):
            if questionnaire in QUESTIONNAIRES:
                sequences.setdefault(questionnaire, []).append(m)
        elif m.name.endswith('.json'):
            json_files[questionnaire] = m
    for files in sequences.values():
        files.sort(key=lambda x: x.name)
    return sequences, json_files

def rescale_info(old_tar, json_files, questionnaire, template_size_data):
    if questionnaire not in json_files:
        return False, (1.0, 1.0)
    json_data = json.load(old_tar.extractfile(json_files[questionnaire]))
    return get_images_to_rescale(
        json_data, questionnaire, template_size_data, scale_tolerance=params['scale_tolerance']
    )

def encode_image(img):
    """Encode to the configured output format. Returns (bytes, extension)."""
    if params['convert_to_jpg']:
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        ext = 'jpg'
    else:
        ok, buf = cv2.imencode('.png', img)
        ext = 'png'
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for .{ext}")
    return buf.tobytes(), ext

def transform_modality(file_bytes, id_data, questionnaire, clean_name,
                       to_rescale, rescale_factor):
    """Decode -> tile detection -> recolor -> rescale -> preprocess -> encode."""
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    num_tiles, array_val = id_data['q' + questionnaire][clean_name][:2]
    coords = get_tiles(img, array_val, num_tiles)

    if params['convert_to_white']:
        # recolor before rescaling, otherwise grid and image dimensions mismatch
        img = recolor_border_via_profiles(img, coords)
    if to_rescale:
        img = cv2.resize(img, (0, 0), fx=rescale_factor[0], fy=rescale_factor[1],
                         interpolation=cv2.INTER_LANCZOS4)
    img = preprocess_image(img, params['resize'], params['padded'], params['convert_to_white'])
    return encode_image(img)


# ---------------------------------------------------------------- config

@dataclass
class ShardSpec:
    """Everything that differs between the three pipelines."""
    modalities: Iterable[str]
    # subject_id -> id used for the source tar filename and the h5 lookup
    source_id: Callable[[str], str] = lambda sid: sid
    # prefix subject id inside the file key (needed when a sample holds many subjects)
    prefix_subject_in_key: bool = False
    # return True to skip a questionnaire entirely
    skip_questionnaire: Optional[Callable[[object, str], bool]] = None
    # extra per-questionnaire metadata
    q_info_extra: Callable[[object, str], dict] = lambda row, q: {}
    # per-subject metadata written into the json
    subject_meta: Callable[[object, str], dict] = lambda row, sid: {}

def _extract_subject(tar_path, subject_id, id_row, spec, template_size_data):
    """Process one subject's tar. Returns (files dict, questionnaire_info dict)."""
    id_data = get_id_data_from_h5_file(params['hd5_FILE_PATH'], spec.source_id(subject_id))
    files_out, questionnaire_info = {}, {}

    with tarfile.open(tar_path, 'r') as old_tar:
        sequences, json_files = scan_tar(old_tar)

        for questionnaire, members in sequences.items():
            if spec.skip_questionnaire and spec.skip_questionnaire(id_row, questionnaire):
                continue

            to_rescale, rescale_factor = rescale_info(
                old_tar, json_files, questionnaire, template_size_data)
            questionnaire_info[questionnaire] = {
                'to_rescale': to_rescale,
                'rescale_factor': rescale_factor,
                **spec.q_info_extra(id_row, questionnaire),
            }

            for m in members:
                clean_name = os.path.basename(m.name).split('.')[0]
                if clean_name not in spec.modalities:
                    continue
                data_bytes, ext = transform_modality(
                    old_tar.extractfile(m).read(), id_data, questionnaire,
                    clean_name, to_rescale, rescale_factor)
                prefix = f"{subject_id}." if spec.prefix_subject_in_key else ""
                files_out[f"{prefix}q{questionnaire}.{clean_name}.{ext}"] = data_bytes

    return files_out, questionnaire_info

def _resolve_tars(subject_ids, spec):
    pairs = []
    for subject_id in subject_ids:
        tar_path = os.path.join(params['SOURCE_folder'], f"id_{spec.source_id(subject_id)}.tar")
        if os.path.isfile(tar_path):
            pairs.append((tar_path, subject_id))
        else:
            print(f"Warning: {tar_path} not found. Skipping.")
    return pairs


# ---------------------------------------------------------------- drivers

def _run_flat(output_path, worker_id, subject_ids, spec, data):
    """One WDS sample per subject."""
    template_size_data = json.load(open(params['questionnaire_templates_PATH']))
    pairs = _resolve_tars(subject_ids, spec)
    pattern = os.path.join(output_path, f"worker{worker_id}_shard-%06d.tar")
    print(f"[Worker {worker_id}] Starting conversion of {len(pairs)} subjects...")

    with wds.ShardWriter(pattern, maxsize=MAX_SHARD_SIZE, maxcount=MAX_SHARD_COUNT) as sink:
        for i, (tar_path, subject_id) in enumerate(pairs, 1):
            if i % 10 == 0:
                print(f"[Worker {worker_id}] Processing {i}/{len(pairs)}")
            id_row = data.loc[data['unique_id'] == subject_id].iloc[0]
            files, q_info = _extract_subject(tar_path, subject_id, id_row,
                                             spec, template_size_data)
            sample = dict(files)
            sample["__key__"] = subject_id
            sample["json"] = json.dumps({
                "subject": subject_id,
                "questionnaire_info": q_info,
                "shard_name": sink.fname,
                **spec.subject_meta(id_row, subject_id),
            }).encode("utf-8")
            sink.write(sample)
    print(f"[Worker {worker_id}] Complete!")

def _run_grouped(output_path, worker_id, group_ids, spec, data):
    """One WDS sample per group of subjects."""
    template_size_data = json.load(open(params['questionnaire_templates_PATH']))
    pattern = os.path.join(output_path, f"worker{worker_id}_shard-%06d.tar")
    print(f"[Worker {worker_id}] Starting conversion of {len(group_ids)} groups...")

    with wds.ShardWriter(pattern, maxsize=MAX_SHARD_SIZE, maxcount=MAX_SHARD_COUNT) as sink:
        for group_id in group_ids:
            members = data[data['group_id'] == group_id]['unique_id'].tolist()
            sample, subjects_dict = {}, {}
            for tar_path, subject_id in _resolve_tars(members, spec):
                id_row = data.loc[data['unique_id'] == subject_id].iloc[0]
                files, q_info = _extract_subject(tar_path, subject_id, id_row,
                                                 spec, template_size_data)
                sample.update(files)
                subjects_dict[subject_id] = {
                    "questionnaire_info": q_info,
                    "shard_name": sink.fname,
                    **spec.subject_meta(id_row, subject_id),
                }
            sample["__key__"] = group_id
            sample["json"] = json.dumps({
                "subjects": subjects_dict, "group_id": group_id
            }).encode("utf-8")
            sink.write(sample)
    print(f"[Worker {worker_id}] Complete!")

# ---------------------------------------------------------------- public API

def _pd_meta(row, subject_id):
    return {
        "label": row['diag_park_final1_quest'].item(),
        "at_least_warning": row['at_least_warning'].item(),
        "case_grid_pattern": row['case_grid_pattern'],
        "last_q": int(row['last_avail_q']),
    }

def _case_dt(row, q):
    return {"case_dt_dateq": row[f'case_dt_dateq{q}'].item()}

def process_chunk_PD(output_path, worker_id, id_chunk, modalities_to_shard, data=None):
    spec = ShardSpec(
        modalities=set(modalities_to_shard),
        source_id=lambda sid: sid.split("_")[0],
        q_info_extra=_case_dt,
        subject_meta=lambda row, sid: {**_pd_meta(row, sid),
                                       "rempli_seulq12": row['rempli_seulq12'].item()},
    )
    _run_flat(output_path, worker_id, id_chunk, spec, data)

def process_chunk_PD_grouped(output_path, worker_id, id_chunk, modalities_to_shard,data=None):
    spec = ShardSpec(
        modalities=set(modalities_to_shard),
        source_id=lambda sid: sid.split("_")[0],
        prefix_subject_in_key=True,
        #skip_questionnaire=lambda row, q: row['case_grid_pattern'][int(q) - 1] == '0',
        q_info_extra=_case_dt,
        subject_meta=_pd_meta,
    )
    _run_grouped(output_path, worker_id, id_chunk, spec, data)

def process_chunk_PD_pretraining(output_path, worker_id, id_chunk, modalities_to_shard, data=None):
    spec = ShardSpec(
        modalities=set(modalities_to_shard),
        subject_meta=lambda row, sid: {"grid_pattern": row['grid_pattern'],
                                       "avail_pattern": row['avail_pattern']},
    )
    #by default subject_id, questionnaire_info (with rescale values) and shardname are saved in the metadata
    _run_flat(output_path, worker_id, id_chunk, spec, data)

def preprocess_image(img_source,resize=None,padded=False, convert_bg_to_white=False):
    img = img_source.copy()

    h, w, c = img.shape  # height, width, channels

    if padded:
        # 2. Find the longer side
        max_dim = max(h, w)
        # 3. Create a square black canvas of zeros with the max dimension
        # (Matches the data type of the original image, usually uint8)
        padded_img = np.zeros((max_dim, max_dim, c), dtype=img.dtype)
        # 4. Paste the original image into the top-left corner
        padded_img[0:h, 0:w] = img
        img = padded_img
    if resize:
        if resize == 'half':
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        elif resize==224:
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    '''if convert_bg_to_white:
        img = convert_background_to_white(img)'''
    return img

def process_chunk_PD_test(output_path,worker_id, id_chunk,data=None):
    """
    Worker function executed by individual CPU cores.
    Each worker gets a unique ID and its own subset of subject IDs.
    """
    if not id_chunk:
        return

    # Create a unique output pattern for this specific worker to prevent file-write collisions
    output_pattern = os.path.join(output_path, f"worker{worker_id}_shard-%06d.tar")
    
    source_tars = []
    for subject_id in id_chunk:
        tar_path = os.path.join(params['SOURCE_folder'], f"id_{subject_id}.tar")
        if os.path.isfile(tar_path):
            source_tars.append(tar_path)
        else:
            print(f"Warning: {tar_path} not found. Skipping.")

    print(f"[Worker {worker_id}] Starting conversion of {len(source_tars)} subjects...")

    with wds.ShardWriter(output_pattern, maxsize=MAX_SHARD_SIZE, maxcount=MAX_SHARD_COUNT) as sink:
        for i, tar_path in enumerate(source_tars):
            if (i+1) % 10 == 0:
                print(f"[Worker {worker_id}] Processing {i+1}/{len(source_tars)}")
            
            subject_id = os.path.basename(tar_path).split('.')[0].split('_')[1]
            
            with tarfile.open(tar_path, 'r') as old_tar:
                members = old_tar.getmembers()
                sequences = {}
                
                # 1. Parse members to group PNGs
                for m in members:
                    if m.isfile() and m.name.endswith('.png'):
                        folder = m.name.split('/')[1] 
                        if folder not in sequences:
                            sequences[folder] = []
                        sequences[folder].append(m)
                
                # 2. Build ONE single WDS sample dictionary for the subject
                sample = {
                    "__key__": subject_id,
                    "json": json.dumps({
                        "subject": subject_id, 
                    }).encode("utf-8")
                }
                
                # 3. Add images
                for timestep, files in sequences.items():
                    files.sort(key=lambda x: x.name) 
                    
                    for m in files:
                        if os.path.basename(m.name) in ["hand.png", "number_random.png", "X.png"]:
                            file_bytes = old_tar.extractfile(m).read() 
                            clean_name = os.path.basename(m.name).split('.')[0]
                            img = Image.open(io.BytesIO(file_bytes))
                            
                            if params['resize']:
                                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                            
                            if params['convert_to_jpg']:
                                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                                    img = img.convert("RGB")
                                buffer = io.BytesIO()
                                img.save(buffer, format="JPEG", quality=90)
                                file_bytes = buffer.getvalue()
                                extension_out = 'jpg'
                            else:
                                buffer = io.BytesIO()
                                img.save(buffer, format="PNG") 
                                file_bytes = buffer.getvalue()
                                extension_out = 'png'
                            
                            file_key = f"{timestep}.{clean_name}.{extension_out}"
                            sample[file_key] = file_bytes
                            
                # 4. Write the massive subject sample to the shard
                sink.write(sample)

    print(f"[Worker {worker_id}] Complete!")

def process_chunk_handedness(output_path,worker_id, id_chunk,data=None):
    """
    Worker function executed by individual CPU cores.
    Each worker gets a unique ID and its own subset of subject IDs.
    """

    if not id_chunk:
        return

    # Create a unique output pattern for this specific worker to prevent file-write collisions
    output_pattern = os.path.join(output_path, f"worker{worker_id}_shard-%06d.tar")
    
    source_tars = []
    for subject_id in id_chunk:
        tar_path = os.path.join(params['SOURCE_folder'], f"id_{subject_id}.tar")
        if os.path.isfile(tar_path):
            source_tars.append(tar_path)
        else:
            print(f"Warning: {tar_path} not found. Skipping.")

    print(f"[Worker {worker_id}] Starting conversion of {len(source_tars)} subjects...")

    with wds.ShardWriter(output_pattern, maxsize=MAX_SHARD_SIZE, maxcount=MAX_SHARD_COUNT) as sink:
        current_shard_name = sink.fname 
        for i, tar_path in enumerate(source_tars):
            if (i+1) % 10 == 0:
                print(f"[Worker {worker_id}] Processing {i+1}/{len(source_tars)}")
            
            subject_id = os.path.basename(tar_path).split('.')[0].split('_')[1]

            id_data = get_id_data_from_h5_file(params['hd5_FILE_PATH'], subject_id)
            
            with tarfile.open(tar_path, 'r') as old_tar:
                members = old_tar.getmembers()
                sequences = {}
                
                # 1. Parse members to group PNGs
                for m in members:
                    if m.isfile() and m.name.endswith('.png'):
                        folder = m.name.split('/')[1] 
                        if folder[1:] in QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS: #folder=qX 
                            if folder not in sequences:
                                sequences[folder] = []
                            sequences[folder].append(m)
                

                # 2. Build ONE single WDS sample dictionary for the subject
                sample = {}
                #tile_coords = {}
                
                # 3. Add images
                for timestep, files in sequences.items():
                    files.sort(key=lambda x: x.name) 
                    #tile_coords[timestep] = {}
                    
                    for m in files:
                        clean_name = os.path.basename(m.name).split('.')[0]
                        if clean_name in MODALITIES_TO_INCLUDE: 

                            if clean_name == "X":
                                num_marks = data[data['ident_projet'] == subject_id][f'q_{timestep[1:]}_num_X'].values[0] # Access the value to ensure it exists and is not NaN
                            elif clean_name == "hand":
                                num_marks = data[data['ident_projet'] == subject_id][f'q_{timestep[1:]}_num_text'].values[0]
                            elif clean_name == "number_random":
                                num_marks = data[data['ident_projet'] == subject_id][f'q_{timestep[1:]}_num_digit'].values[0]
                            elif clean_name == "hand_sentences_full":
                                num_marks = data[data['ident_projet'] == subject_id][f'q_{timestep[1:]}_num_sent'].values[0]
                            if num_marks < 1:
                                continue

                            file_bytes = old_tar.extractfile(m).read()
                            np_arr = np.frombuffer(file_bytes, np.uint8)
                            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                            if len(img.shape) == 2:  # If it only has height and width (1 channel)
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                            array_val = id_data[timestep][clean_name][1] #get the array value for q6 number class
                            num_tiles = id_data[timestep][clean_name][0]
                            coords = get_tiles(img,array_val,num_tiles) #returns a list of [(xtl, ytl, xbr, ybr),x] 
                            #with x=tile number if tile contains a mark, -1 otherwise 
                            #tile_coords[timestep][clean_name] = coords[:]
                            if params['convert_to_white']:
                                img = recolor_border_via_profiles(img, coords)
                            
                            img = preprocess_image(img,params['resize'],params['padded'], params['convert_to_white'])
                            
                            if params['convert_to_jpg']:
                                # 3. Check for alpha channel (4 channels: BGRA) and drop it for JPEG conversion
                                if len(img.shape) == 3 and img.shape[2] == 4:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                                    
                                # 4. Encode to JPEG bytes with quality=90
                                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                                success, buffer = cv2.imencode('.jpg', img, encode_param)
                                if success:
                                    file_bytes = buffer.tobytes()
                                extension_out = 'jpg'
                            else:
                                # 5. Encode to PNG bytes
                                success, buffer = cv2.imencode('.png', img)
                                if success:
                                    file_bytes = buffer.tobytes()
                                extension_out = 'png'
                            
                            file_key = f"{timestep}.{clean_name}.{extension_out}"
                            sample[file_key] = file_bytes
                
                sample["__key__"] = str(subject_id)
                sample["json"]= json.dumps({
                        "subject": subject_id, 
                        "label": data[data['ident_projet'] == subject_id]['lateralite'].values[0],
                        #"tile_coords": tile_coords,
                        "shard_name" : current_shard_name
                    }).encode("utf-8")
                            
                # 4. Write the massive subject sample to the shard
                sink.write(sample)

    print(f"[Worker {worker_id}] Complete!")


def get_images_to_rescale(data,questionnaire, template_data, scale_tolerance=0.1):
    rescaling_factors = []
    n_images = len(data)
    for page in data:
        if questionnaire=='8' and page['page_number'] in [1,2]:
            #print("Skipping page ", page['page_number'], " of questionnaire 8 due to known issues with template dimensions.")
            n_images = len(data) - 2
            continue
        page_number = page['page_number']
        template_dim = template_data[questionnaire][str(page_number)]
        page_dim = page['dimensions']
        scale_x = page_dim[0]/template_dim[0]
        scale_y = page_dim[1]/template_dim[1]
        if abs(scale_x-1)>scale_tolerance or abs(scale_y-1)>scale_tolerance:
            #print(f"Page {page_number} has a scale factor outside the tolerance: (x: {scale_x:.2f}, y: {scale_y:.2f})")
            rescaling_factors.append((page_number, 1/scale_x, 1/scale_y))
    #get the average x and y rescaling factor across the pages that need to be rescaled
    avg_rescale_x = np.mean([f[1] for f in rescaling_factors]).item() if rescaling_factors else 1.0
    avg_rescale_y = np.mean([f[2] for f in rescaling_factors]).item() if rescaling_factors else 1.0
    if n_images>0 and len(rescaling_factors)>=int(n_images/2):
        return True, (avg_rescale_x, avg_rescale_y)
    else:
        return False, (1.0,1.0)


def convert_to_wds_parallel(output_path,id_list,function,modalities_to_shard,num_test_ids=None,data=None):
    # Ensure output directory exists (safe for multiple workers vs recreate_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Read and limit IDs
    if num_test_ids:
        id_list = id_list[:num_test_ids]

    # 2. Slurm Environmental Variables
    # SLURM_ARRAY_TASK_ID determines which node we are on (default 0 for local testing)
    slurm_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    # SLURM_ARRAY_TASK_MAX gives the upper bound of the array (default 0 for local testing)
    slurm_task_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", 0))
    total_slurm_tasks = slurm_task_max + 1
    
    # How many CPUs are allocated to this specific Slurm job? (default to local cores)
    cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))

    # 3. Calculate this Slurm Job's specific slice of the dataset
    total_ids = len(id_list)
    chunk_size = math.ceil(total_ids / total_slurm_tasks)
    start_idx = slurm_task_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_ids)
    task_id_list = id_list[start_idx:end_idx]
    
    print(f"--- Slurm Task {slurm_task_id}/{total_slurm_tasks-1} ---")
    print(f"Total dataset size: {total_ids}")
    print(f"Assigned to this node: {len(task_id_list)} subjects (indices {start_idx} to {end_idx-1})")
    print(f"Using {cpus_per_task} local CPUs/Workers")

    # 4. Distribute this node's slice among its local CPU cores
    local_chunk_size = math.ceil(len(task_id_list) / cpus_per_task)
    
    chunks_for_pool = []
    for i in range(cpus_per_task):
        c_start = i * local_chunk_size
        c_end = min(c_start + local_chunk_size, len(task_id_list))
        if c_start < len(task_id_list):
            # Create a globally unique worker ID across the entire cluster
            global_worker_id = (slurm_task_id * cpus_per_task) + i
            chunks_for_pool.append((global_worker_id, task_id_list[c_start:c_end]))

    # 5. Execute in parallel using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=cpus_per_task) as executor:
        futures = [
            executor.submit(function, output_path, worker_id, chunk, modalities_to_shard,data) 
            for worker_id, chunk in chunks_for_pool
        ]
        
        # Wait for all processes to finish and raise any exceptions that occurred inside workers
        for future in concurrent.futures.as_completed(futures):
            future.result() 

def run_checks(output_path=OUTPUT_PATH):
    # Only try to run the check if you aren't doing a massive distributed run, 
    # or look for a specific shard file that exists.
    try:
        sample_shard = glob.glob(os.path.join(output_path, "*.tar"))[0]
    except IndexError:
        print("No shards found to check.")
        return

    print("="*50)
    print(f"🔍 Inspecting: {sample_shard}")
    print("="*50)
    
    print("\n[Test 1] Raw Tar Archive Contents (First 15 files):")
    try:
        with tarfile.open(sample_shard, 'r') as tar:
            members = tar.getmembers()
            for m in members[:15]:
                print(f"  - {m.name} ({m.size} bytes)")
            print(f"  ... and {len(members) - 15} more files.")
    except Exception as e:
        print(f"❌ Failed to open tar file: {e}")
        return

    print("\n[Test 2] WebDataset Dictionary Grouping:")
    transform = T.ToTensor()
    dataset = wds.WebDataset(sample_shard).decode("pil")
    
    try:
        sample = next(iter(dataset))
        print(f"✅ Successfully loaded Sample Key: {sample['__key__']}")
        for key, value in sample.items():
            if key.startswith("__"):
                continue
            if key == "json":
                print(f"  - {key}: {value}")
            else:
                tensor_img = transform(value)
                print(f"  - {key}: Valid Image -> Tensor Shape: {list(tensor_img.shape)}")
    except StopIteration:
        print("❌ The shard is empty!")
    except Exception as e:
        print(f"❌ Failed to decode sample: {e}")

if __name__ == "__main__":
    main()