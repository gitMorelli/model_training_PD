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

# from src.utils.file_utils import recreate_dir 
# NOTE: Avoid clearing directories programmatically in a distributed environment 
# to prevent race conditions. Clear the output directory manually before running the Slurm array.

CONVERT_TO_JPG = False
RESIZE = True
PADDED = True

# --- CONFIGURATION ---
SOURCE_folder = "/mnt/beegfs01/scratch/a_morelli/extraction/final/data"

LIST_OF_IDS_PD_TEST_PATH = "/mnt/beegfs01/scratch/a_morelli/extraction/progress.csv"

LIST_OF_IDS_HANDEDNESS_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/handedness_model_ids.csv"
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']

#OUTPUT_PATH = "/mnt/beegfs01/scratch/a_morelli/model_training/sharded_test_parallel"
OUTPUT_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"

if CONVERT_TO_JPG:
    OUTPUT_PATH = os.path.join(OUTPUT_PATH,"jpeg")
else:
    OUTPUT_PATH = os.path.join(OUTPUT_PATH,"png")
    
if RESIZE:
    OUTPUT_PATH += "_resized"
if PADDED:
    OUTPUT_PATH += "_padded"

MAX_SHARD_SIZE = 1e9 # 1e9 ~1 GB per shard
MAX_SHARD_COUNT = 1000 # Max items per shard

CODE_TO_RUN = "for_handedness" #for_PD or for_handedness or for_PD_test

def preprocess_image(img_source,resize=False,padded=False):
    img = img_source.copy()

    if len(img.shape) == 2:  # If it only has height and width (1 channel)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
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
        tar_path = os.path.join(SOURCE_folder, f"id_{subject_id}.tar")
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
                            
                            if RESIZE:
                                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                            
                            if CONVERT_TO_JPG:
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
        tar_path = os.path.join(SOURCE_folder, f"id_{subject_id}.tar")
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
                        if folder[1] in QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS: #folder=qX 
                            if folder not in sequences:
                                sequences[folder] = []
                            sequences[folder].append(m)
                

                # 2. Build ONE single WDS sample dictionary for the subject
                sample = {
                    "__key__": subject_id,
                    "json": json.dumps({
                        "subject": subject_id, 
                        "label": data[data['ident_projet'] == subject_id]['lateralite'].values[0],
                    }).encode("utf-8")
                }
                
                # 3. Add images
                for timestep, files in sequences.items():
                    files.sort(key=lambda x: x.name) 
                    
                    for m in files:
                        if os.path.basename(m.name) in ["hand.png", "number_random.png", "X.png"]: 
                            clean_name = os.path.basename(m.name).split('.')[0]

                            if clean_name == "X":
                                num_marks = data[data['ident_projet'] == subject_id][f'q_{timestep[1]}_num_X'].values[0] # Access the value to ensure it exists and is not NaN
                            elif clean_name == "hand":
                                num_marks = data[data['ident_projet'] == subject_id][f'q_{timestep[1]}_num_text'].values[0]
                            elif clean_name == "number_random":
                                num_marks = data[data['ident_projet'] == subject_id][f'q_{timestep[1]}_num_digit'].values[0]
                            if num_marks < 1:
                                continue

                            file_bytes = old_tar.extractfile(m).read()
                            np_arr = np.frombuffer(file_bytes, np.uint8)
                            img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                            
                            img = preprocess_image(img,RESIZE,PADDED)
                            
                            if CONVERT_TO_JPG:
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
                            
                # 4. Write the massive subject sample to the shard
                sink.write(sample)

    print(f"[Worker {worker_id}] Complete!")

def convert_to_wds_parallel(output_path,id_list,function,num_test_ids=None,data=None):
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
            executor.submit(function, output_path, worker_id, chunk, data) 
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