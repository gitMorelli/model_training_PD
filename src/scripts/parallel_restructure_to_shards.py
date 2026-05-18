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

# from src.utils.file_utils import recreate_dir 
# NOTE: Avoid clearing directories programmatically in a distributed environment 
# to prevent race conditions. Clear the output directory manually before running the Slurm array.

CONVERT_TO_JPG = False
RESIZE = False

# --- CONFIGURATION ---
SOURCE_folder = "/mnt/beegfs01/scratch/a_morelli/extraction/final/data"
LIST_OF_IDS_PATH = "/mnt/beegfs01/scratch/a_morelli/extraction/progress.csv"

if CONVERT_TO_JPG:
    OUTPUT_PATH = "/mnt/beegfs01/scratch/a_morelli/model_training/sharded_test_parallel_jpeg"
else:
    OUTPUT_PATH = "/mnt/beegfs01/scratch/a_morelli/model_training/sharded_test_parallel"
    
if RESIZE:
    OUTPUT_PATH += "_resized"

MAX_SHARD_SIZE = 1e9 # 1e9 ~1 GB per shard
MAX_SHARD_COUNT = 1000 # Max items per shard


def process_chunk(worker_id, id_chunk):
    """
    Worker function executed by individual CPU cores.
    Each worker gets a unique ID and its own subset of subject IDs.
    """
    if not id_chunk:
        return

    # Create a unique output pattern for this specific worker to prevent file-write collisions
    output_pattern = os.path.join(OUTPUT_PATH, f"worker{worker_id}_shard-%06d.tar")
    
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


def convert_to_wds_parallel(num_test_ids=None):
    # Ensure output directory exists (safe for multiple workers vs recreate_dir)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 1. Read and limit IDs
    id_list = pd.read_csv(LIST_OF_IDS_PATH)["id"].tolist()
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
            executor.submit(process_chunk, worker_id, chunk) 
            for worker_id, chunk in chunks_for_pool
        ]
        
        # Wait for all processes to finish and raise any exceptions that occurred inside workers
        for future in concurrent.futures.as_completed(futures):
            future.result() 

def run_checks():
    # Only try to run the check if you aren't doing a massive distributed run, 
    # or look for a specific shard file that exists.
    try:
        sample_shard = glob.glob(os.path.join(OUTPUT_PATH, "*.tar"))[0]
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
    # Remove num_test_ids limits to process the whole dataset
    convert_to_wds_parallel(num_test_ids=5000) 
    
    # Run checks only on Task 0 to avoid messy logs
    if int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) == 0:
        run_checks()