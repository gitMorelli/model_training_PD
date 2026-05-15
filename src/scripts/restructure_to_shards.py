import tarfile
import webdataset as wds
import glob
import os
import json
import pandas as pd
import torchvision.transforms as T

from src.utils.file_utils import recreate_dir

# --- CONFIGURATION ---
SOURCE_folder = "/mnt/beegfs01/scratch/a_morelli/extraction/final/data"
LIST_OF_IDS_PATH = "/mnt/beegfs01/scratch/a_morelli/extraction/progress.csv"
OUTPUT_PATH = "/mnt/beegfs01/scratch/a_morelli/model_training/sharded_test"
OUTPUT_PATTERN = OUTPUT_PATH+"/dataset_shard-%06d.tar" # The output format
MAX_SHARD_SIZE = 0.5e9 #1e9 ~1 GB per shard is Optimal for WDS
MAX_SHARD_COUNT = 10000 # Max items per shard


def convert_to_wds():
    recreate_dir(OUTPUT_PATH) # Clear output directory before writing new shards

    num_test_ids = 500
    id_list =  pd.read_csv(LIST_OF_IDS_PATH)["id"].tolist()  # Assuming 'id' column contains the subject IDs
    id_list = id_list[:num_test_ids]
    print(f"Found {len(id_list)} source tar files. Starting conversion...")

    source_tars = []
    for subject_id in id_list:
        tar_path = os.path.join(SOURCE_folder, f"id_{subject_id}.tar")
        if os.path.isfile(tar_path):
            source_tars.append(tar_path)
        else:
            print(f"Warning: {tar_path} not found. Skipping.")
    
    # ShardWriter automatically creates new tar files (shards) when they reach 1GB
    with wds.ShardWriter(OUTPUT_PATTERN, maxsize=MAX_SHARD_SIZE, maxcount=MAX_SHARD_COUNT) as sink:
        
        for i,tar_path in enumerate(source_tars):
            if (i+1) % 50 == 0:
                print(f"Processing {i+1}/{len(source_tars)}: {tar_path}")
            # Extract subject ID (e.g., 'subject_001.tar' -> '001')
            subject_id = os.path.basename(tar_path).split('.')[0].split('_')[1]
            
            with tarfile.open(tar_path, 'r') as old_tar:
                members = old_tar.getmembers()
                
                sequences = {}
                page_dimensions = None
                
                # 1. Parse all members to group PNGs and find the JSON file
                for m in members:
                    if m.isfile():
                        # --- FIX FOR TO_DO: Read JSON file ---
                        '''if m.name.endswith('.json'):
                            json_bytes = old_tar.extractfile(m).read()
                            # Assuming the JSON contains a key called "dimensions" or similar
                            parsed_json = json.loads(json_bytes.decode("utf-8"))
                            page_dimensions = parsed_json.get("page_dimensions", "Not Found")'''
                            
                        # Group PNG files by folder
                        if m.name.endswith('.png'):
                            # e.g., "id_xxxx/q1/file_1.png" -> split gives ['id_xxxx', 'q1', 'file_1.png']
                            folder = m.name.split('/')[1] 
                            if folder not in sequences:
                                sequences[folder] = []
                            sequences[folder].append(m)
                
                # 2. Build ONE single WDS sample dictionary for the entire subject
                sample = {
                    "__key__": subject_id,
                    "json": json.dumps({
                        "subject": subject_id, 
                        "page_dimensions": page_dimensions
                    }).encode("utf-8")
                }
                
                # 3. Add the images from ALL timesteps into this single sample
                for timestep, files in sequences.items():
                    # Sort them to maintain consistent ordering
                    files.sort(key=lambda x: x.name) 
                    
                    for m in files:
                        # Apply your filename filter
                        if os.path.basename(m.name) in ["hand.png", "number_radom.png", "X.png"]:
                            # Read raw bytes directly from old tar
                            file_bytes = old_tar.extractfile(m).read() 
                            clean_name = os.path.basename(m.name).split('.')[0]
                            
                            # Use format: timestep.index.png (e.g., "q1.0.png", "q1.1.png")
                            # This prevents 'q2' files from overwriting 'q1' files in the dictionary
                            file_key = f"{timestep}.{clean_name}.png"
                            sample[file_key] = file_bytes
                            
                # 4. Write the massive subject sample to the shard
                sink.write(sample)
    print("Conversion complete! Your WDS shards are ready.")

def run_checks():
    SHARD_PATH = OUTPUT_PATTERN.replace("%06d", "000000") # Check the first shard
    print("="*50)
    print(f"🔍 Inspecting: {SHARD_PATH}")
    print("="*50)
    
    # ---------------------------------------------------------
    # TEST 1: Raw File Inspection
    # ---------------------------------------------------------
    print("\n[Test 1] Raw Tar Archive Contents (First 15 files):")
    try:
        with tarfile.open(SHARD_PATH, 'r') as tar:
            members = tar.getmembers()
            for m in members[:15]:
                print(f"  - {m.name} ({m.size} bytes)")
            print(f"  ... and {len(members) - 15} more files.")
    except Exception as e:
        print(f"❌ Failed to open tar file: {e}")
        return

    # ---------------------------------------------------------
    # TEST 2: WebDataset Parsing & Decoding
    # ---------------------------------------------------------
    print("\n[Test 2] WebDataset Dictionary Grouping:")
    
    # Simple transform to convert the PIL images to PyTorch Tensors
    transform = T.ToTensor()
    
    # Load just this single shard
    dataset = (
        wds.WebDataset(SHARD_PATH)
        .decode("pil") # Decodes raw bytes into PIL Images
    )
    
    try:
        # Grab the very first sample from the dataset
        sample = next(iter(dataset))
        
        print(f"✅ Successfully loaded Sample Key: {sample['__key__']}")
        print("Dictionary Contents:")
        
        # Print out exactly what WebDataset grouped together
        for key, value in sample.items():
            if key == "__key__":
                continue
                
            if key == "json":
                print(f"  - {key}: {value}")
            else:
                # Apply the transform to prove it's a valid, uncorrupted image
                tensor_img = transform(value)
                print(f"  - {key}: Valid Image -> Tensor Shape: {list(tensor_img.shape)}")
                
    except StopIteration:
        print("❌ The shard is empty!")
    except Exception as e:
        print(f"❌ Failed to decode sample: {e}")

if __name__ == "__main__":
    convert_to_wds()