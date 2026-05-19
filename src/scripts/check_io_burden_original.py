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

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset
from src.utils.model_utils import SimpleMockModel

LIST_OF_IDS_PATH = "/mnt/beegfs01/scratch/a_morelli/extraction/progress.csv"
DATA_PATH = "/mnt/beegfs01/scratch/a_morelli/extraction/final/data"
#SHARD_PATTERN = "/mnt/beegfs01/scratch/a_morelli/model_training/sharded_test/dataset_shard-{000000..000000}.tar"
SHARD_PATTERN = "/mnt/beegfs01/scratch/a_morelli/model_training/sharded_test/dataset_shard-*.tar"
SHARD_PATTERN_jpeg = "/mnt/beegfs01/scratch/a_morelli/model_training/sharded_test_jpeg/dataset_shard-*.tar" 
SHARD_PATTERN_resized = "/mnt/beegfs01/scratch/a_morelli/model_training/sharded_test_resized/dataset_shard-*.tar"
SHARD_PATTERN_parallel = "/mnt/beegfs01/scratch/a_morelli/model_training/sharded_test_parallel/worker*_shard-*.tar"

resize_transform = T.Compose([
    T.Resize((224, 224)), # Change 224, 224 to whatever resolution your model expects
    T.ToTensor()
])
resize_transform_no_to_tensor = T.Resize((224, 224)) # For WebDataset, we decode to PIL first, then resize, then convert to tensor in the process_wds_sample function


def io_benchmark():
    args = get_args()
    num_workers = args.num_workers
    batch_size = args.batch_size
    # This function can be used to run the I/O benchmark separately if needed
    # --- SETUP YOUR MOCK DATA HIERARCHY HERE ---
    mock_sequence_list = []
    
    id_names= pd.read_csv(LIST_OF_IDS_PATH)["id"].tolist()  # Assuming 'id' column contains the subject IDs

    # Simulating 100 subjects (100 .tar files)
    for subject_id in range(500):
        id_name = id_names[subject_id]
        tar_file_path = os.path.join(DATA_PATH, f"id_{id_name}.tar")

        '''with tarfile.open(tar_file_path, 'r') as tar:
            print(f"--- First 15 files in {tar_file_path} ---")
            for member in tar.getmembers()[:15]:
                print(f"'{member.name}'")
        return'''
        
        # Simulating K=10 time steps (folders) per subject
        internal_files = []
        for q_idx in range(1, 14): 
            # 6 png images inside folder q{i}
            internal_files.extend([f"id_{id_name}/q{q_idx}/hand.png", f"id_{id_name}/q{q_idx}/number_radom.png", f"id_{id_name}/q{q_idx}/X.png"])
            
        mock_sequence_list.append((tar_file_path, internal_files))

    dataset = MultiTarSequenceDataset(mock_sequence_list, max_open_tars=20, transform=resize_transform)
    
    for workers in [0, 4, 8, 16]:
        print(f"\n--- Benchmarking with num_workers={workers} ---")
        # NOTE: Set num_workers to your intended production amount (e.g., 4, 8, or 16)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

        print("Starting I/O Benchmark for Multi-Tar Structure...")
        start_time = time.time()
        
        batches_to_test = 50
        total_images_processed = 0

        # Ensure your dummy .tar files actually exist if you run this exactly as is,
        # otherwise replace the mock generation above with your actual parsed filename lists!
        try:
            for i, batch in enumerate(dataloader):
                total_images_processed += batch.shape[0] * batch.shape[1]
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1} batches...")
                    
                if i == batches_to_test - 1:
                    break
        except FileNotFoundError as e:
            print(f"\n[!] Error: {e}")
            print("Make sure to map the 'mock_sequence_list' logic to your actual file lists!")
            exit()

        end_time = time.time()
        total_time = end_time - start_time
        
        io_throughput = total_images_processed / total_time
        print("-" * 30)
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Images Processed: {total_images_processed}")
        print(f"** I/O Throughput: {io_throughput:.2f} images/sec **")


def gpu_dummy_benchmark():
    # Ensure GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu = torch.cuda.get_device_name(device_id)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
        print(f"--- GPU DIAGNOSTICS ---")
        print(f"Active GPU: {gpu}")
        print(f"CUDA_VISIBLE_DEVICES: {visible}")
        print(f"-----------------------")

    # --- DEFINE BATCH SHAPE ---
    # Batch size: 32, Sequence length: 6, Channels: 3, Image size: 224x224
    batch_size = 4
    images_per_sequence = 13*3 
    h, w = 224, 224

    model = SimpleMockModel(seq_length=images_per_sequence).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Create dummy data living entirely on the GPU
    dummy_input = torch.randn(batch_size, images_per_sequence, 3, h, w).to(device)
    dummy_target = torch.randint(0, 10, (batch_size,)).to(device)

    print("Starting Compute Benchmark (Warming up GPU...)")
    # Warmup loop (GPUs take a few iterations to reach peak clock speeds)
    for _ in range(5):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = loss_fn(output, dummy_target)
        loss.backward()
        optimizer.step()
        
    # Force CUDA to finish all operations before starting the timer
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    batches_to_test = 50

    print("Benchmarking...")
    for _ in range(batches_to_test):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = loss_fn(output, dummy_target)
        loss.backward()
        optimizer.step()

    # Synchronize again before stopping the timer
    if device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time
    total_images_processed = batches_to_test * batch_size * images_per_sequence

    compute_throughput = total_images_processed / total_time
    print("-" * 30)
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Images Processed: {total_images_processed}")
    print(f"** Compute Throughput: {compute_throughput:.2f} images/sec **")


def process_wds_sample(sample,transform='no_transform',seq_length=13*3):
    """
    Simulates the real processing load: finding all images in the dictionary,
    resizing them, and stacking them into a single tensor.
    """
    images = []
    empty_image = torch.zeros(3, 224, 224) # For padding if we have fewer than expected images
    
    for key, value in sample.items():
        # Skip metadata and hidden WebDataset keys
        if key.endswith(".png") or key.endswith(".jpg") or key.endswith(".jpeg"):
            # The 'value' is already a PIL Image because we used .decode("pil")
            try:
                if isinstance(value, torch.Tensor):
                    if transform == 'resize':
                        img_tensor = resize_transform_no_to_tensor(value)
                    elif transform == 'no_transform':
                        img_tensor = value
                else:
                    if transform == 'resize':
                        img_tensor = resize_transform(value)
                    elif transform == 'no_transform':
                        img_tensor = T.ToTensor()(value)
                images.append(img_tensor)
            except Exception as e:
                print(f"Skipping corrupted image {key}: {e}")

    # Stack all images found for this subject into [N_images, 3, 224, 224]
    if len(images) < seq_length:
        # If we have fewer than the expected number of images, pad with zeros
        padding_needed = seq_length - len(images)
        images.extend([empty_image] * padding_needed)
    return torch.stack(images), sample.get("__key__", "unknown")

def io_benchmark_webdataset():
    args = get_args()
    num_workers = args.num_workers
    batch_size = args.batch_size
    batches_to_test = -1 #if you want to test the entire dataset, set this to -1 or a very large number
    load_in_memory = True # Set to True if you want to test the in-memory dataset approach 
    print(f"Setting up WebDataset pipeline with {num_workers} workers...")

    patterns_to_test = [SHARD_PATTERN_resized]#[SHARD_PATTERN, SHARD_PATTERN_jpeg, SHARD_PATTERN_resized]
    for shard_pattern in patterns_to_test:
        approaches_to_test = ["pil"]#["torchrgb", "pil"]
        for decode_approach in approaches_to_test:
            split_workers_tests = [True]#[True, False]  
            for split_workers in split_workers_tests:
                test_prefetch = [2]#[None, 2, 4]
                for prefetch_factor in test_prefetch: 
                    # 1. Use glob to find all files matching the pattern
                    shard_files = glob.glob(shard_pattern)
                    # Sort them just to be safe so they load in order
                    shard_files.sort()

                    parts = shard_pattern.split("/")
                    if "resize" in parts[-2]:
                        print("Found 'resize' in the immediate parent folder!")
                        transform = 'no_transform'
                    else:
                        transform = 'resize'

                    if load_in_memory:
                        dataset = InMemoryWdsDataset(
                            shard_files=shard_files, 
                            decode_approach=decode_approach, 
                            transform=transform, 
                            seq_length=3*13
                        )
                    else:
                        # 1. Define the base WDS Pipeline
                        dataset = wds.WebDataset(shard_files, shardshuffle=100)

                        # 2. Conditionally apply worker splitting
                        if split_workers:
                            dataset = dataset.select(wds.split_by_worker)

                        # 3. Apply the remaining transformations
                        dataset = (dataset
                            .decode(decode_approach)
                            .map(lambda sample: process_wds_sample(sample, transform, 3*13)) 
                            .batched(batch_size)
                        )
                    print("#" * 50)
                    print(f"🚀 Starting WebDataset I/O Benchmark for pattern={shard_pattern} 🚀")
                    print(f"Loading Approach: {'In-Memory' if load_in_memory else 'On-the-Fly'}")
                    print(f"Decoding Approach: {decode_approach}, Transform: {transform}")
                    print(f"Worker Splitting: {'Enabled' if split_workers else 'Disabled'}")
                    print(f"Prefetch Factor: {prefetch_factor}")
                    print(f"Found {len(shard_files)} shard files matching the pattern.")
                    worker_setting = [2, 4, 8, 16, 32]
                    for worker in worker_setting:
                        print(f"\n--- Benchmarking with num_workers={worker} ---")
                        # 2. Wrap in a standard PyTorch DataLoader for multiprocessing
                        # Note: WDS handles batching, so DataLoader batch_size is None
                        if load_in_memory:
                            dataloader = DataLoader(
                                dataset, 
                                batch_size=batch_size,  # Passed here directly!
                                shuffle=True, 
                                num_workers=worker, 
                                pin_memory=True if torch.cuda.is_available() else False
                            )
                        else:
                            dataloader = DataLoader(
                                dataset, 
                                num_workers=worker, 
                                batch_size=None, 
                                prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
                                pin_memory=True
                            )
                        
                        print("Starting WDS I/O Benchmark...")
                        start_time = time.time()
                        
                        total_images_processed = 0
                        actual_batches = 0

                        # 3. The Benchmark Loop
                        for i, batch in enumerate(dataloader):
                            # 1. Extract the tensor from the WebDataset batch list
                            image_tensor = batch[0] # This is your [32, 39, 3, 224, 224] tensor
                            #keys = batch[1]         # This is the list of subject IDs
                            
                            # 2. NOW use .shape on the tensor, not the batch!
                            num_images_in_batch = image_tensor.shape[0] * image_tensor.shape[1]

                            total_images_processed += num_images_in_batch
                            actual_batches += 1
                            
                            '''if (i + 1) % 10 == 0:
                                print(f"Processed {i + 1} batches...")'''
                                
                            if i == batches_to_test - 1:
                                break

                        # Force synchronization if a GPU happened to be involved (though this is CPU only)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        end_time = time.time()
                        total_time = end_time - start_time
                        
                        if total_time > 0:
                            io_throughput = total_images_processed / total_time
                        else:
                            io_throughput = 0
                            
                        print(f"Total Time:           {total_time:.2f} seconds")
                        print(f"Batches Processed:    {actual_batches}")
                        print(f"Images Processed:     {total_images_processed}")
                        print(f"** I/O Throughput:    {io_throughput:.2f} images/sec **")
                        print("-" * 40)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

if __name__ == "__main__":
    #io_benchmark()
    #gpu_dummy_benchmark()
    io_benchmark_webdataset()
    