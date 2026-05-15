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

LIST_OF_IDS_PATH = "/mnt/beegfs01/scratch/a_morelli/extraction/progress.csv"
DATA_PATH = "/mnt/beegfs01/scratch/a_morelli/extraction/final/data"

resize_transform = T.Compose([
    T.Resize((224, 224)), # Change 224, 224 to whatever resolution your model expects
    T.ToTensor()
])

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

def io_benchmark():
    args = get_args()
    num_workers = args.num_workers
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
        dataloader = DataLoader(dataset, batch_size=4, num_workers=workers, shuffle=True)

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

# --- A MOCK MODEL ---
# Replace this with your actual model class
class SimpleMockModel(nn.Module):
    def __init__(self,seq_length=13*3):
        super().__init__()
        self.conv = nn.Conv2d(seq_length*3, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10) # 10 output classes
        self.seq_length = seq_length

    def forward(self, x):
        # x shape: [Batch, 6, 3, H, W]
        batch_size = x.shape[0]
        channels = x.shape[2] # Extract the channel dimension (3)
        # Flatten the time and channel dimensions: [Batch, 18, H, W]
        x = x.view(batch_size, -1, x.shape[3], x.shape[4]) 
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)
        return self.fc(x)
    
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

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

def sharded_benchmark():
    import webdataset as wds
    import torchvision.transforms as T
    import torch

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    def process_wds_sample(sample):
        """
        WebDataset passes us the dictionary we created above.
        We just need to decode the PNGs and stack them.
        """
        images = []
        # Loop through 0.png to 5.png
        for i in range(6):
            # WDS automatically decodes standard image formats to PIL images
            img_key = f"{i}.png"
            img = sample[img_key] 
            images.append(transform(img))
            
        # Stack into [6, 3, 224, 224]
        sequence_tensor = torch.stack(images)
        
        # You can return (input, target) or whatever your model expects
        return sequence_tensor, sample["json"]

    # --- THE DATALOADER ---
    # WDS handles shuffling the shards, decoding, and batching automatically
    dataset = (
        wds.WebDataset("wds_shards/dataset_shard-{000000..000050}.tar") # Load shards 0 to 50
        .shuffle(1000) # Shuffle a buffer of 1000 sequences
        .decode("pil") # Automatically decode .png bytes to PIL images
        .map(process_wds_sample) # Apply our stacking/transform logic
        .batched(32) # Group into batches of 32
    )

    # WDS doesn't use the standard PyTorch multiprocessing in the same way,
    # so you often use a standard DataLoader just to manage workers.
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None)

    for batch, metadata in dataloader:
        print(batch.shape) # Output: [32, 6, 3, 224, 224]
        break

if __name__ == "__main__":
    #io_benchmark()
    gpu_dummy_benchmark()
    