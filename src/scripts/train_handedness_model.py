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
import torch.optim as optim
from torchvision import models
import torchvision.utils as vutils
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
#from lightning.pytorch.loggers import CSVLogger

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset
from src.utils.model_utils import SimpleMockModel

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"

LIST_OF_IDS_HANDEDNESS_PATH = os.path.join(SOURCE_PATH,"handedness_model_ids.csv")
SOURCE_PATTERN = os.path.join(SOURCE_PATH,"png_resized_padded")

SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")

NUM_CLASSES = 2  
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']

MODEL='resnet18'
OUTPUT_PATH = os.path.join(SOURCE_PATH,f"{MODEL}_model_output")


def prepare_handedness_dataset(shard_pattern, decode_approach='pil',load_in_memory=False, split_workers=True, batch_size=4, transform = None, modality='X'):
    
    def select_single_modality(sample, transform=None, modality='X'):

        img_tensor = None
        label = None
        blank_image = torch.zeros(3, 224, 224)  # Assuming 3 channels and 224x224 size for ResNet18
        
        for key, value in sample.items():
            #print(f"Processing key: {key} with value type: {type(value)}")
            if key.endswith((".png", ".jpg", ".jpeg")):
                parts = key.split('.')

                #print(f"Processing key: {key} with parts: {parts}")
                #raise Exception("Debugging: Stopping after processing the first image key to check the key structure and modality matching.")

                # If it matches our target modality, process it
                if len(parts) == 3 and parts[1].lower() == modality.lower():
                    try:
                        if transform is not None:
                            img_tensor = transform(value) 
                        elif isinstance(value, torch.Tensor):
                            img_tensor = value
                        else:
                            img_tensor = T.ToTensor()(value)
                    except Exception as e:
                        print(f"Skipping corrupted image {key}: {e}")
                            
            elif key.endswith("json"):
                label = torch.tensor(value.get("label", -1))
        #raise Exception("Debugging: Stopping after processing the first sample to check the key structure, modality matching, and label extraction.")
                
        # If we are missing either the image or the label, return the filter flag (-1)
        if img_tensor is None or label is None:
            return blank_image, torch.tensor(-1)
        
        # Return the image directly. 
        # Shape will be (Channels, Height, Width) instead of (1, Channels, Height, Width)
        return img_tensor, label
    
    # 1. Use glob to find all files matching the pattern
    shard_files = glob.glob(shard_pattern)
    # Sort them just to be safe so they load in order
    shard_files.sort()


    if load_in_memory:
        return 0
    else:
        # 1. Define the base WDS Pipeline
        dataset = wds.WebDataset(shard_files, shardshuffle=100)

        # 2. Conditionally apply worker splitting
        if split_workers:
            dataset = dataset.select(wds.split_by_worker)

        # 3. Apply the remaining transformations
        dataset = (dataset
            .decode(decode_approach)
            .map(lambda sample: select_single_modality(sample, transform,modality=modality)) 
            .select(lambda sample: sample[1].item() != -1) # to filter missing data (labelled as -1)
            .batched(batch_size)
        )
    return dataset

# --- 5. Training & Validation Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 10)
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            # Backward pass + Optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_train_samples += inputs.size(0)
            
        epoch_train_loss = running_loss / total_train_samples
        epoch_train_acc = running_corrects.double() / total_train_samples
        
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total_val_samples = 0
        
        # Turn off gradients for validation to save memory and speed up
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                total_val_samples += inputs.size(0)
                
        epoch_val_loss = val_loss / total_val_samples
        epoch_val_acc = val_corrects.double() / total_val_samples
        
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    return model

class LitModel(L.LightningModule):
    def __init__(self, model, criterion, lr=1e-3,log_file=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        # Initialize attributes to store sample counts
        self.train_sample_count = 0
        self.log_file = log_file

    def forward(self, x):
        return self.model(x)
    
    # --- Epoch Start Hooks ---
    def on_train_epoch_start(self):
        # Reset the counter at the beginning of every training epoch
        self.train_sample_count = 0

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        outputs = self(inputs)

        # Accumulate the number of samples in the current batch
        self.train_sample_count += inputs.size(0)

        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data).float() / inputs.size(0)
        
        # Log metrics (Lightning tracks epoch averages automatically)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data).float() / inputs.size(0)
        
        # 'val_loss' must be logged so the callbacks can monitor it
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
    
    # --- Epoch End Hooks ---
    def on_train_epoch_end(self):
        # Print out the total collected count at the end of the training loop
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {self.current_epoch + 1}: Total Training Samples Processed: {self.train_sample_count}\n")
        else:
            print(f"\n[Epoch {self.current_epoch + 1}] Total Training Samples Processed: {self.train_sample_count}")

    def configure_optimizers(self):
        # Pass your preferred optimizer here
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

def debug_images_dataset(dataset, output_path="anteprima_dataset.png", num_immagini=16, mean=None, std=None):
    """
    Estrae un numero specifico di immagini da un WebDataset (o standard Dataset) 
    iterando sugli elementi e le salva in una griglia su file.
    """
    # 1. Creiamo un DataLoader temporaneo (batch_size=None mantiene lo streaming nativo)
    dataloader = DataLoader(
        dataset, 
        num_workers=0, 
        batch_size=None, 
        prefetch_factor=None,
    )
    
    immagini_raccolte = []
    data_iter = iter(dataloader)
    
    # 2. Iteriamo ed estraiamo campioni finché non raggiungiamo 'num_immagini'
    print(f"Raccolta di {num_immagini} immagini dal WebDataset...")
    for sample in data_iter:
        for img in sample[0]:
            if len(immagini_raccolte) < num_immagini:
                immagini_raccolte.append(img.cpu())
            else:
                break
        if len(immagini_raccolte) >= num_immagini:
            break
            
    if len(immagini_raccolte) == 0:
        print("Errore: Il dataset è vuoto o non è stato possibile estrarre immagini.")
        return

    # Se lo stream si è esaurito prima del previsto, avvisiamo l'utente
    if len(immagini_raccolte) < num_immagini:
        print(f"Nota: Trovate solo {len(immagini_raccolte)} immagini rispetto alle {num_immagini} richieste.")

    # 3. Stack delle immagini individuali in un unico batch tensor [B, C, H, W]
    immagini = torch.stack(immagini_raccolte, dim=0)
    print("Dimensione del batch di immagini raccolte:", immagini.size())

    # 4. Denormalizzazione (opzionale ma consigliata se usi transforms.Normalize)
    if mean is not None and std is not None:
        # Convertiamo in tensor con dimensioni compatibili [1, C, 1, 1] per il broadcasting su un batch
        mean_t = torch.tensor(mean).view(1, -1, 1, 1)
        std_t = torch.tensor(std).view(1, -1, 1, 1)
        # Ripristiniamo i colori originali: img * std + mean
        immagini = immagini * std_t + mean_t
    
    # Assicuriamoci che i valori siano nel range [0, 1] per il salvataggio corretto
    immagini = torch.clamp(immagini, 0.0, 1.0)

    # 5. Creiamo la cartella di destinazione se non esiste
    cartella = os.path.dirname(output_path)
    if cartella and not os.path.exists(cartella):
        os.makedirs(cartella)

    # 6. Creiamo la griglia e salviamo su file
    nrow = int(len(immagini_raccolte) ** 0.5)
    vutils.save_image(immagini, output_path, nrow=nrow, padding=2, normalize=False)
    
    print(f"Anteprima salvata con successo in: {output_path}")

if __name__ == "__main__":
    args = get_args()
    worker = args.num_workers
    batch_size = 16
    prefetch_factor = 2
    decode_approach = 'pil'
    load_in_memory = False
    split_workers = True
    transform = None

    csv_data = pd.read_csv(LIST_OF_IDS_HANDEDNESS_PATH)
    train_data = csv_data[csv_data['split'] == 'train']
    print(f"Total training samples: {len(train_data)}")
    train_data_without_X =train_data[train_data[f'q_{QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS[0]}_num_X'] >= 1]
    print(f"Training samples with at least 1 X modality: {len(train_data_without_X)}")

    
    # Automatically use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu = torch.cuda.get_device_name(device_id)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
        print(f"--- GPU DIAGNOSTICS ---")
        print(f"Active GPU: {gpu}")
        print(f"CUDA_VISIBLE_DEVICES: {visible}")

    # --- 2. Load Pre-trained ResNet18 ---
    # We fetch the best available weights trained on ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # --- 3. Modify the Final Layer ---
    # ResNet18's last layer is named 'fc'. We swap it out to match your number of classes.
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)

    #define a transform that normalize to the imagenet mean and std
    transform = T.Compose([
        #T.Resize((224, 224)),  # ResNet18 expects 224x224 input; in this case data ia already resized
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    print(f"Reading from {SHARD_PATTERN_train} and {SHARD_PATTERN_val} with decode approach '{decode_approach}' and load_in_memory={load_in_memory}")
    train_dataset = prepare_handedness_dataset(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                               split_workers=split_workers, batch_size=batch_size, 
                                               transform=transform)
    # Iterate through the first few items to see what labels are coming out
    for i, (img, label) in enumerate(train_dataset):
        print(f"Sample {i}: Label {label}")
        if i > 10: break
    #print(f"Number of dataset samples (train): {len(train_dataset)}")
    val_dataset = prepare_handedness_dataset(SHARD_PATTERN_val, decode_approach=decode_approach, load_in_memory=load_in_memory,
                                                split_workers=split_workers, batch_size=batch_size, 
                                                transform=transform)
    #sample N data points at random from the train dataset, save them in an image with the corresponding label
    debug_images_dataset(train_dataset, output_path="anteprima_dataset.png", num_immagini=16, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    #raise Exception("Debugging: Stopping after dataset preparation and image debugging. Check 'anteprima_dataset.png' for a visual preview of the data and verify labels in the console output.")

    train_loader = DataLoader(
        train_dataset, 
        num_workers=worker, 
        batch_size=None, 
        prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        num_workers=worker, 
        batch_size=None, 
        prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    lit_model = LitModel(model=model, criterion=criterion, lr=0.001, log_file=os.path.join(OUTPUT_PATH,"custom_training_log.txt"))

    # 2. Setup Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",               # Monitor validation loss
        dirpath=os.path.join(OUTPUT_PATH,"checkpoints/"),           # Directory where weights will be saved
        filename="best",            # Filename for the best model
        save_top_k=1,                     # Save only the 1 best model
        mode="min",                       # Stop when val_loss stops minimizing
        save_last=True                    # Automatically creates 'last.ckpt' every epoch
    )

    # 3. Setup Early Stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True
    )

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(OUTPUT_PATH,'tensor_board_logging/'),  # Main root folder
        name="my_experiment"         # Sub-folder for this specific project
    )
    '''
    csv_logger = CSVLogger(
        save_dir="text_logs/",
        name="experiment_1"
    )
    '''

    # 4. Initialize Trainer and Fit
    trainer = L.Trainer(
        max_epochs=50,
        logger = tb_logger,
        accelerator="auto",                # Automatically selects GPU/CPU/MPS
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    # if you want you can set anothr kind of logger (not tensorboard but csv ..)

    # This replaces your entire training loop function
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    