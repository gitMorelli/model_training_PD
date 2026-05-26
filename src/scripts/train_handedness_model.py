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
import random
import shutil
#from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset
from src.utils.model_utils import SimpleMockModel

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"

LIST_OF_IDS_HANDEDNESS_PATH = os.path.join(SOURCE_PATH,"handedness_model_ids.csv")
SOURCE_PATTERN = os.path.join(SOURCE_PATH,"png_resized_padded")

SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")


QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']

MODEL = 'resnet18'
TEST = 'balanced_data_and_loss' #'balanced_loss', 'balanced_data'
EXPERIMENT_NAME = f"{MODEL}_{TEST}"
OUTPUT_PATH = os.path.join(SOURCE_PATH,f"{MODEL}_model_results")
TRAIN = False  # Set to False to skip training and only run validation evaluation
CHECKPOINT_PATH = os.path.join(OUTPUT_PATH, "checkpoints")
checkpoint_to_load='v_1/best-epoch=00-val_loss=0.61.ckpt'#best.ckpt , None last.ckpt
DEBUG_IMGS = True
GET_STATISTICS = False
SEED=42
DATA_MODALITY = 'X' # 'X', 'text', 'digit'
MODEL_MODALITY = 'full' # 'full', 'feature_ext', 'partial_unfr' 

BALANCED_DATA = True
USE_BALANCED_WEIGHTS = True
BALANCING_FACTOR = 5
MAJORITY_CLASS_ID = 0

def prepare_handedness_dataset(shard_pattern, decode_approach='pil',load_in_memory=False, split_workers=True, 
                               batch_size=4, transform = None, modality='X',rate=1, balanced_data=False):
    def filter(sample):
        # 'sample' is a dictionary. 
        # Assumes you have decoded the class label (e.g., via .cls or custom key)
        label = sample[1].item()  # Adjust this if your label is stored differently
        
        if label == -1:
            return False  # Filter out samples with missing labels
        elif label == MAJORITY_CLASS_ID and balanced_data:
            # Keep only 20% of the majority class samples
            return random.random() < BALANCING_FACTOR*rate  # Adjust the rate as needed (e.g., 0.2 for 20%)
        # Always keep minority classes
        return True
    def select_single_modality(sample, transform=None, modality='X'):
        if modality == 'X':
            modality_string = 'X'
        elif modality == 'text':
            modality_string = 'hand'
        elif modality == 'digit':
            modality_string = 'number_random'

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
                if len(parts) == 3 and parts[1].lower() == modality_string.lower():
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
                label = torch.tensor(value.get("label", -1), dtype=torch.long)
        #raise Exception("Debugging: Stopping after processing the first sample to check the key structure, modality matching, and label extraction.")
                
        # If we are missing either the image or the label, return the filter flag (-1)
        if img_tensor is None or label is None:
            return blank_image, torch.tensor(-1, dtype=torch.long)
        
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
            .select(filter) # to filter missing data (labelled as -1)
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
    def __init__(self, model,num_0, num_1, lr=1e-3,log_file=None):
        super().__init__()
        self.save_hyperparameters()
        if BALANCED_DATA:
            self.num_1 = num_1
            self.num_0 = num_1*BALANCING_FACTOR 
        else:
            self.num_1 = num_1
            self.num_0 = num_0
        self.total = self.num_0 + self.num_1
        weight_0 = self.total / (2 * self.num_0)
        weight_1 = self.total / (2 * self.num_1)

        # Array matching [weight_for_class_0, weight_for_class_1]
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)
        '''
        Watch out for device mismatches: A common mistake is doing self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight])) without using register_buffer. 
        If you do that, the weight tensor stays on the CPU, and the moment your model moves to a GPU, your code will crash with a runtime device mismatch error. 
        Using self.register_buffer binds the tensor to the module's lifetime and device state seamlessly
        '''

        self.model = model
        if USE_BALANCED_WEIGHTS:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.lr = lr
        # Initialize attributes to store sample counts
        self.train_sample_count = 0
        self.log_file = log_file
        self.example_input_array = torch.randn(1, 3, 224, 224)  # For visualizing the graph in TensorBoard

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
        # 1. Identify trainable layers and calculate their sizes
        trainable_layers_info = []
        total_trainable_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # param.shape gives the tensor dimensions (e.g., [512, 2])
                # param.numel() gives the total number of scalar elements in that tensor
                layer_str = f"  - {name} | Shape: {list(param.shape)} | Parameters: {param.numel():,}"
                trainable_layers_info.append(layer_str)
                total_trainable_params += param.numel()

        # Optional: Print to the terminal console so you can see it live during execution
        print(f"\n[Epoch {self.current_epoch + 1}] Total Trainable Parameters: {total_trainable_params:,}")
        
        # 2. Check if it is the first epoch and write everything to your log file
        if self.log_file and self.current_epoch == 0:
            with open(self.log_file, 'w') as f:
                # Your existing tracking metadata
                f.write(f"Epoch {self.current_epoch + 1}: Total Training Samples Processed: {self.train_sample_count}\n")
                f.write(f"Expected total is: {self.total}\n")
                f.write(f"Class 0 samples: {self.num_0}, Class 1 samples: {self.num_1}\n")
                f.write(f"Balancing Factor: {BALANCING_FACTOR}\n")
                f.write(f"Balanced Data: {BALANCED_DATA}, Use Balanced Weights: {USE_BALANCED_WEIGHTS}\n")
                f.write(f"Weights for Loss Function: {self.class_weights.tolist()}\n")
                
                # New: Append the model architecture specifics
                f.write("\n--- Trainable Model Architecture Summary ---\n")
                f.write(f"Total Trainable Parameters Count: {total_trainable_params:,}\n")
                f.write("Trainable Layers Structure:\n")
                for layer_info in trainable_layers_info:
                    f.write(f"{layer_info}\n")

    def configure_optimizers(self):
        trainable_parameters = filter(lambda p: p.requires_grad, self.model.parameters()) 
        
        return torch.optim.Adam(trainable_parameters, lr=self.lr)

    def predict_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        _, preds = torch.max(outputs, 1)

        # Convert logits to probabilities for both classes
        probs = torch.softmax(outputs, dim=1)
        
        # Detach and move to CPU to avoid hoarding GPU memory
        return {
            "probs": probs.detach().cpu(),
            "preds": preds.detach().cpu(), 
            "labels": labels.detach().cpu()
        }

class BestMetricsTracker(Callback):
    def __init__(self):
        super().__init__()
        self.best_epoch = None
        self.best_val_loss = None
        self.best_val_acc = None
        self.best_train_acc = None

    def on_validation_end(self, trainer, pl_module):
        # Ignore the preliminary sanity check run
        if trainer.sanity_checking:
            return

        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback and checkpoint_callback.best_model_score is not None:
            # Grab the current epoch's value of the monitored metric
            current_score = trainer.callback_metrics.get(checkpoint_callback.monitor)
            
            # If the current score matches the best score on record, snapshot everything
            if current_score == checkpoint_callback.best_model_score:
                self.best_epoch = trainer.current_epoch
                self.best_val_loss = trainer.callback_metrics.get("val_loss").item() if "val_loss" in trainer.callback_metrics else None
                self.best_val_acc = trainer.callback_metrics.get("val_acc").item() if "val_acc" in trainer.callback_metrics else None
                self.best_train_acc = trainer.callback_metrics.get("train_acc").item() if "train_acc" in trainer.callback_metrics else None

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
    #for the first image in immagini get the max and min value and print them
    print(f"Valori pixel prima del clamp: min={immagini.min().item():.4f}, max={immagini.max().item():.4f}")
    
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


def main():
    args = get_args()
    worker = args.num_workers
    batch_size = 16
    prefetch_factor = 2
    decode_approach = 'pil'
    load_in_memory = False
    split_workers = True
    transform = None
    lr=1e-4
    num_classes=2
    num_epochs=50
    patience = 5

    csv_data = pd.read_csv(LIST_OF_IDS_HANDEDNESS_PATH)
    print("Columns in the CSV:", csv_data.columns.tolist())
    #cols: 'ident_projet', 'lateralite', 'q_5_num_X', 'q_5_num_text', 'q_5_num_digit', 'split']
    train_data = csv_data[csv_data['split'] == 'train']
    print(f"Total training samples: {len(train_data)}")
    train_data_without_X =train_data[train_data[f'q_{QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS[0]}_num_{DATA_MODALITY}'] >= 1]
    print(f"Training samples with at least 1 chunck for modality {DATA_MODALITY}: {len(train_data_without_X)}")

    #get the number of samples for each class
    class_counts = train_data_without_X['lateralite'].value_counts()
    print(f"Class distribution in training set (after filtering for modality {DATA_MODALITY}):\n{class_counts}")

    num_0 = class_counts.get(0.0, 0)  # Count of class 0 (e.g., right-handed)
    num_1 = class_counts.get(1.0, 0)  # Count of class 1 (e.g., left-handed)
    rate = num_1/num_0 if num_0 > 0 else 0

    if GET_STATISTICS:
        #lateralite, X filtered
        #0.0    39271
        #1.0     3370
        print("Statistics requested. Stopping after data analysis and class distribution check.")
        return 

    #read the current version number (starts from 1)
    current_version=1
    #if it exists open the log file in CHECKPOINT_PATH, else create it 
    log_path = os.path.join(CHECKPOINT_PATH,"experiments_log.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        #get the maximum version number in the log file and add 1 to it for the current experiment
        current_version = df['current_version'].max() + 1

    # Automatically use GPU if available
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu = torch.cuda.get_device_name(device_id)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
        print(f"--- GPU DIAGNOSTICS ---")
        print(f"Active GPU: {gpu}")
        print(f"CUDA_VISIBLE_DEVICES: {visible}")

    # 1. Load the pre-trained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 2. Modify the final layer first
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Determine layers to unfreeze based on modality
    if MODEL_MODALITY == 'feature_ext':
        list_layers = ['fc']
    elif MODEL_MODALITY == 'partial_unfr':
        list_layers = ['layer4', 'fc']
    else:
        list_layers = None  # Flag to indicate EVERYTHING should be trainable

    # 3. Dynamic Selective Freezing Loop
    for name, param in model.named_parameters():
        # If list_layers is None, the 'or' statement short-circuits and sets True for all parameters
        if list_layers is None or any(layer in name for layer in list_layers):
            param.requires_grad = True
        else:
            param.requires_grad = False

    #define a transform that normalize to the imagenet mean and std
    transform = T.Compose([
        #T.Resize((224, 224)),  # ResNet18 expects 224x224 input; in this case data ia already resized
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    print(f"Reading from {SHARD_PATTERN_train} and {SHARD_PATTERN_val} with decode approach '{decode_approach}' and load_in_memory={load_in_memory}")
    if TRAIN:
        train_dataset = prepare_handedness_dataset(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                                split_workers=split_workers, batch_size=batch_size, 
                                                transform=transform, modality=DATA_MODALITY, rate=rate, balanced_data=BALANCED_DATA)
    if DEBUG_IMGS and TRAIN:
        # Iterate through the first few items to see what labels are coming out
        for i, (img, label) in enumerate(train_dataset):
            print(f"Sample {i}: Label {label}")
            if i > 10: break
    #print(f"Number of dataset samples (train): {len(train_dataset)}")
    val_dataset = prepare_handedness_dataset(SHARD_PATTERN_val, decode_approach=decode_approach, load_in_memory=load_in_memory,
                                                split_workers=split_workers, batch_size=batch_size, 
                                                transform=transform, modality=DATA_MODALITY, balanced_data=False)
    if DEBUG_IMGS and TRAIN:
        #sample N data points at random from the train dataset, save them in an image with the corresponding label
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        debug_images_dataset(train_dataset, output_path="data/anteprima_dataset.png", num_immagini=16, mean=None, std=None)

    #raise Exception("Debugging: Stopping after dataset preparation and image debugging. Check 'anteprima_dataset.png' for a visual preview of the data and verify labels in the console output.")

    if TRAIN:
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

    #criterion = nn.CrossEntropyLoss()
    log_folder = os.path.join(OUTPUT_PATH,"custom_logs")
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    lit_model = LitModel(model=model, num_0=num_0, num_1=num_1, lr=lr, 
                         log_file=os.path.join(log_folder,f"version_{current_version}_training_log.txt"))


    # 2. Setup Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",               # Monitor validation loss
        dirpath=os.path.join(CHECKPOINT_PATH,f'v_{current_version}'),           # Directory where weights will be saved
        filename="best-{epoch:02d}-{val_loss:.2f}",            # Filename for the best model
        save_top_k=1,                     # Save only the 1 best model
        mode="min",                       # Stop when val_loss stops minimizing
        save_last=True ,                   # Automatically creates 'last.ckpt' every epoch
        #enable_version_counter=False
    )

    # 3. Setup Early Stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True
    )

    metrics_tracker = BestMetricsTracker()

    if TRAIN:
        # 1. Construct clean paths (removed trailing slash, kept version as a string/int)
        log_root = os.path.join(SOURCE_PATH, 'tensor_board_logging')
        version_dir = os.path.join(log_root, MODEL, 'version_'+str(current_version)) #tensorboard automatically adds 'version_' prefix, 
        #so we match that format here.

        # 2. Wipe the old folder if it exists
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)

        # 3. Initialize the logger
        tb_logger = TensorBoardLogger(
            save_dir=log_root,
            name=MODEL,
            log_graph=True,
            version=current_version  # Works perfectly as an integer or string
        )
    else:
        tb_logger=False
    '''
    csv_logger = CSVLogger(
        save_dir="text_logs/",
        name="experiment_1"
    )
    '''

    # 4. Initialize Trainer and Fit
    trainer = L.Trainer(
        max_epochs=num_epochs,
        logger = tb_logger,
        accelerator="auto",                # Automatically selects GPU/CPU/MPS
        callbacks=[checkpoint_callback, early_stop_callback, metrics_tracker]
    )
    # if you want you can set anothr kind of logger (not tensorboard but csv ..)

    if TRAIN:
        # This replaces your entire training loop function
        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        #save experiment parameters in a dict 
        #generate timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        #get the best_epoch the model was saved at, the best val_loss and val_acc
        
        experiment_params = {
            "timestamp": timestamp,
            "MODEL": MODEL,
            "TEST": TEST,
            "EXPERIMENT_NAME": EXPERIMENT_NAME,
            "DATA_MODALITY": DATA_MODALITY,
            "MODEL_MODALITY": MODEL_MODALITY,
            "BALANCED_DATA": BALANCED_DATA,
            "USE_BALANCED_WEIGHTS": USE_BALANCED_WEIGHTS,
            "NUM_0": num_0,
            "NUM_1": num_1,
            "RATE": rate,
            "BALANCING_FACTOR": BALANCING_FACTOR,
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "patience": patience,
            "current_version": current_version,
            "best_epoch": metrics_tracker.best_epoch,
            "best_val_loss": metrics_tracker.best_val_loss,
            "best_val_acc": metrics_tracker.best_val_acc,
            "best_train_acc": metrics_tracker.best_train_acc,
        }
        #if it exists open the log file in CHECKPOINT_PATH, else create it 
        log_path = os.path.join(CHECKPOINT_PATH,"experiments_log.csv")
        if not os.path.exists(log_path):
            #save the experiment_parameters converting the dict to a dataframe and then to csv
            df = pd.DataFrame([experiment_params])
            df.to_csv(log_path, index=False)
        else:
            df = pd.read_csv(log_path)
            #get the maximum version number in the log file and add 1 to it for the current experiment
            df = df.append(experiment_params, ignore_index=True)
            df.to_csv(log_path, index=False)
    else:
        print("\n--- Starting Validation Evaluation ---")
    
        # 1. Gather predictions using the best checkpoint saved during training
        # Setting ckpt_path="best" tells Lightning to automatically find your top model
        ckpt_path=os.path.join(CHECKPOINT_PATH,checkpoint_to_load) 
        # 3. Load the checkpoint file (weights are skipped, only reading metadata)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # 4. Extract the exact epoch
        best_epoch = checkpoint["epoch"]
        print(f"The best model was saved at epoch: {best_epoch}")
        outputs = trainer.predict(lit_model, dataloaders=val_loader,ckpt_path = ckpt_path)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
        
        # 2. Concatenate all batch outputs into unified tensors
        all_probs = torch.cat([batch["probs"] for batch in outputs])
        all_preds = torch.cat([batch["preds"] for batch in outputs])
        all_labels = torch.cat([batch["labels"] for batch in outputs])

        #show 10 example outputs and the corresponding expected label
        print("\n--- Sample Predictions ---")
        for i in range(10):
            print(f"Probs: {all_probs[i].tolist()} | Predicted: {all_preds[i].item()} | True Label: {all_labels[i].item()}")
        
        # 3. Convert to numpy arrays for statistics calculation
        y_pred = all_preds.numpy()
        y_true = all_labels.numpy()
        
        # 4. Compute and display metrics using scikit-learn
        from sklearn.metrics import classification_report, confusion_matrix
        
        print("\n================ VALIDATION STATISTICS ================")
        print("\n--- Classification Report ---")
        # Adjust target_names to match your two classes if needed
        print(classification_report(y_true, y_pred, target_names=["Right", "Left"]))
        
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_true, y_pred))
        print("=======================================================")
if __name__ == "__main__":
    main()
    