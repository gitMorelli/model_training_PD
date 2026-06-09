import tarfile
import time
import io
import torch
from PIL import Image, ImageOps
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
import re
import matplotlib.pyplot as plt
import numpy as np

from src.utils.data_loading_utils import prepare_handedness_dataset, prepare_handedness_dataset_all, generate_exclusion_set_val, test_handedness_dataset_all

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"

#LIST_OF_IDS_HANDEDNESS_PATH = os.path.join(SOURCE_PATH,"handedness_model_ids.csv")
LIST_OF_IDS_HANDEDNESS_PATH = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv"

#data_folder = "png_resized_padded_whitebg", "all_png_resized_padded", "all_png_whitebg" , "all_no_grids_png_whitebg" 
data_folder = "all_no_grids_png_resized_half_whitebg"
SOURCE_PATTERN = os.path.join(SOURCE_PATH,data_folder)

SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")

#QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(q) for q in range(1,14)]

SAVE_PATH = "/home/a_morelli/vscode_projects/model_training/data/dataset_info"
input_size = 224
DEBUG_IMGS = True
SEED=42
DATA_MODALITY = 'text' # 'X', 'text', 'digit'

BALANCED_DATA = True
BALANCING_FACTOR = 1
MAJORITY_CLASS_ID = 0


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

class LitModel(L.LightningModule):
    def __init__(self, write_log,model,num_0, num_1,num_classes=2, lr_backbone=1e-4,
                 lr_classifier_head=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.num_1 = num_1
        self.num_0 = num_1 * BALANCING_FACTOR if BALANCED_DATA else num_0
        self.total = self.num_0 + self.num_1

        #cross entropy
        weight_0 = self.total / (2 * self.num_0)
        weight_1 = self.total / (2 * self.num_1)
        # For BCE loss, the weight is applied to the positive class (Left-handed / Class 1)
        # Formula: pos_weight = majority_class_count / minority_class_count
        pos_weight_val = self.num_0 / self.num_1 if self.num_1 > 0 else 1.0

        # Array matching [weight_for_class_0, weight_for_class_1]
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32)
        self.register_buffer("class_weights", class_weights)
        #BCE loss
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
        self.register_buffer("pos_weight", pos_weight)
        '''
        Watch out for device mismatches: A common mistake is doing self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight])) without using register_buffer. 
        If you do that, the weight tensor stays on the CPU, and the moment your model moves to a GPU, your code will crash with a runtime device mismatch error. 
        Using self.register_buffer binds the tensor to the module's lifetime and device state seamlessly
        '''

        self.model = model

        if num_classes == 2:
            if USE_BALANCED_WEIGHTS:
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif num_classes == 1:
            if USE_BALANCED_WEIGHTS:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        
        self.num_classes = num_classes
        self.lr_backbone = lr_backbone
        self.lr_classifier_head = lr_classifier_head
        # Initialize attributes to store sample counts
        self.train_sample_count = 0
        self.write_log = write_log
        self.example_input_array = torch.randn(1, 3, 224, 224)  # For visualizing the graph in TensorBoard

    def forward(self, x):
        return self.model(x)
    
    # --- Epoch Start Hooks ---
    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.write_log(f"Device of model at start of training: {next(self.model.parameters()).device}")

        # Reset the counter at the beginning of every training epoch
        self.train_sample_count = 0

    def training_step(self, batch, batch_idx):
        inputs, labels, *_ = batch

        if self.num_classes == 1:
            labels = labels.float().unsqueeze(1)
        outputs = self(inputs)

        # Check for NaN/inf in model outputs
        if not torch.isfinite(outputs).all():
            self.write_log(f"Warning: NaN or inf detected in model outputs at step {self.trainer.global_step}.")
            # You might want to return or handle this case, e.g., by skipping the step
            return None

        # Accumulate the number of samples in the current batch
        self.train_sample_count += inputs.size(0)
        loss = self.criterion(outputs, labels)

        # Check for NaN/inf in loss
        if not torch.isfinite(loss):
            self.write_log(f"Warning: NaN or inf detected in loss at step {self.trainer.global_step}. Skipping update.")
            return None
        
        # Calculate accuracy
        if self.num_classes == 1:
            preds = (outputs > 0.0).float()
        else:
            _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data).float() / inputs.size(0)
        
        # Log metrics (Lightning tracks epoch averages automatically)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        # Log learning rates directly from the optimizer
        opt = self.optimizers()
        self.log("lr_backbone", opt.param_groups[0]['lr'], on_step=False, on_epoch=True)
        self.log("lr_classifier_head", opt.param_groups[1]['lr'], on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, *_ = batch

        if self.num_classes == 1:
            labels = labels.float().unsqueeze(1)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        
        if self.num_classes == 1:
            preds = (outputs > 0.0).float()
        else:
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
        self.write_log(f"\n[Epoch {self.current_epoch + 1}] Total Trainable Parameters: {total_trainable_params:,}")
        
        # 2. Check if it is the first epoch and write everything to your log file
        if self.current_epoch in [0,1]:
            self.write_log(f"\n--- Epoch {self.current_epoch + 1} Summary ---")
            # Your existing tracking metadata
            self.write_log(f"Epoch {self.current_epoch + 1}: Total Training Samples Processed: {self.train_sample_count}\n")
            self.write_log(f"Expected total is: {self.total}\n")
            self.write_log(f"Class 0 samples: {self.num_0}, Class 1 samples: {self.num_1}\n")
            self.write_log(f"Balancing Factor: {BALANCING_FACTOR}\n")
            self.write_log(f"Balanced Data: {BALANCED_DATA}, Use Balanced Weights: {USE_BALANCED_WEIGHTS}\n")
            self.write_log(f"Weights for Loss Function: {self.class_weights.tolist()}\n")
            
            # New: Append the model architecture specifics
            self.write_log("\n--- Trainable Model Architecture Summary ---\n")
            self.write_log(f"Total Trainable Parameters Count: {total_trainable_params:,}\n")
            self.write_log("Trainable Layers Structure:\n")
            for layer_info in trainable_layers_info:
                self.write_log(f"{layer_info}\n")

    def configure_optimizers(self):
        backbone_params = []
        head_params = []
        
        # Segregate parameters based on whether they belong to the classification head
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Adjust the string matching here depending on how 'JoinedModels' names your layers.
            # Usually, new heads are named 'classifier', 'fc', or 'head'.
            if any(key in name.lower() for key in ['classifier']):
                head_params.append(param)
            else:
                backbone_params.append(param)
                
        # Apply a 10x smaller learning rate to the backbone
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': self.lr_backbone},
            {'params': head_params, 'lr': self.lr_classifier_head}
        ])
        
        return optimizer

    def predict_step(self, batch, batch_idx):
        inputs, labels, subject_id,*_ = batch
        outputs = self(inputs)

        if self.num_classes == 1:
            preds = (outputs > 0.0).float()
            probs = torch.sigmoid(outputs)  # Get probabilities for the positive class
        else:
            _, preds = torch.max(outputs, 1)
            # Convert logits to probabilities for both classes
            probs = torch.softmax(outputs, dim=1)
        
        # Detach and move to CPU to avoid hoarding GPU memory
        return {
            "probs": probs.detach().cpu(),
            "preds": preds.detach().cpu(), 
            "labels": labels.detach().cpu(),
            "subject_id": subject_id  # Assuming subject_id is already a CPU tensor or a list of strings
        }
    
    def on_after_backward(self):
        # This hook is called after loss.backward() and before optimizer.step()
        # We check gradients only on the first batch of the first training epoch
        if self.trainer.global_step == 0:
            self.write_log("\n--- Gradient Check (First Batch) ---")
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Print the mean absolute gradient for key layers
                    if 'layer4' in name or 'classifier' in name:
                        grad_abs_mean = param.grad.abs().mean().item()
                        self.write_log(f"Layer '{name}': Mean Abs Gradient = {grad_abs_mean:.2e}")
                        if grad_abs_mean < 1e-8:
                            self.write_log(f"  -> WARNING: Potential vanishing gradient in layer {name}")
            self.write_log("-------------------------------------\n")

def save_img_with_info(image_data,properties_text,path):
    # Create a 1-row, 2-column figure layout
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1.5, 1]})

    # Column 1: Display the actual image
    axs[0].imshow(image_data, cmap='viridis')
    axs[0].set_title("Processed Sample", fontsize=14, fontweight='bold')
    axs[0].axis('off')  # Hide image axis ticks

    # Column 2: Turn off the plot lines and render the long text block
    axs[1].axis('off')
    axs[1].text(
        x=0.0, y=1.0,               # Coordinate starting point (top-left of this subplot)
        s=properties_text, 
        fontsize=11, 
        fontfamily='monospace',     # Monospace keeps alignment neat
        verticalalignment='top', 
        horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', edgecolor='gray', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')

def melt_df(df,modality):
    avail_columns=[f'q_{q}_num_{modality}' for q in QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS]
    df_source = df[['ident_projet', 'lateralite','split'] + avail_columns]
    df_long = df_source.melt(
        id_vars=['ident_projet', 'lateralite','split'], 
        value_vars=avail_columns,
        var_name='original_col', 
        value_name='score'
    )
    print(f"Length of melted df before filtering: {len(df_long)}")

    # 2. Filter rows where the score/value is >= 1
    df_long = df_long[df_long['score'] >= 1]

    print(f"Length of melted df after filtering: {len(df_long)}")

    # 3. Extract the 'q' number from the column name
    # This regex looks for 'q_' followed by digits at the start of the string
    df_long['questionnaire'] = df_long['original_col'].str.extract(r'^q_(\d+)_').astype(int)

    df_long['ident_projet'] = df_long['ident_projet'].astype(str) + '_' + df_long['questionnaire'].astype(str)

    # 4. Drop the temporary columns to get your final desired structure
    new_df = df_long[['ident_projet', 'lateralite','split']].reset_index(drop=True)

    return new_df

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

def calculate_mean_std(dataloader):
    # Running sums for channels
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    print("Calculating mean and std...")
    for data, *_ in dataloader:
        # data shape: [batch_size, channels, height, width]
        # We average over batch, height, and width (dims 0, 2, 3) to keep channel dim (1)
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    # Global mean
    mean = channels_sum / num_batches
    
    # Global standard deviation: sqrt( E[X^2] - (E[X])^2 )
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    return mean, std

def main():
    args = get_args()
    worker = args.num_workers
    batch_size = 32
    prefetch_factor = 2
    decode_approach = 'pil'
    load_in_memory = False
    split_workers = True
    transform = None
    exclusion_set = set() # you can add here the ids you want to exclude from the dataset (for example because they are corrupted or for debugging purposes)
    apply_augmentation = True
    invert_color=True
    random.seed(SEED)


    if apply_augmentation:
        #add a random crop transform without resizing
        augmentation_transform = T.Compose([
            T.RandomCrop(
                336, 
                pad_if_needed=True, 
                padding_mode='constant', 
                fill=(255, 255, 255) # <-- White fill for RGB PIL images
            )
        ])
    else:
        augmentation_transform = None
    transform = T.Compose([
        T.ToTensor(),          # Scales pixels to [0, 1]
        T.Normalize(mean=[0.06040578708052635, 0.06040578708052635, 0.06040578708052635], 
                        std=[0.23823712766170502, 0.23823712766170502, 0.23823712766170502]),
    ])

    #### EXPECTED class properties #############
    ############################################
    csv_data = pd.read_csv(LIST_OF_IDS_HANDEDNESS_PATH)
    print("Columns in the CSV:", csv_data.columns.tolist())
    csv_data = melt_df(csv_data, modality=DATA_MODALITY)
    print("CSV after melting:", csv_data.head())
    #cols: 'ident_projet', 'lateralite', 'q_5_num_X', 'q_5_num_text', 'q_5_num_digit', 'split']
    train_data = csv_data[csv_data['split'] == 'train']
    print(f"Training samples with at least 1 chunck for modality {DATA_MODALITY}: {len(train_data)}")

    #get the number of samples for each class
    class_counts = train_data['lateralite'].value_counts()
    print(f"Class distribution in training set (after filtering for modality {DATA_MODALITY}):\n{class_counts}")

    num_0 = class_counts.get(0.0, 0)  # Count of class 0 (e.g., right-handed)
    num_1 = class_counts.get(1.0, 0)  # Count of class 1 (e.g., left-handed)
    rate = num_1/num_0 if num_0 > 0 else 0
    print(f"Number of samples per class: Class 0 = {num_0}, Class 1 = {num_1}; Total = {num_0 + num_1}")
    ################################################
    ###############################################
    
    if BALANCED_DATA:
        exclusion_set = generate_exclusion_set_val(csv_data, data_modality=DATA_MODALITY,
                                                    majority_class_id=MAJORITY_CLASS_ID, balancing_factor=BALANCING_FACTOR, 
                                                    label_col='lateralite', id_col='ident_projet', split='train')
    print(len(exclusion_set), "samples will be excluded from the training set to achieve balancing.")

    #Load your dataset here
    '''train_dataset = test_handedness_dataset_all(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                            split_workers=split_workers, batch_size=batch_size, 
                                            transform=transform, modality=DATA_MODALITY, exclusion_set=exclusion_set, 
                                            huggingface_transform=False, augmentation_transform=augmentation_transform,
                                            invert_color=invert_color)'''
    train_dataset = prepare_handedness_dataset_all(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory, 
                                            split_workers=split_workers, batch_size=batch_size, 
                                            transform=transform, modality=DATA_MODALITY, exclusion_set=exclusion_set, 
                                            huggingface_transform=False, augmentation_transform=augmentation_transform,
                                            invert_color=invert_color)

    train_loader = DataLoader(
        train_dataset, 
        num_workers=worker, 
        batch_size=None, 
        prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=True
    )

    
    #select random ids or provide a list
    selected_ids = set(train_data['ident_projet'].unique())-exclusion_set
    selected_ids = random.sample(selected_ids, min(5, len(selected_ids)))
    selected_ids = [f'B4R2N6D9_{i}' for i in range(1,6)]
    #these are in the format XXX_N
    print("Selected subject IDs for inspection:", selected_ids)

    # Execute
    '''mean, std = calculate_mean_std(train_loader)

    print(f"Dataset Mean: {mean.tolist()}")
    print(f"Dataset Std:  {std.tolist()}")'''

    
    #iterate over the dataloader 
    start_time = time.time()
    n_batches = 0
    for batch_idx, batch in enumerate(train_loader):
        n_batches += 1
        img_tensor, label, subject_id_batch, questionnaire_batch, modality_type = batch
        for i,subject_id in enumerate(subject_id_batch): 
            full_id = f"{subject_id}_{questionnaire_batch[i][1:]}"
            if full_id in selected_ids:
                print(f"Batch {batch_idx}: Subject ID = {subject_id}, Label = {label[i]}, Questionnaire = {questionnaire_batch[i]}, Modality = {modality_type[i]}")
                # FIX: Index into the batch to get the single 3D image tensor [C, H, W]
                single_img = img_tensor[i] 
                
                # Convert the single 3D tensor to PIL
                img_pil = T.ToPILImage()(single_img.cpu())
                image_data = np.array(img_pil)
                
                properties_text = f"Subject ID: {subject_id}\n Label: {label[i]}\n Questionnaire: {questionnaire_batch[i]}\n Modality: {modality_type[i]}"
                save_img_with_info(image_data, properties_text, os.path.join(SAVE_PATH, f"sample_{full_id}.png"))
    end_time = time.time()
    
    print(f"Processed {n_batches} batches in {end_time - start_time:.2f} seconds -> Images per second: {n_batches * batch_size / (end_time - start_time):.2f}")
        
    
if __name__ == "__main__":
    main()
    