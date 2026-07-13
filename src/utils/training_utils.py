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
from torchmetrics.classification import MulticlassRecall
import random
import shutil
#from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback

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
    def __init__(self, write_log,model,num_0, num_1,num_classes=2, lr_backbone=1e-4,
                 lr_classifier_head=1e-3,example_input_array=torch.randn(1, 3, 224, 224),
                 opt_groups=None,num_epochs=10,lr_scheduling='cosine',
                 balancing_factor=1.0, balanced_data=False, use_balanced_weights=True, weight_decay=1e-4, warmup_fraction=0.1, 
                 eta_min_cosine=1e-6, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.opt_groups = opt_groups

        self.num_1 = num_1
        self.num_0 = num_1 * balancing_factor if balanced_data else num_0
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
            if use_balanced_weights:
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif num_classes == 1:
            if use_balanced_weights:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        
        self.num_classes = num_classes
        self.lr_backbone = lr_backbone
        self.lr_classifier_head = lr_classifier_head
        # Initialize attributes to store sample counts
        self.train_sample_count = 0
        self.write_log = write_log
        self.example_input_array = example_input_array
        self.lr_scheduling = lr_scheduling
        self.num_epochs = num_epochs
        self.total_steps = int(self.num_epochs * (self.total//batch_size))

        n = 2 if self.num_classes == 1 else self.num_classes
        self.val_balanced_acc = MulticlassRecall(num_classes=n, average="macro")

        self.balancing_factor = balancing_factor
        self.balanced_data = balanced_data
        self.use_balanced_weights = use_balanced_weights
        self.weight_decay = weight_decay
        self.warmup_fraction = warmup_fraction
        self.eta_min_cosine = eta_min_cosine
        self.batch_size= batch_size

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
        if self.opt_groups:
            for i, group in enumerate(self.opt_groups):
                self.log(group['lr_name'], opt.param_groups[i]['lr'], on_step=False, on_epoch=True)
        else:
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

        # Shape/dtype for the metric: flat int tensors of class indices
        preds_int = preds.long().view(-1)
        targets_int = labels.long().view(-1)
        self.val_balanced_acc.update(preds_int, targets_int)
        
        # 'val_loss' must be logged so the callbacks can monitor it
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        # Pass the metric object so Lightning computes it at epoch end and resets it
        self.log("val_balanced_acc", self.val_balanced_acc, on_epoch=True, prog_bar=True)
    
    # --- Epoch End Hooks ---
    def on_train_epoch_end(self):
        # ANSI color codes
        WHITE = "\033[97m"
        RED = "\033[91m"
        RESET = "\033[0m"

        # 1. Walk every parameter in model order, coloring each line by trainable status
        all_layers_info = []
        total_trainable_params = 0
        total_non_trainable_params = 0

        for name, param in self.model.named_parameters():
            # param.shape gives tensor dimensions (e.g. [512, 2])
            # param.numel() gives the total number of scalar elements
            layer_str = f"  - {name} | Shape: {list(param.shape)} | Parameters: {param.numel():,}"

            if param.requires_grad:
                # White = trainable
                all_layers_info.append(f"{WHITE}{layer_str}{RESET}")
                total_trainable_params += param.numel()
            else:
                # Red = frozen / non-trainable
                all_layers_info.append(f"{RED}{layer_str}{RESET}")
                total_non_trainable_params += param.numel()

        total_params = total_trainable_params + total_non_trainable_params

        # Log the counts
        self.write_log(
            f"\n[Epoch {self.current_epoch + 1}] "
            f"Total Parameters: {total_params:,} | "
            f"{WHITE}Trainable: {total_trainable_params:,}{RESET} | "
            f"{RED}Non-trainable: {total_non_trainable_params:,}{RESET}"
        )

        # 2. On the first epochs, write the full summary to the log file
        if self.current_epoch in [0, 5]:
            self.write_log(f"\n--- Epoch {self.current_epoch + 1} Summary ---")
            self.write_log(f"Expected number of stepping batches: {self.trainer.estimated_stepping_batches}")
            # Your existing tracking metadata
            self.write_log(f"Epoch {self.current_epoch + 1}: Total Training Samples Processed: {self.train_sample_count}\n")
            self.write_log(f"Expected total is: {self.total}\n")
            self.write_log(f"Class 0 samples: {self.num_0}, Class 1 samples: {self.num_1}\n")
            self.write_log(f"Balancing Factor: {self.balancing_factor}\n")
            self.write_log(f"Balanced Data: {self.balanced_data}, Use Balanced Weights: {self.use_balanced_weights}\n")
            self.write_log(f"Weights for Loss Function: {self.class_weights.tolist()}\n")

            # Model architecture specifics
            self.write_log("\n--- Model Architecture Summary ---\n")
            self.write_log(f"Total Parameters Count: {total_params:,}\n")
            self.write_log(f"Total Trainable Parameters Count: {total_trainable_params:,}\n")
            self.write_log(f"Total Non-trainable Parameters Count: {total_non_trainable_params:,}\n")

            self.write_log("Model Layers Structure (white = trainable, red = frozen):\n")
            for layer_info in all_layers_info:
                self.write_log(f"{layer_info}\n")

    def configure_optimizers(self):

        def split_decay(named_params):
            """(decay, no_decay): exclude bias and 1-D (BatchNorm/LayerNorm) params from weight decay."""
            decay, no_decay = [], []
            for name, p in named_params:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        def add_group(named, lr):
            decay, no_decay = split_decay(named)
            if decay:
                param_groups.append({'params': decay,    'lr': lr, 'weight_decay': self.weight_decay})
            if no_decay:
                param_groups.append({'params': no_decay, 'lr': lr, 'weight_decay': 0.0})

        param_groups = []

        if self.opt_groups:
            for group in self.opt_groups:
                named = [(name, param) for name, param in self.model.named_parameters()
                        if any(key in name.lower() for key in group['names']) and param.requires_grad]
                add_group(named, group['lr'])
        else:
            backbone, head = [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(key in name.lower() for key in ['classifier']):
                    head.append((name, param))
                else:
                    backbone.append((name, param))
            add_group(backbone, self.lr_backbone)
            add_group(head, self.lr_classifier_head)

        optimizer = optim.AdamW(param_groups)   # weight_decay now lives per-group

        if self.lr_scheduling == 'cosine':
            # total optimizer steps for the whole run; accounts for grad accumulation & devices
            #total_steps = int(self.trainer.estimated_stepping_batches)
            total_steps = self.total_steps
            warmup_steps = max(1, int(self.warmup_fraction * total_steps))

            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-2,        # start each group at 1% of its base LR
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=self.eta_min_cosine,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",      # was "epoch"; warmup/cosine now advance per step
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    def predict_step(self, batch, batch_idx):
        inputs, labels, subject_id, questionnaire,*_ = batch
        outputs = self(inputs)

        if self.num_classes == 1:
            p1 = torch.sigmoid(outputs).flatten()        # P(class 1), (B,)
            probs = torch.stack([1 - p1, p1], dim=1)     # (B, 2): [P(0), P(1)]
            preds = (outputs > 0.0).long().flatten()
        else:
            _, preds = torch.max(outputs, 1)
            # Convert logits to probabilities for both classes
            probs = torch.softmax(outputs, dim=1)
        
        # Detach and move to CPU to avoid hoarding GPU memory
        return {
            "probs": probs.detach().cpu(),
            "preds": preds.detach().cpu(), 
            "labels": labels.detach().cpu().flatten(),
            "subject_id": subject_id,  # Assuming subject_id is already a CPU tensor or a list of strings
            "questionnaire": questionnaire
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

class ModelPD(L.LightningModule):
    def __init__(self, write_log,model,num_0, num_1,num_classes=2, lr_backbone=1e-4,
                 lr_classifier_head=1e-3,example_input_array=torch.randn(1, 3, 224, 224),
                 opt_groups=None,num_epochs=10,lr_scheduling='cosine',
                 balancing_factor=1.0, balanced_data=False, use_balanced_weights=True, weight_decay=1e-4, warmup_fraction=0.1, 
                 eta_min_cosine=1e-6, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.opt_groups = opt_groups

        self.num_1 = num_1
        self.num_0 = num_0
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
            if use_balanced_weights:
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif num_classes == 1:
            if use_balanced_weights:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        
        self.num_classes = num_classes
        self.lr_backbone = lr_backbone
        self.lr_classifier_head = lr_classifier_head
        # Initialize attributes to store sample counts
        self.train_sample_count = 0
        self.write_log = write_log
        self.example_input_array = example_input_array
        self.lr_scheduling = lr_scheduling
        self.num_epochs = num_epochs
        self.total_steps = int(self.num_epochs * (self.total//batch_size))

        n = 2 if self.num_classes == 1 else self.num_classes
        self.val_balanced_acc = MulticlassRecall(num_classes=n, average="macro")

        self.balancing_factor = balancing_factor
        self.balanced_data = balanced_data
        self.use_balanced_weights = use_balanced_weights
        self.weight_decay = weight_decay
        self.warmup_fraction = warmup_fraction
        self.eta_min_cosine = eta_min_cosine
        self.batch_size= batch_size

        self.train_class_0_count = 0
        self.train_class_1_count = 0
        self.val_class_0_count = 0
        self.val_class_1_count = 0

    def forward(self, frames, seq_ids, slot_ids, lengths):
        return self.model(frames, seq_ids, slot_ids, lengths)
    
    @staticmethod
    def make_example_input(k, n_slots, n_views_frames=2, C=3, H=224, W=224):
        T = min(n_views_frames, n_slots)
        frames   = torch.randn(T, k, C, H, W)
        seq_ids  = torch.zeros(T, dtype=torch.long)           # all one subject
        slot_ids = torch.arange(T, dtype=torch.long)          # first T slots present
        lengths  = torch.tensor([T])
        return (frames, seq_ids, slot_ids, lengths)

    # --- Epoch Start Hooks ---
    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.write_log(f"Device of model at start of training: {next(self.model.parameters()).device}")

        # Reset the counter at the beginning of every training epoch
        self.train_sample_count = 0

    def training_step(self, batch, batch_idx):
        frames, seq_ids, slot_ids, lengths, labels, *_ = batch
        bsz = labels.size(0)                      # subjects in batch, NOT frames

        if self.num_classes == 1:
            labels = labels.float().unsqueeze(1)
        outputs = self(frames, seq_ids, slot_ids, lengths)

        if not torch.isfinite(outputs).all():
            self.write_log(f"Warning: NaN/inf in outputs at step {self.trainer.global_step}.")
            return None

        self.train_sample_count += bsz            # was inputs.size(0)
        loss = self.criterion(outputs, labels)

        if not torch.isfinite(loss):
            self.write_log(f"Warning: NaN/inf in loss at step {self.trainer.global_step}. Skipping.")
            return None

        if self.num_classes == 1:
            preds = (outputs > 0.0).float()
        else:
            _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data).float() / bsz   # was / inputs.size(0)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log("train_acc",  acc,  on_epoch=True, prog_bar=True, batch_size=bsz)
        '''
        The batch_size=bsz on self.log calls matters here: Lightning infers batch size from the first tensor in the batch to weight epoch averages, 
        and it would otherwise pick up frames (size N), skewing your epoch-mean loss/acc toward batches that happen to have more frames. 
        Passing batch_size explicitly fixes the weighting
        '''

        opt = self.optimizers()
        if self.opt_groups:
            logged = set()
            for g in opt.param_groups:
                name = g.get('lr_name')
                if name and name not in logged:
                    self.log(name, g['lr'], on_step=False, on_epoch=True)
                    logged.add(name)
        else:
            self.log("lr_backbone",        opt.param_groups[0]['lr'], on_step=False, on_epoch=True)
            self.log("lr_classifier_head", opt.param_groups[1]['lr'], on_step=False, on_epoch=True)
        
        # Accumulate class counts
        targets_int = labels.long().view(-1)
        self.train_class_0_count += (targets_int == 0).sum().item()
        self.train_class_1_count += (targets_int == 1).sum().item()

        return loss

    def validation_step(self, batch, batch_idx):
        frames, seq_ids, slot_ids, lengths, labels, *_ = batch
        bsz = labels.size(0)

        if self.num_classes == 1:
            labels = labels.float().unsqueeze(1)
        outputs = self(frames, seq_ids, slot_ids, lengths)
        loss = self.criterion(outputs, labels)

        if self.num_classes == 1:
            preds = (outputs > 0.0).float()
        else:
            _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == labels.data).float() / bsz

        preds_int   = preds.long().view(-1)
        targets_int = labels.long().view(-1)
        self.val_balanced_acc.update(preds_int, targets_int)

        # Accumulate class counts
        targets_int = labels.long().view(-1)
        self.val_class_0_count += (targets_int == 0).sum().item()
        self.val_class_1_count += (targets_int == 1).sum().item()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log("val_acc",  acc,  on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log("val_balanced_acc", self.val_balanced_acc, on_epoch=True, prog_bar=True, batch_size=bsz)
    
    # --- Epoch End Hooks ---
    def on_train_epoch_end(self):
        # ANSI color codes
        WHITE = "\033[97m"
        RED = "\033[91m"
        RESET = "\033[0m"

        # 1. Walk every parameter in model order, coloring each line by trainable status
        all_layers_info = []
        total_trainable_params = 0
        total_non_trainable_params = 0

        for name, param in self.model.named_parameters():
            # param.shape gives tensor dimensions (e.g. [512, 2])
            # param.numel() gives the total number of scalar elements
            layer_str = f"  - {name} | Shape: {list(param.shape)} | Parameters: {param.numel():,}"

            if param.requires_grad:
                # White = trainable
                all_layers_info.append(f"{WHITE}{layer_str}{RESET}")
                total_trainable_params += param.numel()
            else:
                # Red = frozen / non-trainable
                all_layers_info.append(f"{RED}{layer_str}{RESET}")
                total_non_trainable_params += param.numel()

        total_params = total_trainable_params + total_non_trainable_params

        # Log the counts
        self.write_log(
            f"\n[Epoch {self.current_epoch + 1}] "
            f"Total Parameters: {total_params:,} | "
            f"{WHITE}Trainable: {total_trainable_params:,}{RESET} | "
            f"{RED}Non-trainable: {total_non_trainable_params:,}{RESET}"
        )

        # 2. On the first epochs, write the full summary to the log file
        if self.current_epoch in [0, 5]:
            self.write_log(f"\n--- Epoch {self.current_epoch + 1} Summary ---")
            self.write_log(f"Expected number of stepping batches: {self.trainer.estimated_stepping_batches}")
            # Your existing tracking metadata
            self.write_log(f"Epoch {self.current_epoch + 1}: Total Training Samples Processed: {self.train_sample_count}\n")
            self.write_log(f"Expected total is: {self.total}\n")
            self.write_log(f"Class 0 samples: {self.num_0}, Class 1 samples: {self.num_1}\n")
            self.write_log(f"Balancing Factor: {self.balancing_factor}\n")
            self.write_log(f"Balanced Data: {self.balanced_data}, Use Balanced Weights: {self.use_balanced_weights}\n")
            self.write_log(f"Weights for Loss Function: {self.class_weights.tolist()}\n")

            # Model architecture specifics
            self.write_log("\n--- Model Architecture Summary ---\n")
            self.write_log(f"Total Parameters Count: {total_params:,}\n")
            self.write_log(f"Total Trainable Parameters Count: {total_trainable_params:,}\n")
            self.write_log(f"Total Non-trainable Parameters Count: {total_non_trainable_params:,}\n")

            self.write_log("Model Layers Structure (white = trainable, red = frozen):\n")
            for layer_info in all_layers_info:
                self.write_log(f"{layer_info}\n")
        
        #class counts
        n0 = self.train_class_0_count
        n1 = self.train_class_1_count
        ratio = n0 / n1 if n1 > 0 else float("inf")
        self.log("train_class_ratio", ratio, prog_bar=False)

        # Reset for next epoch
        self.train_class_0_count = 0
        self.train_class_1_count = 0

    def on_validation_epoch_end(self):
        n0 = self.val_class_0_count
        n1 = self.val_class_1_count
        ratio = n0 / n1 if n1 > 0 else float("inf")

        #print(f"\n[Epoch {self.current_epoch}] Val class ratio  →  class0: {n0}  |  class1: {n1}  |  0/1 ratio: {ratio:.3f}")
        self.log("val_class_ratio", ratio, prog_bar=False)

        # Reset for next epoch
        self.val_class_0_count = 0
        self.val_class_1_count = 0

    def configure_optimizers(self):

        def split_decay(named_params):
            """(decay, no_decay): exclude bias and 1-D (BatchNorm/LayerNorm) params from weight decay."""
            decay, no_decay = [], []
            for name, p in named_params:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(p)
                else:
                    decay.append(p)
            return decay, no_decay

        def add_group(named, lr, lr_name):
            decay, no_decay = split_decay(named)
            if decay:
                param_groups.append({'params': decay,    'lr': lr, 'weight_decay': self.weight_decay, 'lr_name': lr_name})
            if no_decay:
                param_groups.append({'params': no_decay, 'lr': lr, 'weight_decay': 0.0, 'lr_name': lr_name})

        param_groups = []

        if self.opt_groups:
            for group in self.opt_groups:
                named = [(name, param) for name, param in self.model.named_parameters()
                        if any(key in name.lower() for key in group['names']) and param.requires_grad]
                add_group(named, group['lr'], group['lr_name'])
        else:
            backbone, head = [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(key in name.lower() for key in ['classifier']):
                    head.append((name, param))
                else:
                    backbone.append((name, param))
            add_group(backbone, self.lr_backbone, 'lr_backbone')
            add_group(head, self.lr_classifier_head, 'lr_head')

        optimizer = optim.AdamW(param_groups)   # weight_decay now lives per-group

        if self.lr_scheduling == 'cosine':
            # total optimizer steps for the whole run; accounts for grad accumulation & devices
            #total_steps = int(self.trainer.estimated_stepping_batches)
            total_steps = self.total_steps
            warmup_steps = max(1, int(self.warmup_fraction * total_steps))

            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-2,        # start each group at 1% of its base LR
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=self.eta_min_cosine,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",      # was "epoch"; warmup/cosine now advance per step
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    def predict_step(self, batch, batch_idx):

        frames, seq_ids, slot_ids, lengths, labels, \
            resizing_factors, subject_ids, modalities = batch
        

        outputs = self(frames, seq_ids, slot_ids, lengths)

        if self.num_classes == 1:
            p1 = torch.sigmoid(outputs).flatten()
            probs = torch.stack([1 - p1, p1], dim=1)
            preds = (outputs > 0.0).long().flatten()
        else:
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

        return {
            "probs": probs.detach().cpu(),
            "preds": preds.detach().cpu(),
            "labels": labels.detach().cpu().flatten(),
            "subject_ids": subject_ids,
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

#------ Loss functions ---------------
def grouped_ce_loss(scores, labels, group_ids):
    """scores (B,), labels (B,) with exactly one 1 per group, group_ids (B,) in 0..G-1.
    Conditional logistic / within-group softmax NLL: mean over groups of
    -score_case + logsumexp(scores in group)."""
    G = int(group_ids.max().item()) + 1

    # numerically stable logsumexp per group
    m = torch.full((G,), float('-inf'), device=scores.device)
    m = m.scatter_reduce(0, group_ids, scores, reduce='amax')
    exp_sum = torch.zeros(G, device=scores.device).scatter_add(
        0, group_ids, torch.exp(scores - m[group_ids]))
    lse = m + exp_sum.log()                                   # (G,)

    pos_score = torch.zeros(G, device=scores.device).scatter_add(
        0, group_ids, scores * labels.float())                # (G,)

    return (lse - pos_score).mean()

#tracking experiments
class BestMetricTracker(L.Callback):
    def __init__(self):
        super().__init__()
        self.best_train_acc = 0.0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        # Prevent tracking during the initial sanity check pass
        if trainer.sanity_checking:
            return

        # Fetch the logged metrics dictionary
        metrics = trainer.callback_metrics
        
        # Extract values (handling the _epoch suffix for training metrics)
        train_acc = metrics.get("train_acc_epoch") 
        val_loss = metrics.get("val_loss")
        val_acc = metrics.get("val_acc")
        
        current_epoch = trainer.current_epoch

        # Keep track of global maximums / minimums
        if train_acc is not None:
            self.best_train_acc = max(self.best_train_acc, train_acc.item())
        
        if val_acc is not None:
            self.best_val_acc = max(self.best_val_acc, val_acc.item())
            
        if val_loss is not None and val_loss.item() < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            # If you want the epoch where the best validation loss happened:
            self.best_epoch = current_epoch