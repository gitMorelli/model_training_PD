import math

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
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassRecall,
    BinaryAveragePrecision,   BinaryF1Score,       BinaryMatthewsCorrCoef,
    MulticlassAveragePrecision, MulticlassF1Score, MulticlassMatthewsCorrCoef,
)
import random
import shutil
#from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
import psutil
import time
from collections import deque


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

class ModelPDBase(L.LightningModule):
    """Shared machinery: forward, optimizers/schedulers, epoch logging,
    gradient checks, class-ratio bookkeeping. Subclasses implement
    `compute_loss_and_metrics(batch, stage)`."""

    def __init__(self, write_log, model,
                 total_units,                      # subjects (flat) or groups (grouped)
                 num_classes=1,
                 lr_backbone=1e-4, lr_classifier_head=1e-3,
                 example_input_array=None,
                 opt_groups=None, num_epochs=10, lr_scheduling='cosine',
                 weight_decay=1e-4, warmup_fraction=0.1,
                 eta_min_cosine=1e-6, batch_size=32, accumulate_grad_batches=1, world_size=1):
        super().__init__()
        self.write_log = write_log
        self.model = model
        self.opt_groups = opt_groups

        self.num_classes = num_classes
        self.lr_backbone = lr_backbone
        self.lr_classifier_head = lr_classifier_head
        self.lr_scheduling = lr_scheduling
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_fraction = warmup_fraction
        self.eta_min_cosine = eta_min_cosine
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        # NOTE: `total_units` counts *whatever the loader batches*:
        # subjects for the flat loader, groups for the grouped loader.
        self.total_units = total_units
        steps_per_epoch = math.ceil(
            math.ceil(total_units / (batch_size * world_size)) / accumulate_grad_batches
        )
        self.total_steps = num_epochs * steps_per_epoch

        if example_input_array is not None:
            self.example_input_array = example_input_array

        # bookkeeping shared by both regimes
        self.train_sample_count = 0

        # class counts + per-class hit counts live in one dict per stage.
        # correct_k = number of samples with true label k that were predicted k
        # -> recall_k = correct_k / n_k, balanced_acc = mean(recall_k)
        self._stats = {"train": self._empty_stats(), "val": self._empty_stats()}

    # ---- forward ------------------------------------------------------------
    def forward(self, frames, seq_ids, slot_ids, lengths):
        return self.model(frames, seq_ids, slot_ids, lengths)

    @staticmethod
    def make_example_input(k, n_slots, n_views_frames=2, C=3, H=224, W=224):
        T = min(n_views_frames, n_slots)
        frames = torch.randn(T, k, C, H, W)
        seq_ids = torch.zeros(T, dtype=torch.long)
        slot_ids = torch.arange(T, dtype=torch.long)
        lengths = torch.tensor([T])
        return (frames, seq_ids, slot_ids, lengths)

    # ---- steps delegate to the subclass --------------------------------------
    def compute_loss_and_metrics(self, batch, stage):
        """Return the loss tensor (or None to skip the step) for stage='train';
        return value is ignored for stage='val'."""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self.compute_loss_and_metrics(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.compute_loss_and_metrics(batch, "val")

    # ---- small shared helpers -------------------------------------------------
    def _guard_finite(self, tensor, what):
        if not torch.isfinite(tensor).all():
            self.write_log(f"Warning: NaN/inf in {what} at step {self.trainer.global_step}. Skipping.")
            return False
        return True

    def _log_lrs(self):
        opt = self.optimizers()
        logged = set()
        for g in opt.param_groups:
            name = g.get('lr_name')
            if name and name not in logged:
                self.log(f'lr/{name}', g['lr'], on_step=False, on_epoch=True)
                logged.add(name)

    # ---------- metric bookkeeping (manual counts / balanced acc) --------------
    @property
    def _n_stat_classes(self):
        """2 for the BCE head (num_classes == 1), else num_classes."""
        return 2 if self.num_classes <= 1 else self.num_classes

    def _empty_stats(self):
        k = self._n_stat_classes
        return {"n": [0] * k, "correct": [0] * k}

    def _logits_to_preds(self, logits):
        """Binary (1 logit / no class dim) -> sigmoid threshold; else argmax."""
        logits = logits.detach()
        if self.num_classes <= 1 or logits.ndim == 1 or logits.shape[-1] == 1:
            return (torch.sigmoid(logits.reshape(-1)) > 0.5).long()
        return logits.argmax(dim=-1).reshape(-1)

    def _update_class_counts(self, labels, stage, preds=None):
        k = self._n_stat_classes
        t = labels.detach().long().view(-1)
        if t.numel() and (t.min() < 0 or t.max() >= k):
            raise ValueError(f"labels contain values outside [0, {k})")
        s = self._stats[stage]
        n = torch.bincount(t, minlength=k).tolist()
        for i in range(k):
            s["n"][i] += n[i]
        if preds is not None:
            p = preds.detach().long().view(-1)
            hit = torch.bincount(t[p == t], minlength=k).tolist()
            for i in range(k):
                s["correct"][i] += hit[i]

    def _update_metrics(self, logits, labels, stage):
        """Convenience wrapper — call this from `compute_loss_and_metrics`
        instead of `_update_class_counts` and you get balanced accuracy for free."""
        self._update_class_counts(labels, stage, preds=self._logits_to_preds(logits))

    # ---------- TensorBoard helpers -------------------------------------------
    def _sum_across_ranks(self, value):
        """Sum an int, or elementwise-sum a list of ints, over DDP ranks."""
        seq = isinstance(value, (list, tuple))
        try:
            world = self.trainer.world_size
        except RuntimeError:
            world = 1
        if world <= 1:
            return [int(v) for v in value] if seq else int(value)
        t = torch.tensor([float(v) for v in value] if seq else float(value),
                         device=self.device)
        out = self.all_gather(t).sum(dim=0)          # (world,) or (world, k)
        return [int(x) for x in out] if seq else int(out.item())

    def _log_class_metrics(self, stage):
        s = self._stats[stage]
        n = self._sum_across_ranks(s["n"])
        c = self._sum_across_ranks(s["correct"])
        k, total = len(n), sum(n)

        recalls = [c[i] / n[i] if n[i] > 0 else float("nan") for i in range(k)]

        # CHANGED: all counts now live under "{stage}/class_counts/..."
        for i in range(k):
            self.log(f"class_counts/{stage}_class_{i}_count", float(n[i]),
                     rank_zero_only=True)
            if total > 0:
                self.log(f"class_counts/{stage}_class_{i}_fraction", n[i] / total,
                         rank_zero_only=True)

        if k == 2:   # keep the familiar binary chart
            self.log(f"class_counts/{stage}_class_ratio",
                     n[0] / n[1] if n[1] > 0 else float("inf"), rank_zero_only=True)
        else:
            self.log(f"class_counts/{stage}_imbalance_ratio",
                     max(n) / min(n) if min(n) > 0 else float("inf"),
                     rank_zero_only=True)

        for i, r in enumerate(recalls):
            if r == r:                                # skip NaN
                self.log(f"class_counts/{stage}_recall_class_{i}", r, rank_zero_only=True)

        balanced_acc = (sum(recalls) / k
                        if all(r == r for r in recalls) else float("nan"))
        if balanced_acc == balanced_acc:
            self.log(f"{stage}/balanced_accuracy", balanced_acc,
                     prog_bar=(stage == "val"), rank_zero_only=True)

        counts_str = " ".join(f"n{i}={n[i]}" for i in range(k))
        rec_str = " ".join(f"recall{i}={recalls[i]:.4f}" for i in range(k))
        self.write_log(f"[Epoch {self.current_epoch + 1}] {stage}: {counts_str} "
                       f"{rec_str} balanced_acc={balanced_acc:.4f}\n")
        self._stats[stage] = self._empty_stats()

    # ---- epoch hooks ------------------------------------------------------------
    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.write_log(f"Device of model at start of training: {next(self.model.parameters()).device}")
        self.train_sample_count = 0

    def _log_epoch_summary_extras(self):
        """Subclasses may append regime-specific lines to the epoch summary."""
        pass

    def on_train_epoch_end(self):
        WHITE = "\033[97m"
        RED = "\033[91m"
        RESET = "\033[0m"

        # 0. Map each parameter to its optimizer group's current lr (and name)
        param_lr = {}
        optimizer = self.optimizers()
        optimizers = optimizer if isinstance(optimizer, (list, tuple)) else [optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    param_lr[id(p)] = (group["lr"], group.get("lr_name", "?"))

        # 1. Walk every parameter in model order
        all_layers_info = []
        total_trainable_params = 0
        total_non_trainable_params = 0
        for name, param in self.model.named_parameters():
            layer_str = f"  - {name} | Shape: {list(param.shape)} | Parameters: {param.numel():,}"
            if param.requires_grad:
                lr, lr_name = param_lr.get(id(param), (None, None))
                lr_str = f" | LR: {lr:.3e} ({lr_name})" if lr is not None else " | LR: NOT IN OPTIMIZER"
                all_layers_info.append(f"{WHITE}{layer_str}{lr_str}{RESET}")
                total_trainable_params += param.numel()
            else:
                all_layers_info.append(f"{RED}{layer_str}{RESET}")
                total_non_trainable_params += param.numel()

        total_params = total_trainable_params + total_non_trainable_params

        self.write_log(
            f"\n[Epoch {self.current_epoch + 1}] "
            f"Total Parameters: {total_params:,} | "
            f"{WHITE}Trainable: {total_trainable_params:,}{RESET} | "
            f"{RED}Non-trainable: {total_non_trainable_params:,}{RESET}"
        )

        if self.current_epoch in [0, 5]:
            self.write_log(f"\n--- Epoch {self.current_epoch + 1} Summary ---")
            self.write_log(f"Expected number of stepping batches: {self.trainer.estimated_stepping_batches}")
            self.write_log(f"Epoch {self.current_epoch + 1}: Total Training Samples Processed: {self.train_sample_count}\n")
            self.write_log(f"Expected total units (subjects or groups): {self.total_units}\n")
            self._log_epoch_summary_extras()

            self.write_log("\n--- Model Architecture Summary ---\n")
            self.write_log(f"Total Parameters Count: {total_params:,}\n")
            self.write_log(f"Total Trainable Parameters Count: {total_trainable_params:,}\n")
            self.write_log(f"Total Non-trainable Parameters Count: {total_non_trainable_params:,}\n")
            self.write_log("Model Layers Structure (white = trainable, red = frozen):\n")
            for layer_info in all_layers_info:
                self.write_log(f"{layer_info}\n")

        self._log_class_metrics("train")

    def on_validation_epoch_end(self):
        self._log_class_metrics("val")

    # ---- optimizers ---------------------------------------------------------------
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
            assigned = {}  # param name -> lr_name of the group that claimed it (first match wins)
            for group in self.opt_groups:
                matched = [(name, param) for name, param in self.model.named_parameters()
                           if any(key in name.lower() for key in group['names'])
                           and param.requires_grad]

                named = [(name, param) for name, param in matched if name not in assigned]
                duplicates = [name for name, _ in matched if name in assigned]
                if duplicates:
                    for name in duplicates:
                        self.write_log(
                            f"[configure_optimizers] WARNING: parameter '{name}' matches "
                            f"group '{group['lr_name']}' but was already assigned to "
                            f"group '{assigned[name]}' (first group in order wins).\n"
                        )

                assigned.update({name: group['lr_name'] for name, _ in named})
                add_group(named, group['lr'], group['lr_name'])

            leftover = [name for name, param in self.model.named_parameters()
                        if param.requires_grad and name not in assigned]
            if leftover:
                raise ValueError(
                    f"configure_optimizers: {len(leftover)} trainable parameters "
                    f"matched no optimization group and would silently not be "
                    f"trained: {leftover}"
                )
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

        optimizer = optim.AdamW(param_groups)

        if self.lr_scheduling == 'cosine':
            try:
                total_steps = int(self.trainer.estimated_stepping_batches)
            except (RuntimeError, AttributeError):
                # no trainer attached (unit tests, manual instantiation) -> fall back
                total_steps = self.total_steps
            if not math.isfinite(total_steps) or total_steps <= 0:
                total_steps = self.total_steps

            self.write_log(f"[configure_optimizers] scheduler total_steps={total_steps}")
            warmup_steps = max(1, int(self.warmup_fraction * total_steps))

            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=self.eta_min_cosine)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }
        return {"optimizer": optimizer}

    # ---- gradient sanity check ------------------------------------------------------
    def on_after_backward(self):
        if getattr(self, "_grad_check_done", False):
            #cause with gradient accumulation self.trainer.global_step == 0 accumulate_grad_batches times
            return
        self._grad_check_done = True
        if self.trainer.global_step == 0:
            self.write_log("\n--- Gradient Check (First Batch) ---")
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if 'layer4' in name or 'classifier' in name:
                        grad_abs_mean = param.grad.abs().mean().item()
                        self.write_log(f"Layer '{name}': Mean Abs Gradient = {grad_abs_mean:.2e}")
                        if grad_abs_mean < 1e-8:
                            self.write_log(f"  -> WARNING: Potential vanishing gradient in layer {name}")
            self.write_log("-------------------------------------\n")


# Flat regime: standard BCE / CE training (original ModelPD behaviour)
class ModelPDClassification(ModelPDBase):
    """Per-subject classification supporting an arbitrary number of classes.

        num_classes == 1  -> binary head trained with BCEWithLogitsLoss
                             (requires exactly 2 counts: [num_neg, num_pos])
        num_classes >= 2  -> multiclass head trained with CrossEntropyLoss
                             (requires num_classes counts)

    Class balancing uses sklearn-style inverse-frequency weights:

        w_c = total / (n_eff * count_c)

    optionally interpolated toward uniform (1.0) by ``balancing_factor``
    (0.0 -> no balancing, 1.0 -> full inverse-frequency balancing). At the
    default of 1.0 this matches the original two-class weighting exactly.

    Expects the flat collate:
    (frames, seq_ids, slot_ids, lengths, labels, resized, subject_ids, modalities).
    """

    def __init__(self, write_log, model, class_counts, num_classes=None,
                 use_balanced_weights=True, balancing_factor=1.0, balanced_data=False, drop_train_pr_auc=True,
                 **base_kwargs):
        # ---- normalize counts --------------------------------------------------
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or counts.numel() < 2:
            raise ValueError("class_counts must be a 1-D sequence with >= 2 entries "
                             "(e.g. [n_class0, n_class1, ...]).")

        if num_classes is None:
            num_classes = counts.numel()

        n_eff = 2 if num_classes == 1 else num_classes
        if counts.numel() != n_eff:
            raise ValueError(
                f"Expected {n_eff} class counts for num_classes={num_classes}, "
                f"got {counts.numel()}.")

        super().__init__(write_log, model, num_classes=num_classes, **base_kwargs)
        self.save_hyperparameters(ignore=['model', 'write_log'])

        self.class_counts = counts.tolist()
        self.total = float(counts.sum().item())
        self.balancing_factor = balancing_factor
        self.balanced_data = balanced_data
        self.use_balanced_weights = use_balanced_weights

        # ---- balancing weights -------------------------------------------------
        raw_weights = self.total / (n_eff * counts.clamp(min=1.0))
        class_weights = 1.0 + balancing_factor * (raw_weights - 1.0)
        self.register_buffer("class_weights", class_weights)

        raw_pos = (counts[0] / counts[1]) if counts[1] > 0 else torch.tensor(1.0)
        pos_weight = 1.0 + balancing_factor * (raw_pos - 1.0)
        self.register_buffer("pos_weight",
                             pos_weight.reshape(1).to(torch.float32))

        # ---- criteria ----------------------------------------------------------
        # Weighted criterion = training objective (unchanged).
        if num_classes == 1:
            self.criterion = (nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
                              if use_balanced_weights else nn.BCEWithLogitsLoss())
        else:
            self.criterion = (nn.CrossEntropyLoss(weight=self.class_weights)
                              if use_balanced_weights else nn.CrossEntropyLoss())

        # NEW: unweighted criterion, val-only, as a *calibration diagnostic*.
        # This is the proper NLL under val's actual class prior — NOT a
        # selection metric (its constant-predictor optimum is the base rate,
        # so it would reward the majority-leaning degenerate solution).
        self.criterion_unweighted = (nn.BCEWithLogitsLoss() if num_classes == 1
                                     else nn.CrossEntropyLoss())

        # kept for backward compatibility (macro recall == balanced accuracy)
        self.val_balanced_acc = MulticlassRecall(num_classes=n_eff, average="macro")

        # ---- NEW: pr-auc / f1(pos) / mcc as torchmetrics objects --------------
        # n_eff == 2 covers BOTH the 1-logit BCE head and a 2-logit CE head:
        #   -> Binary* metrics, F1 is the positive-class (label 1) F1,
        #      PR-AUC is average precision of the positive class.
        # n_eff  > 2 -> Multiclass* with macro averaging; there is no single
        #      "positive class", so f1_pos falls back to macro-F1.
        def _build_metrics(k, prefix, include_pr_auc=True):
            metrics = {}
            if k == 2:
                if include_pr_auc:
                    metrics["pr_auc"] = BinaryAveragePrecision()
                metrics["f1_pos"] = BinaryF1Score()
                metrics["mcc"]    = BinaryMatthewsCorrCoef()
            else:
                if include_pr_auc:
                    metrics["pr_auc"] = MulticlassAveragePrecision(num_classes=k, average="macro")
                metrics["f1_pos"] = MulticlassF1Score(num_classes=k, average="macro")
                metrics["mcc"]    = MulticlassMatthewsCorrCoef(num_classes=k)
            # prefix goes straight on the collection -> no .clone() needed,
            # and fresh instances per collection (metrics are stateful).
            return MetricCollection(metrics, prefix=prefix)

        # drop_train_pr_auc: skip the memory-heavy AveragePrecision on train only.
        self.train_metrics = _build_metrics(n_eff, prefix="train/",
                                            include_pr_auc=not drop_train_pr_auc)
        self.val_metrics   = _build_metrics(n_eff, prefix="val/",
                                            include_pr_auc=True)


    # ---- score shaping for the torchmetrics collection ------------------------
    def _scores_for_metrics(self, outputs):
        """Per-sample scores in the shape the metric collection expects.

        n_eff == 2 -> (N,) positive-class probability
                       (sigmoid for the 1-logit BCE head; softmax[:, 1] for a
                        2-logit CE head — both threshold at 0.5, matching the
                        preds used for acc/recall).
        n_eff  > 2 -> (N, C) class probabilities.
        """
        if self._n_stat_classes == 2:
            if self.num_classes == 1:                          # BCE single logit
                return torch.sigmoid(outputs).reshape(-1)
            return torch.softmax(outputs, dim=1)[:, 1]         # 2-logit CE head
        return torch.softmax(outputs, dim=1)                   # multiclass

    # ---- loss + metrics --------------------------------------------------------
    def compute_loss_and_metrics(self, batch, stage):
        frames, seq_ids, slot_ids, lengths, labels, *_ = batch
        bsz = labels.size(0)

        targets = labels.float().unsqueeze(1) if self.num_classes == 1 else labels.long()
        outputs = self(frames, seq_ids, slot_ids, lengths)

        if stage == "train" and not self._guard_finite(outputs, "outputs"):
            return None

        loss = self.criterion(outputs, targets)
        if stage == "train" and not self._guard_finite(loss, "loss"):
            return None

        if self.num_classes == 1:
            preds = (outputs > 0.0).long().view(-1)
        else:
            preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels.long().view(-1)).float().mean()

        self._update_class_counts(labels, stage, preds=preds)

        # CHANGED: everything rerooted under "{stage}/..."
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log(f"{stage}/acc",  acc,  on_epoch=True, prog_bar=True, batch_size=bsz)

        # NEW: pr-auc / f1(pos) / mcc — logged as objects so Lightning
        # computes + resets + DDP-syncs at epoch end. Names become
        # "{stage}/pr_auc", "{stage}/f1_pos", "{stage}/mcc".
        scores = self._scores_for_metrics(outputs)
        tgt = labels.long().view(-1)
        mcoll = self.train_metrics if stage == "train" else self.val_metrics
        mcoll.update(scores, tgt)
        self.log_dict(mcoll, on_step=False, on_epoch=True, batch_size=bsz)

        if stage == "train":
            self.train_sample_count += bsz
            self._log_lrs()
            return loss
        else:
            # NEW: unweighted (proper) val loss — diagnostic only.
            loss_unweighted = self.criterion_unweighted(outputs, targets)
            self.log("val/loss_unweighted", loss_unweighted,
                     on_epoch=True, batch_size=bsz)

            self.val_balanced_acc.update(preds.view(-1), labels.long().view(-1))
            self.log("val/balanced_acc", self.val_balanced_acc, on_epoch=True,
                     prog_bar=True, batch_size=bsz)
            return None

    def _log_epoch_summary_extras(self):
        counts_str = ", ".join(f"Class {i}: {int(c)}"
                               for i, c in enumerate(self.class_counts))
        self.write_log(f"Samples per class -> {counts_str}\n")
        self.write_log(f"Balancing Factor: {self.balancing_factor}\n")
        self.write_log(f"Balanced Data: {self.balanced_data}, "
                       f"Use Balanced Weights: {self.use_balanced_weights}\n")
        self.write_log(f"Weights for Loss Function: {self.class_weights.tolist()}\n")

    # ---- prediction ------------------------------------------------------------
    def predict_step(self, batch, batch_idx):
        frames, seq_ids, slot_ids, lengths, labels, \
            resizing_factors, subject_ids, modalities = batch

        outputs = self(frames, seq_ids, slot_ids, lengths)

        if self.num_classes == 1:
            p1 = torch.sigmoid(outputs).flatten()
            probs = torch.stack([1 - p1, p1], dim=1)
            preds = (outputs > 0.0).long().flatten()
        else:
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

        return {
            "probs": probs.detach().cpu(),
            "preds": preds.detach().cpu(),
            "labels": labels.detach().cpu().flatten(),
            "subject_ids": subject_ids,
        }

# Grouped regime: conditional-logistic (within-group softmax) training
class ModelPDGrouped(ModelPDBase):
    """Case/control matched-set training. Expects the grouped collate:
    (frames, seq_ids, slot_ids, lengths, labels, resized, subject_ids,
     modalities, group_ids), where group_ids is per-subject with within-batch
    indices 0..G-1 and each group contains exactly one positive.
 
    The model must output one logit per subject (n_classes=1); an unweighted
    BCE auxiliary term (bce_aux_weight) keeps the logit calibrated so
    sigmoid(score) > 0.5 remains a meaningful standalone 0/1 prediction.
    `total_units` passed to the base must be the number of GROUPS (batch_size
    in the grouped loader counts groups)."""
 
    def __init__(self, write_log, model, bce_aux_weight=0.3,
                 **base_kwargs):
        base_kwargs.pop('num_classes', None)   # grouped regime is always 1-logit
        super().__init__(write_log, model,
                         num_classes=1, **base_kwargs)
        self.save_hyperparameters(ignore=['model', 'write_log'])
 
        self.bce_aux_weight = bce_aux_weight
        self.criterion_bce = nn.BCEWithLogitsLoss()   # unweighted, calibration only
 
        self.val_balanced_acc = MulticlassRecall(num_classes=2, average="macro")
 
    # ---- loss + metrics ---------------------------------------------------------
    def compute_loss_and_metrics(self, batch, stage):
        frames, seq_ids, slot_ids, lengths, labels, \
            resized, subject_ids, modalities, group_ids = batch
        bsz = labels.size(0)                          # subjects in batch
        n_groups = int(group_ids.max().item()) + 1
 
        scores = self(frames, seq_ids, slot_ids, lengths).squeeze(-1)   # (B,)
 
        if stage == "train" and not self._guard_finite(scores, "outputs"):
            return None
 
        loss_grp = grouped_ce_loss(scores, labels, group_ids)
        loss_bce = self.criterion_bce(scores, labels.float())
        loss = loss_grp + self.bce_aux_weight * loss_bce
 
        if stage == "train" and not self._guard_finite(loss, "loss"):
            return None
 
        with torch.no_grad():
            grp_acc = group_accuracy(scores, labels, group_ids)
            preds = (scores > 0.0).long()             # standalone 0/1
            acc = (preds == labels.long()).float().mean()
 
        self._update_class_counts(labels, stage)
        self.log(f"{stage}_loss",      loss,     on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log(f"{stage}_loss_grp",  loss_grp, on_epoch=True, batch_size=bsz)
        self.log(f"{stage}_loss_bce",  loss_bce, on_epoch=True, batch_size=bsz)
        self.log(f"{stage}_group_acc", grp_acc,  on_epoch=True, prog_bar=True, batch_size=n_groups)
        self.log(f"{stage}_acc",       acc,      on_epoch=True, batch_size=bsz)
 
        if stage == "train":
            self.train_sample_count += bsz
            self._log_lrs()
            return loss
        else:
            self.val_balanced_acc.update(preds, labels.long().view(-1))
            self.log("val_balanced_acc", self.val_balanced_acc, on_epoch=True,
                     prog_bar=True, batch_size=bsz)
            return None
 
    def _log_epoch_summary_extras(self):
        self.write_log(f"BCE auxiliary weight: {self.bce_aux_weight}\n")
 
    # ---- prediction --------------------------------------------------------------
    def predict_step(self, batch, batch_idx):
        frames, seq_ids, slot_ids, lengths, labels, \
            resized, subject_ids, modalities, group_ids = batch
 
        scores = self(frames, seq_ids, slot_ids, lengths).squeeze(-1)
 
        # (a) standalone probability / 0-1 per subject
        p1 = torch.sigmoid(scores)
        probs = torch.stack([1 - p1, p1], dim=1)
        preds_standalone = (scores > 0.0).long()
 
        # (b) within-group prediction: the argmax of each group is flagged as the case
        m = group_max(scores, group_ids)
        preds_group = (scores == m[group_ids]).long()
 
        return {
            "scores": scores.detach().cpu(),
            "probs": probs.detach().cpu(),
            "preds": preds_standalone.detach().cpu(),
            "preds_group": preds_group.detach().cpu(),
            "labels": labels.detach().cpu().flatten(),
            "group_ids": group_ids.detach().cpu(),
            "subject_ids": subject_ids,
        }
 

#------ Loss functions ---------------
def grouped_ce_loss(scores, labels, group_ids):
    """Conditional logistic / within-group softmax NLL.
 
    scores    : (B,) raw logits, one per subject
    labels    : (B,) 0/1, exactly one 1 per group
    group_ids : (B,) group index in 0..G-1 (within-batch indices)
 
    loss = mean over groups of ( logsumexp(scores in group) - score_of_case )
    """
    G = int(group_ids.max().item()) + 1
 
    m = group_max(scores, group_ids, G)                                    # (G,)
    exp_sum = torch.zeros(G, device=scores.device, dtype=scores.dtype).scatter_add(
        0, group_ids, torch.exp(scores - m[group_ids]))
    lse = m + exp_sum.log()                                                # (G,)
 
    pos_score = torch.zeros(G, device=scores.device, dtype=scores.dtype).scatter_add(
        0, group_ids, scores * labels.float())                             # (G,)
 
    return (lse - pos_score).mean()

# -------------- Callbacks -------------------
#tracking experiments
class BestMetricTracker(L.Callback):
    """Tracks best-so-far validation metrics. Selection now follows PR-AUC
    (max), matching the ModelCheckpoint / EarlyStopping monitor, so best_epoch
    points at the epoch whose checkpoint was saved."""

    # "higher is better"  -> logged metric name : attribute
    _MAX_METRICS = {
        "val/pr_auc":       "best_val_pr_auc",
        "val/f1_pos":       "best_val_f1_pos",
        "val/mcc":          "best_val_mcc",
        "val/balanced_acc": "best_val_balanced_acc",
        "val/acc":          "best_val_acc",
        "train/acc_epoch":  "best_train_acc",   # note: train keeps the _epoch suffix
    }
    # "lower is better"
    _MIN_METRICS = {
        "val/loss":            "best_val_loss",              # weighted (training objective)
        "val/loss_unweighted": "best_val_loss_unweighted",   # proper NLL, diagnostic only
    }
    # metric that defines "the best epoch"
    _ANCHOR = ("val/pr_auc", "max")

    def __init__(self):
        super().__init__()
        for attr in self._MAX_METRICS.values():
            setattr(self, attr, float("-inf"))
        for attr in self._MIN_METRICS.values():
            setattr(self, attr, float("inf"))
        self.best_epoch = 0
        self._anchor_best = float("-inf") if self._ANCHOR[1] == "max" else float("inf")

    @staticmethod
    def _get(metrics, key):
        """Fetch a metric as a float, or None if missing/NaN."""
        v = metrics.get(key)
        if v is None:
            return None
        v = v.item() if hasattr(v, "item") else float(v)
        return v if v == v else None      # drop NaN (e.g. degenerate pr-auc epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:       # skip the sanity pass
            return
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        for key, attr in self._MAX_METRICS.items():
            v = self._get(metrics, key)
            if v is not None:
                setattr(self, attr, max(getattr(self, attr), v))

        for key, attr in self._MIN_METRICS.items():
            v = self._get(metrics, key)
            if v is not None:
                setattr(self, attr, min(getattr(self, attr), v))

        # anchor best_epoch to the selection metric
        anchor_key, anchor_mode = self._ANCHOR
        v = self._get(metrics, anchor_key)
        if v is not None:
            improved = (v > self._anchor_best if anchor_mode == "max"
                        else v < self._anchor_best)
            if improved:
                self._anchor_best = v
                self.best_epoch = epoch

    @property
    def best_metrics(self):
        """Convenience dict for end-of-run printing."""
        d = {attr: getattr(self, attr)
             for attr in list(self._MAX_METRICS.values()) + list(self._MIN_METRICS.values())}
        d["best_epoch"] = self.best_epoch
        return d

class ClearCache(L.Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

class TimeLoader(Callback):
    def __init__(self):
        self.n, self.tot, self.win = 0, 0.0, []
    def on_train_batch_end(self, trainer, pl_module, *a):
        d = trainer.profiler.recorded_durations[
            "[_TrainingEpochLoop].train_dataloader_next"][-1]
        self.n += 1; self.tot += d; self.win.append(d)
        if self.n % 100 == 0:
            print(f"batch {self.n} mean={self.tot/self.n:.4f} "
                  f"last100={sum(self.win)/100:.4f} max={max(self.win):.4f}")
            self.win = []

class BatchTimer(Callback):
    """Times each training batch and pushes it to TensorBoard.
 
    Measures three things:
      * time/data_wait_s  -- how long the loop waited on the dataloader
                             (read from the profiler; needs Trainer(profiler="simple"))
      * time/batch_compute_s -- forward + backward + optimizer step
      * time/epoch_s      -- wall clock per epoch
 
    `sync_cuda=True` inserts torch.cuda.synchronize() so the compute number is
    real; without it CUDA calls are async and you mostly time the Python loop.
    """
 
    DATA_KEY = "[_TrainingEpochLoop].train_dataloader_next"
 
    def __init__(self, window=100, print_every=100, sync_cuda=False, verbose=False):
        self.window = window
        self.print_every = print_every
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.verbose = verbose
 
        self.n = 0
        self.tot_data = 0.0
        self.tot_compute = 0.0
        self.win_data = []
        self.win_compute = []
        self._t0 = None
        self._epoch_t0 = None
 
    # -- helpers --------------------------------------------------------------
    def _sync(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
 
    def _data_wait(self, trainer):
        rec = getattr(trainer.profiler, "recorded_durations", None)
        if rec:
            durations = rec.get(self.DATA_KEY)
            if durations:
                return durations[-1]
        return float("nan")   # no profiler attached -> silently skip
 
    # -- hooks ----------------------------------------------------------------
    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_t0 = time.perf_counter()
 
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._sync()
        self._t0 = time.perf_counter()
 
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync()
        compute = time.perf_counter() - self._t0
        data = self._data_wait(trainer)
 
        pl_module.log("time/batch_compute_s", compute,
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        if data == data:  # not NaN
            pl_module.log("time/data_wait_s", data,
                          on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
            frac = data / (data + compute) if (data + compute) > 0 else 0.0
            pl_module.log("time/data_wait_fraction", frac,
                          on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
            self.tot_data += data
            self.win_data.append(data)
 
        self.n += 1
        self.tot_compute += compute
        self.win_compute.append(compute)
 
        if self.verbose and self.n % self.print_every == 0:
            def stats(win, tot):
                if not win:
                    return "n/a"
                return (f"mean={tot / self.n:.4f} "
                        f"last{len(win)}={sum(win) / len(win):.4f} "
                        f"max={max(win):.4f}")
            print(f"batch {self.n} | data  {stats(self.win_data, self.tot_data)}")
            print(f"batch {self.n} | compute {stats(self.win_compute, self.tot_compute)}")
            self.win_data, self.win_compute = [], []
 
        # keep SimpleProfiler's recorded_durations from growing without bound
        rec = getattr(trainer.profiler, "recorded_durations", None)
        if rec and self.DATA_KEY in rec and len(rec[self.DATA_KEY]) > 10_000:
            rec[self.DATA_KEY] = rec[self.DATA_KEY][-100:]
 
    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_t0 is not None:
            pl_module.log("time/epoch_s", time.perf_counter() - self._epoch_t0,
                          on_step=False, on_epoch=True, batch_size=1)

class ThroughputMonitor(Callback):
    """Rolling-window training speed monitor.

    Logged every `window` batches:
      time/samples_per_s    -- wall-clock throughput. The number that matters.
      time/batches_per_s    -- same, per batch.
      time/gpu_busy_s       -- mean GPU time per batch, measured with CUDA events.
      time/gpu_utilization  -- gpu_busy / wall_clock. ~1.0 = GPU-bound,
                               < ~0.85 = the GPU is idling and you are losing time.
      time/data_wait_s      -- mean host-side wait on the dataloader. Only a
                               problem if utilization is also low; a wait that
                               overlaps with GPU work is free.

    Dataloader wait is read from the profiler when one is attached
    (Trainer(profiler="simple")); otherwise it falls back to the gap between
    consecutive batch hooks, which also includes callback/logging overhead.

    Notes / limits:
      * CUDA events time the current stream. Work issued on side streams
        (some DDP comm, custom prefetchers) is not counted, so utilization can
        read low on a run that is actually saturated.
      * Under DDP every rank logs its own numbers; these are rank-local by
        design, since stragglers are exactly what you want to see.
      * Utilization can drift slightly above 1.0 from event/wall-clock skew.
    """

    DATA_KEY = "[_TrainingEpochLoop].train_dataloader_next"

    def __init__(self, window=50, event_lag=16, samples_per_batch=None,
                 verbose=False):
        self.window = window
        self.event_lag = event_lag          # batches to wait before reading events
        self.samples_per_batch = samples_per_batch   # override if inference fails
        self.verbose = verbose
        self.cuda = torch.cuda.is_available()

        self._pending = deque()
        self._last_batch_end = None
        self._epoch_t0 = None
        self._reset_window()

    # -- internals ------------------------------------------------------------
    def _reset_window(self):
        self._wall_t0 = time.perf_counter()
        self._n_batches = 0
        self._n_samples = 0
        self._gpu_ms = 0.0
        self._gpu_n = 0
        self._data_s = 0.0
        self._data_n = 0

    @staticmethod
    def _infer_batch_size(batch):
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        if isinstance(batch, (list, tuple)):
            for item in batch:
                if isinstance(item, torch.Tensor):
                    return item.shape[0]
        if isinstance(batch, dict):
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
        return None

    def _data_wait(self, trainer):
        """Host-side dataloader wait. Profiler if available, hook gap otherwise."""
        rec = getattr(trainer.profiler, "recorded_durations", None)
        if rec:
            durations = rec.get(self.DATA_KEY)
            if durations:
                # trim so SimpleProfiler does not grow without bound
                if len(durations) > 10_000:
                    rec[self.DATA_KEY] = durations[-100:]
                return durations[-1]
        if self._last_batch_end is not None:
            return time.perf_counter() - self._last_batch_end
        return None

    def _drain(self, force=False):
        """Read back completed CUDA events. Never blocks unless force=True."""
        while self._pending:
            if not force and len(self._pending) <= self.event_lag:
                break
            start_ev, end_ev = self._pending[0]
            if not force and not end_ev.query():
                break                      # GPU still behind, try again later
            self._pending.popleft()
            end_ev.synchronize()           # no-op if query() already passed
            self._gpu_ms += start_ev.elapsed_time(end_ev)
            self._gpu_n += 1

    def _flush(self, pl_module, on_step=True):
        if self._n_batches == 0:
            return
        wall = time.perf_counter() - self._wall_t0
        if wall <= 0:
            return

        logs = {"time/batches_per_s": self._n_batches / wall}
        if self._n_samples:
            logs["time/samples_per_s"] = self._n_samples / wall
        if self._data_n:
            logs["time/data_wait_s"] = self._data_s / self._data_n
        if self._gpu_n:
            gpu_per_batch = self._gpu_ms / self._gpu_n / 1000.0
            logs["time/gpu_busy_s"] = gpu_per_batch
            # scale by batches in the window, since event readback lags
            logs["time/gpu_utilization"] = gpu_per_batch * self._n_batches / wall

        pl_module.log_dict(logs, on_step=on_step, on_epoch=not on_step,
                       prog_bar=False, batch_size=1)

        if self.verbose:
            parts = [f"{logs.get('time/samples_per_s', float('nan')):8.1f} samp/s"]
            if "time/gpu_utilization" in logs:
                parts.append(f"gpu_util {logs['time/gpu_utilization']:5.1%}")
                parts.append(f"gpu {logs['time/gpu_busy_s'] * 1000:6.1f} ms")
            if "time/data_wait_s" in logs:
                parts.append(f"data {logs['time/data_wait_s'] * 1000:6.1f} ms")
            print("  ".join(parts))

        self._reset_window()

    # -- hooks ----------------------------------------------------------------
    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_t0 = time.perf_counter()
        self._last_batch_end = None
        self._reset_window()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        wait = self._data_wait(trainer)
        if wait is not None:
            self._data_s += wait
            self._data_n += 1

        if self.cuda:
            start_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            self._start_ev = start_ev

        n = self.samples_per_batch or self._infer_batch_size(batch)
        self._pending_samples = n or 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.cuda:
            end_ev = torch.cuda.Event(enable_timing=True)
            end_ev.record()
            self._pending.append((self._start_ev, end_ev))
            self._drain()

        self._n_batches += 1
        self._n_samples += self._pending_samples
        self._last_batch_end = time.perf_counter()

        if self._n_batches >= self.window:
            self._flush(pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.cuda:
            self._drain(force=True)
        self._flush(pl_module, on_step=False)   # <-- add on_step=False
        if self._epoch_t0 is not None:
            pl_module.log("time/epoch_s", time.perf_counter() - self._epoch_t0,
                          on_step=False, on_epoch=True, batch_size=1)

import psutil, os

class MemMonitor(L.Callback):
    def __init__(self, every_n_steps=1000):
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, *args, **kwargs):
        if trainer.global_step % self.every_n_steps != 0:
            return

        p = psutil.Process(os.getpid())
        main = p.memory_full_info().pss / 1e9

        kids = []
        for c in p.children(recursive=True):
            try:
                kids.append((c.pid, c.memory_full_info().pss / 1e9))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass  # worker died/respawned between listing and reading

        total = main + sum(m for _, m in kids)
        print(f"step {trainer.global_step}: main={main:.2f}GB "
              f"workers={[f'{m:.2f}' for _, m in kids]} "
              f"total={total:.2f}GB (PSS)")

#optimization groups utils
def get_optimization_groups(model_name,exp_params):
    if exp_params['use_opt_groups'] == False:
        return None
    if 'resnet' in model_name:
        define_optimization_groups = [
            {'names': ['layer1','vision_model.conv1','vision_model.bn1'],'lr': exp_params['lr_backbone']*0.002, 'lr_name': 'lr_1'},
            {'names': ['layer2'],'lr': exp_params['lr_backbone']*0.01, 'lr_name': 'lr_2'},
            {'names': ['layer3'],'lr': exp_params['lr_backbone']*0.1, 'lr_name': 'lr_3'},
            {'names': ['layer4'],'lr': exp_params['lr_backbone'], 'lr_name': 'lr_4'},
            {'names': ['classifier'], 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_head'},
        ] # or None or other configurations fo other models
    elif 'FiveStageResidualStridedConvNet' in model_name:
        define_optimization_groups = [
            {'names': ['stem', 'stages.0'], 'lr': exp_params['lr_backbone']*0.1*0.1, 'lr_name': 'lr_1'},   # ~ conv1/bn1/layer1
            {'names': ['stages.1'],         'lr': exp_params['lr_backbone']*0.1, 'lr_name': 'lr_2'},   # ~ layer2
            {'names': ['stages.2'],         'lr': exp_params['lr_backbone']*0.5, 'lr_name': 'lr_3'},   # ~ layer3
            {'names': ['stages.3', 'stages.4'], 'lr': exp_params['lr_backbone'], 'lr_name': 'lr_4'},  # ~ layer4
            {'names': ['head', 'projector'], 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_head'},
            {'names': ['classifier'], 'lr': exp_params['lr_classifier_head'], 'lr_name': 'lr_classifier'},
        ]
    return define_optimization_groups

#Set hyperparameters / metadata
def set_automatic_hyperparameters(exp_params):
    if exp_params['grouped'] or exp_params['pre_training']:
        exp_params['balance_validation'] = False
        exp_params['balanced_data'] = False
        exp_params['use_balanced_weights'] = False
    if exp_params['pre_training']:
        exp_params['filter_missing'] = 'all'
        exp_params['censor_time'] = 'all'
    exp_params['num_channels'] = 1 if exp_params['to_grayscale'] else 3
    if isinstance(exp_params['data_modality'],list):
        exp_params['num_tiles'] = len(exp_params['data_modality'])
    huggingface_transform=True if exp_params['model'] in ['clip-vit-large-patch14-un', 'clip-vit-large-patch14-inter'] else False
    exp_params['huggingface_transform'] = huggingface_transform
    if exp_params['debug']:
        exp_params['custom_transform'] = 'pad_resize_pil'
    return exp_params

#-------- others ----------------
def group_max(scores, group_ids, n_groups=None):
    """Per-group max of `scores`. Returns tensor of shape (G,)."""
    G = n_groups if n_groups is not None else int(group_ids.max().item()) + 1
    m = torch.full((G,), float('-inf'), device=scores.device, dtype=scores.dtype)
    return m.scatter_reduce(0, group_ids, scores, reduce='amax')
 
 
def group_accuracy(scores, labels, group_ids):
    """Fraction of groups whose highest-scoring member is the true case."""
    G = int(group_ids.max().item()) + 1
    m = group_max(scores, group_ids, G)
    is_group_max = scores == m[group_ids]                                  # (B,)
    correct = (is_group_max & (labels == 1)).float()
    hits = torch.zeros(G, device=scores.device).scatter_add(0, group_ids, correct)
    return hits.clamp(max=1).mean()
 
 