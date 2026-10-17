import math

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import pandas as pd
import torch.nn as nn
import time
import torch
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
                 eta_min_cosine=1e-6, batch_size=32, label_smoothing=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.opt_groups = opt_groups
        self.label_smoothing = label_smoothing

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
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights,
                                                     label_smoothing=self.label_smoothing)
            else:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
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
    in the grouped loader counts groups).

    ALIGNMENT WITH THE BASE / CLASSIFICATION REGIME
    -----------------------------------------------
    * Log names use the "stage/metric" convention ("train/loss", not
      "train_loss") so the base's "train_window/..." rewrite and the
      "{key}/balanced_accuracy" line from `_log_class_metrics` land in the
      same TensorBoard groups.
    * `_update_metrics` (not the bare `_update_class_counts`) is used, so
      per-class recall and balanced accuracy are actually populated — the old
      call passed no preds, which left every `correct_k` at zero.
    * `train_metrics` / `val_metrics` MetricCollections exist here too
      (pr_auc / f1_pos / mcc on the standalone sigmoid score). The base's
      `_flush_train_window` picks up `self.train_metrics` by name, so in
      aligned mode this class must NOT hand the train collection to
      `log_dict` — same single-ownership rule as ModelPDClassification.
    * Group accuracy is the headline metric of this regime, so on top of the
      Lightning epoch reduction it gets a windowed series
      ("train_window/group_acc") emitted at each validation point, matching
      what the base does for the class-count metrics.
    """

    def __init__(self, write_log, model, bce_aux_weight=0.3,
                 drop_train_pr_auc=True, **base_kwargs):
        base_kwargs.pop('num_classes', None)   # grouped regime is always 1-logit
        super().__init__(write_log, model, num_classes=1, **base_kwargs)
        self.save_hyperparameters(ignore=['model', 'write_log'])

        self.bce_aux_weight = bce_aux_weight
        self.criterion_bce = nn.BCEWithLogitsLoss()   # unweighted, calibration only

        # kept for backward compatibility (macro recall == balanced accuracy)
        self.val_balanced_acc = MulticlassRecall(num_classes=2, average="macro")

        # ---- pr-auc / f1(pos) / mcc, always the binary flavour -------------
        # Mirrors _build_metrics in ModelPDClassification; the multiclass
        # branch is dropped because n_eff is always 2 here.
        def _build_metrics(prefix, include_pr_auc=True):
            metrics = {}
            if include_pr_auc:
                metrics["pr_auc"] = BinaryAveragePrecision()
            metrics["f1_pos"] = BinaryF1Score()
            metrics["mcc"] = BinaryMatthewsCorrCoef()
            return MetricCollection(metrics, prefix=prefix)

        self.drop_train_pr_auc = drop_train_pr_auc
        self.train_metrics = _build_metrics("train/",
                                            include_pr_auc=not drop_train_pr_auc)
        self.val_metrics = _build_metrics("val/", include_pr_auc=True)

        # ===== FRACTIONAL LOGGING: windowed group accuracy =====
        # [correct_groups, total_groups] since the last window flush. Only
        # touched in aligned mode; the epoch-level value comes from Lightning's
        # on_epoch reduction of "train/group_acc".
        self._win_grp = [0, 0]

    # ---- helpers ---------------------------------------------------------
    def _subject_scores(self, outputs):
        """One score per subject. Tolerates a (token_logits, subject_logits)
        tuple from a per-step mixer by taking the subject-level element: the
        conditional-logistic loss is defined over subjects within a group, so
        token-level logits have no group to be normalised in."""
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[-1]
        return outputs.squeeze(-1)

    @staticmethod
    def _group_hits(scores, labels, group_ids):
        """Number of groups whose maximum score belongs to the true case.

        Assumes exactly one positive per group. Exact ties at the maximum
        count as a hit; with float scores that is vanishingly rare, and the
        alternative (counting them as misses) would make the metric depend on
        subject ordering.
        """
        s = scores.detach()
        m = group_max(s, group_ids)
        hit = (s == m[group_ids]) & labels.detach().bool().view(-1)
        return int(hit.sum().item())

    # ===== FRACTIONAL LOGGING: extend the base window flush ================
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._win_grp = [0, 0]

    def _flush_train_window(self, force=False):
        """Base behaviour + the windowed group-accuracy point.

        The gate below intentionally duplicates the one in
        `ModelPDBase._flush_train_window` because `super()` mutates
        `_last_flush_step`; it must be evaluated *before* the super call.
        Like the base gate it reads only `global_step`, which is identical on
        every rank, so the `_sum_across_ranks` collective underneath is
        reached by all ranks or none. (If you ever change the base gate,
        change it here too — or factor it into a shared predicate.)
        """
        step = self.trainer.global_step
        flushing = (step > self._last_flush_step
                    and (force or step - self._last_flush_step >= self.min_window_steps))

        super()._flush_train_window(force=force)

        if not flushing:
            return
        correct, total = self._sum_across_ranks(self._win_grp)   # collective
        self._win_grp = [0, 0]                                   # all ranks
        if total > 0 and self.logger is not None and self.trainer.is_global_zero:
            self.logger.log_metrics({"train_window/group_acc": correct / total},
                                    step=step)

    # ---- loss + metrics ---------------------------------------------------
    def compute_loss_and_metrics(self, batch, stage):
        frames, seq_ids, slot_ids, lengths, labels, \
            resized, subject_ids, modalities, group_ids, *_ = batch
        bsz = labels.size(0)                          # subjects in batch
        n_groups = int(group_ids.max().item()) + 1

        scores = self._subject_scores(self(frames, seq_ids, slot_ids, lengths))

        if stage == "train" and not self._guard_finite(scores, "outputs"):
            return None

        loss_grp = grouped_ce_loss(scores, labels, group_ids)
        loss_bce = self.criterion_bce(scores, labels.float())
        loss = loss_grp + self.bce_aux_weight * loss_bce

        if stage == "train" and not self._guard_finite(loss, "loss"):
            return None

        with torch.no_grad():
            hits = self._group_hits(scores, labels, group_ids)
            grp_acc = hits / n_groups
            preds = (scores > 0.0).long()             # standalone 0/1
            acc = (preds == labels.long().view(-1)).float().mean()

        # class counts + per-class recall + balanced accuracy (base bookkeeping).
        # `_update_metrics` derives preds from the logits exactly as above
        # (num_classes == 1 -> sigmoid > 0.5, i.e. score > 0).
        self._update_metrics(scores, labels, stage)

        if stage == "train" and self.align_train_metrics_to_val:
            self._win_grp[0] += hits
            self._win_grp[1] += n_groups

        # ===== FRACTIONAL LOGGING: per-step train curves =====
        # loss / acc / group_acc need none of the windowing machinery — on_step
        # gives a dense curve that already reads against the mid-epoch val
        # points; the windowed group_acc series above is the smoothed twin.
        on_step = (stage == "train")
        self.log(f"{stage}/loss", loss, on_step=on_step, on_epoch=True,
                 prog_bar=True, batch_size=bsz)
        self.log(f"{stage}/loss_grp", loss_grp, on_step=on_step, on_epoch=True,
                 batch_size=bsz)
        self.log(f"{stage}/loss_bce", loss_bce, on_step=on_step, on_epoch=True,
                 batch_size=bsz)
        self.log(f"{stage}/group_acc", grp_acc, on_step=on_step, on_epoch=True,
                 prog_bar=True, batch_size=n_groups)
        self.log(f"{stage}/acc", acc, on_step=on_step, on_epoch=True,
                 batch_size=bsz)

        # torchmetrics on the standalone (group-agnostic) score
        probs = torch.sigmoid(scores).reshape(-1)
        tgt = labels.long().view(-1)
        mcoll = self.train_metrics if stage == "train" else self.val_metrics
        mcoll.update(probs, tgt)
        # ===== FRACTIONAL LOGGING: single ownership of the train collection =====
        # In aligned mode `_flush_train_window` owns compute()/reset() for
        # train_metrics; handing it to log_dict as well would let Lightning
        # reset it at epoch end underneath us.
        if stage == "val" or not self.align_train_metrics_to_val:
            self.log_dict(mcoll, on_step=False, on_epoch=True, batch_size=bsz)

        if stage == "train":
            self.train_sample_count += bsz
            self._log_lrs()
            return loss
        else:
            self.val_balanced_acc.update(preds.view(-1), tgt)
            self.log("val/balanced_acc", self.val_balanced_acc, on_epoch=True,
                     prog_bar=True, batch_size=bsz)
            return None

    def _log_epoch_summary_extras(self):
        self.write_log(f"BCE auxiliary weight: {self.bce_aux_weight}\n")
        self.write_log(f"Expected groups per epoch: {self.total_units}\n")
        self.write_log(f"Train PR-AUC dropped: {self.drop_train_pr_auc}\n")

    # ---- prediction -------------------------------------------------------
    def predict_step(self, batch, batch_idx):
        frames, seq_ids, slot_ids, lengths, labels, \
            resized, subject_ids, modalities, group_ids , *_ = batch

        scores = self._subject_scores(self(frames, seq_ids, slot_ids, lengths))

        # (a) standalone probability / 0-1 per subject
        p1 = torch.sigmoid(scores)
        probs = torch.stack([1 - p1, p1], dim=1)
        preds_standalone = (scores > 0.0).long()

        # (b) within-group prediction: the argmax of each group is the case
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
 
 