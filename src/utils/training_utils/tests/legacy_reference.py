# Original implementation, kept verbatim (plus imports) as the reference for
# equivalence tests. Not part of the package.
import math
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision, BinaryF1Score, BinaryMatthewsCorrCoef,
    MulticlassAveragePrecision, MulticlassF1Score, MulticlassMatthewsCorrCoef,
    MulticlassRecall,
)

class ModelPDBase(L.LightningModule):
    """Shared machinery: forward, optimizers/schedulers, epoch logging,
    gradient checks, class-ratio bookkeeping. Subclasses implement
    `compute_loss_and_metrics(batch, stage)`.

    FRACTIONAL LOGGING (new)
    ------------------------
    `val_check_interval` makes validation run N times per epoch. It is a
    *Trainer* argument -- pass it through `self.trainer_val_kwargs`, don't set
    it on the Trainer by hand, or the two can drift apart.

    NOTE: `val_check_interval` counts *dataloader batches*, not optimizer
    steps. With accumulate_grad_batches=4, 0.25 still means a quarter of the
    fetched batches. A float requires a finite len(train_dataloader); with an
    iterable-style loader use the int form (which also sets
    check_val_every_n_epoch=None).

    `align_train_metrics_to_val` additionally emits a *windowed* train metric
    at each of those validation points, so train and val curves share an
    x-axis and the generalisation gap is readable. Windowed series are named
    "train_window/..." and are written straight to the logger (see
    `_log_class_metrics(direct_step=...)`) because Lightning mean-reduces
    repeated self.log() calls within an epoch, which would collapse the four
    window values into one point.

    Dataset-composition metrics (counts / fractions / class ratio) are NOT
    windowed: they describe the data, not the model, so windowing them only
    adds sampling noise to a constant.
    """

    def __init__(self, write_log, model,
                 total_units,                      # subjects (flat) or groups (grouped)
                 num_classes=1,
                 lr_backbone=1e-4, lr_classifier_head=1e-3,
                 example_input_array=None,
                 opt_groups=None, num_epochs=10, lr_scheduling='cosine',
                 weight_decay=1e-4, warmup_fraction=0.1,
                 eta_min_cosine=1e-6, batch_size=32, accumulate_grad_batches=1, world_size=1,
                 # ===== FRACTIONAL LOGGING: new args =====
                 val_check_interval=None,          # None -> epoch end; float in (0,1]; int -> every N batches
                 align_train_metrics_to_val=False,  # also emit windowed train metrics at each val point
                 min_window_steps=50, iterable_train_dataset=True,
                 lora_tuning=False, lora_lr_decay=0.75):             # don't emit a window shorter than this many optimizer steps
        super().__init__()
        self.write_log = write_log
        self.model = model
        self.opt_groups = opt_groups
        self.lora_tuning = lora_tuning
        self.lora_lr_decay = lora_lr_decay
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
        self.batches_per_epoch = math.ceil(total_units / (batch_size * world_size))
        steps_per_epoch = math.ceil(self.batches_per_epoch / accumulate_grad_batches)
        self.total_steps = num_epochs * steps_per_epoch

        if example_input_array is not None:
            self.example_input_array = example_input_array

        # bookkeeping shared by both regimes
        self.train_sample_count = 0

        # class counts + per-class hit counts live in one dict per stage.
        # correct_k = number of samples with true label k that were predicted k
        # -> recall_k = correct_k / n_k, balanced_acc = mean(recall_k)
        self._stats = {"train": self._empty_stats(), "val": self._empty_stats()}

        # ===== FRACTIONAL LOGGING: state =====
        self.val_check_interval = val_check_interval
        # alignment is meaningless without mid-epoch validation, so it folds to False
        self.align_train_metrics_to_val = bool(
            align_train_metrics_to_val and val_check_interval is not None)
        self.min_window_steps = min_window_steps
        self._val_run_idx = 0          # which validation run within the current epoch
        self._last_flush_step = 0      # global_step at the last window flush
        # Second accumulator: `_stats["train"]` is cleared at every window flush,
        # so the whole-epoch composition numbers need their own bucket.
        self._epoch_stats_train = self._empty_stats()
        self.iterable_train_dataset = iterable_train_dataset #True if dataloader has no __len__

    # ===== FRACTIONAL LOGGING: Trainer wiring ================================
    @property
    def trainer_val_kwargs(self):
        v = self.val_check_interval
        if v is None:
            return {}
        if isinstance(v, float):
            if not 0.0 < v <= 1.0:
                raise ValueError("float val_check_interval must be in (0, 1]")
            if v == 1.0:
                return {}                       # Trainer default; also legal for iterable
            if self.iterable_train_dataset:
                # no __len__ -> Lightning can't resolve a fraction; convert to batches
                n = max(1, int(round(v * self.batches_per_epoch)))
                self.write_log(f"[trainer_val_kwargs] iterable train dataset: "
                            f"val_check_interval {v} -> {n} batches "
                            f"(batches_per_epoch≈{self.batches_per_epoch})\n")
                return {"val_check_interval": n}
            return {"val_check_interval": v}
        v = int(v)
        # only disable the per-epoch check when the interval can exceed one epoch
        if v > self.batches_per_epoch:
            return {"val_check_interval": v, "check_val_every_n_epoch": None}
        return {"val_check_interval": v}

    def _epoch_progress(self):
        try:
            done = self.trainer.fit_loop.epoch_loop.batch_progress.current.processed
            total = self.trainer.num_training_batches
            if not math.isfinite(total) or total <= 0:
                total = self.batches_per_epoch          # estimate from total_units
            return done / total if total else 0.0
        except (RuntimeError, AttributeError, ZeroDivisionError):
            return 0.0

    def _flush_train_window(self, force=False):
        """Emit train metrics accumulated since the last flush.

        The gate is on `global_step`, which is identical on every rank. Never
        gate this on a locally-computed quantity (e.g. "did this window see
        enough samples"): `_sum_across_ranks` and `MetricCollection.compute()`
        are both collectives, so one rank skipping them hangs the job.
        """
        step = self.trainer.global_step
        if step <= self._last_flush_step:
            return                                     # empty window
        if not force and step - self._last_flush_step < self.min_window_steps:
            return
        self._last_flush_step = step

        # recall / balanced acc for this window only; composition stays epoch-level
        self._log_class_metrics("train", include_composition=False,
                                log_key="train_window", direct_step=step,
                                suffix=f"[window @ {self._epoch_progress():.2f}]")

        # torchmetrics collection: we own compute/reset here, so the subclass
        # must NOT also hand the object to log_dict (see compute_loss_and_metrics).
        tm = getattr(self, "train_metrics", None)      # base class has no such attr
        if tm is not None:
            out = tm.compute()                         # collective -> all ranks
            if self.logger is not None and self.trainer.is_global_zero:
                self.logger.log_metrics(
                    {k.replace("train/", "train_window/"):
                        (v.item() if torch.is_tensor(v) else float(v))
                     for k, v in out.items()},
                    step=step)
            tm.reset()                                 # all ranks

    def on_validation_epoch_start(self):
        """Close the train window exactly at the validation point."""
        if not self.align_train_metrics_to_val:
            return
        # nothing to flush during sanity check or a standalone trainer.validate()
        if self.trainer.sanity_checking or self.trainer.state.fn != "fit":
            return
        self._flush_train_window()

    # ---- forward ------------------------------------------------------------
    def forward(self, frames, seq_ids, slot_ids, lengths,**covariates):
        return self.model(frames, seq_ids, slot_ids, lengths, **covariates)

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
                self.log(f'lr/{name}', g['lr'], on_step=True, on_epoch=False)
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

        n = torch.bincount(t, minlength=k).tolist()
        hit = None
        if preds is not None:
            p = preds.detach().long().view(-1)
            hit = torch.bincount(t[p == t], minlength=k).tolist()

        # ===== FRACTIONAL LOGGING: dual accumulation =====
        # `_stats["train"]` is the sliding window (cleared on flush);
        # `_epoch_stats_train` survives flushes and carries the epoch totals.
        buckets = [self._stats[stage]]
        if stage == "train" and self.align_train_metrics_to_val:
            buckets.append(self._epoch_stats_train)
        for s in buckets:
            for i in range(k):
                s["n"][i] += n[i]
                if hit is not None:
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

    # ===== FRACTIONAL LOGGING: four new params on this method =====
    #   stats               which accumulator to read (None -> self._stats[stage])
    #   include_composition emit dataset counts/fractions/ratio (epoch-level only)
    #   log_key             log-name namespace, decoupled from the stats bucket
    #                       ("train_window" vs "train") so windowed and epoch
    #                       series don't collide and get mean-reduced together
    #   direct_step         write via logger.log_metrics at this step, bypassing
    #                       Lightning's per-epoch mean reduction
    #   suffix              extra text on the write_log line
    def _log_class_metrics(self, stage, stats=None, include_composition=True,
                           suffix="", log_key=None, direct_step=None):
        s = stats if stats is not None else self._stats[stage]
        # collectives first: every rank must reach these before any early return
        n = self._sum_across_ranks(s["n"])
        c = self._sum_across_ranks(s["correct"])
        k, total = len(n), sum(n)

        if total == 0:                                # empty window -> nothing to report
            if stats is None:
                self._stats[stage] = self._empty_stats()
            return

        key = log_key or stage
        recalls = [c[i] / n[i] if n[i] > 0 else float("nan") for i in range(k)]

        payload = {}
        if include_composition:
            for i in range(k):
                payload[f"class_counts/{key}_class_{i}_count"] = float(n[i])
                payload[f"class_counts/{key}_class_{i}_fraction"] = n[i] / total
            if k == 2:   # keep the familiar binary chart
                payload[f"class_counts/{key}_class_ratio"] = (
                    n[0] / n[1] if n[1] > 0 else float("inf"))
            else:
                payload[f"class_counts/{key}_imbalance_ratio"] = (
                    max(n) / min(n) if min(n) > 0 else float("inf"))

        for i, r in enumerate(recalls):
            if r == r:                                # skip NaN
                payload[f"class_counts/{key}_recall_class_{i}"] = r

        balanced_acc = (sum(recalls) / k
                        if all(r == r for r in recalls) else float("nan"))
        if balanced_acc == balanced_acc:
            payload[f"{key}/balanced_accuracy"] = balanced_acc

        # ===== FRACTIONAL LOGGING: two write paths =====
        if direct_step is not None:
            # straight to the logger at an explicit step -> no mean reduction
            if self.logger is not None and self.trainer.is_global_zero:
                self.logger.log_metrics(payload, step=direct_step)
        else:
            for name, v in payload.items():
                self.log(name, v, rank_zero_only=True,
                         prog_bar=(stage == "val" and name.endswith("balanced_accuracy")))

        counts_str = " ".join(f"n{i}={n[i]}" for i in range(k))
        rec_str = " ".join(f"recall{i}={recalls[i]:.4f}" for i in range(k))
        # ===== FRACTIONAL LOGGING: tag lines so repeated runs are distinguishable
        tag = f"[Epoch {self.current_epoch + 1}]"
        if stage == "val" and self.val_check_interval is not None:
            tag += f"[val #{self._val_run_idx} @ {self._epoch_progress():.2f}]"
        if suffix:
            tag += suffix
        self.write_log(f"{tag} {stage}: {counts_str} "
                       f"{rec_str} balanced_acc={balanced_acc:.4f}\n")

        if stats is None:                             # only clear the bucket we own
            self._stats[stage] = self._empty_stats()

    # ---- epoch hooks ------------------------------------------------------------
    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.write_log(f"Device of model at start of training: {next(self.model.parameters()).device}")
        self.train_sample_count = 0
        self._val_run_idx = 0          # FRACTIONAL LOGGING: numbering is per-epoch

    def _log_epoch_summary_extras(self):
        """Subclasses may append regime-specific lines to the epoch summary."""
        pass

    def on_train_epoch_end(self):

        if self.current_epoch in [0, 5]:
            WHITE = "\033[97m"
            RED = "\033[91m"
            RESET = "\033[0m"

            self.write_log(f"\n--- Epoch {self.current_epoch + 1} Summary ---")

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

        # ===== FRACTIONAL LOGGING: tail window, then epoch-level composition =====
        if self.align_train_metrics_to_val:
            # force=True so a short trailing window isn't silently dropped
            # (dropping it would leak counts into the next epoch)
            self._flush_train_window(force=True)
            self._log_class_metrics("train", stats=self._epoch_stats_train,
                                    suffix="[epoch]")
            self._epoch_stats_train = self._empty_stats()
        else:
            self._log_class_metrics("train")

    def on_validation_epoch_end(self):
        # ===== FRACTIONAL LOGGING: sanity-check guard + run counter =====
        # Without the guard the sanity-check pass pollutes the first real run's
        # counts. It matters more now that validation runs many times.
        if self.trainer.sanity_checking:
            self._stats["val"] = self._empty_stats()
            return
        self._val_run_idx += 1
        self._log_class_metrics("val")
    
    def _resolve_total_steps(self):
        try:
            total_steps = self.trainer.estimated_stepping_batches
        except (RuntimeError, AttributeError):
            # no trainer attached (unit tests, manual instantiation) -> fall back
            total_steps = self.total_steps
        # check before int(): int(float('inf')) raises (max_epochs=-1 gives inf)
        if not math.isfinite(total_steps) or total_steps <= 0:
            total_steps = self.total_steps
        return int(total_steps)

    # ---- optimizers ---------------------------------------------------------------
    def configure_optimizers(self):
        NO_DECAY_KEYS = (
            "layer_scale",                   # convnext (torchvision), [C,1,1]
            "gamma",                         # convnext under timm / original naming
            "relative_position_bias_table",  # swin v1
            "cpb_mlp",                       # swin v2 continuous position bias MLP
            "logit_scale",                   # swin v2
            "pos_embed", "pos_embedding", "position_embedding",
            "cls_token", "class_token", "dist_token",
        )


        def split_decay(named_params, extra_no_decay=()):
            """(decay, no_decay): exclude biases, 1-D norm params, and known
            scale/position-bias tensors from weight decay."""
            decay, no_decay = [], []
            keys = NO_DECAY_KEYS + tuple(extra_no_decay)
            for name, p in named_params:
                if not p.requires_grad:
                    continue
                if p.ndim <= 1 or name.endswith(".bias") or any(k in name for k in keys):
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
        elif self.lora_tuning:
            from timm.optim import param_groups_layer_decay
            from peft import PeftModel

            peft_backbone = next(m for m in self.model.modules() if isinstance(m, PeftModel))
            encoder = peft_backbone.get_base_model()   # timm model with LoRA layers injected

            nwd = encoder.no_weight_decay() if hasattr(encoder, "no_weight_decay") else set()
            enc_groups = param_groups_layer_decay(
                encoder,
                weight_decay=self.weight_decay,
                no_weight_decay_list=nwd,
                layer_decay=self.lora_lr_decay,
            )
            for g in enc_groups:
                g["lr"] = self.lr_backbone * g.pop("lr_scale", 1.0)
                g["lr_name"] = "lr_backbone"
            param_groups.extend(enc_groups)

            n_lrs = len({round(g["lr"], 12) for g in enc_groups})
            self.write_log(f"[configure_optimizers] LoRA: {len(enc_groups)} encoder groups, "
                        f"{n_lrs} distinct lrs\n")

            enc_ids = {id(p) for p in encoder.parameters()}
            rest = [(n, p) for n, p in self.model.named_parameters()
                    if p.requires_grad and id(p) not in enc_ids]
            add_group(rest, self.lr_classifier_head, "lr_head")
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
            total_steps = self._resolve_total_steps()
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
 
        if self.lr_scheduling == 'two_stage':
            return self._two_stage_schedule(optimizer)
 
        return {"optimizer": optimizer}
    
    def _two_stage_schedule(self, optimizer):
        total_steps = self._resolve_total_steps()
        switch = min(max(1, int(self.stage1_fraction * total_steps)), total_steps - 1)
        stage2_steps = total_steps - switch
        warm1 = max(1, int(self.warmup_fraction * switch))         # head warmup in stage 1
        warm2 = max(1, int(self.warmup_fraction * stage2_steps))   # backbone warmup in stage 2
        stage1_names = set(self.stage1_lr_names)
 
        if not any(g["lr_name"] in stage1_names for g in optimizer.param_groups):
            raise ValueError(
                f"two_stage: no optimizer group has lr_name in {sorted(stage1_names)}; "
                f"available: {sorted({g['lr_name'] for g in optimizer.param_groups})}"
            )
 
        def ramp(t, n):                       # same shape as LinearLR(start_factor=1e-2)
            return 1e-2 + (1.0 - 1e-2) * min(t / n, 1.0)
 
        def cosine(t, n, base_lr):            # multiplier version of CosineAnnealingLR(eta_min)
            floor = min(self.eta_min_cosine / base_lr, 1.0) if base_lr > 0 else 0.0
            progress = min(t / max(1, n), 1.0)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
 
        def head_lambda(base_lr):
            def f(step):
                if step < warm1:
                    return ramp(step, warm1)
                if step < switch:
                    return 1.0
                return cosine(step - switch, stage2_steps, base_lr)
            return f
 
        def backbone_lambda(base_lr):
            def f(step):
                if step < switch:
                    return 0.0
                t = step - switch
                if t < warm2:
                    return ramp(t, warm2)
                return cosine(t - warm2, stage2_steps - warm2, base_lr)
            return f
 
        # one lambda per param group, in order; g["lr"] is read before LambdaLR rescales it
        lambdas = [(head_lambda if g["lr_name"] in stage1_names else backbone_lambda)(g["lr"])
                   for g in optimizer.param_groups]
 
        # state used by on_train_batch_start to freeze/unfreeze
        self._switch_step = switch
        self._stage2_params = [p for g in optimizer.param_groups
                               if g["lr_name"] not in stage1_names for p in g["params"]]
        stage2_ids = {id(p) for p in self._stage2_params}
        self._stage2_bn = [m for m in self.model.modules()
                           if isinstance(m, nn.modules.batchnorm._BatchNorm)
                           and m.weight is not None and id(m.weight) in stage2_ids]
 
        self.write_log(
            f"[configure_optimizers] two_stage: total_steps={total_steps}, "
            f"stage 1 = steps [0, {switch}) training {sorted(stage1_names)}, "
            f"stage 2 = steps [{switch}, {total_steps}) training all; "
            f"{len(self._stage2_params)} params / {len(self._stage2_bn)} BN layers frozen in stage 1\n"
        )
 
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

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


        # --- per step prediction -------
        mixer = getattr(getattr(model, 'classifier', None), 'mixer', None)
        self.per_step = getattr(mixer, 'per_step', False)
        # unreduced twin of self.criterion, for per-subject normalization
        if num_classes == 1:
            self.criterion_none = (
                nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')
                if use_balanced_weights else nn.BCEWithLogitsLoss(reduction='none'))
        else:
            self.criterion_none = (
                nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
                if use_balanced_weights else nn.CrossEntropyLoss(reduction='none'))

        # Unweighted criterion, val-only, as a *calibration diagnostic*.
        # This is the proper NLL under val's actual class prior — NOT a
        # selection metric (its constant-predictor optimum is the base rate,
        # so it would reward the majority-leaning degenerate solution).
        self.criterion_unweighted = (nn.BCEWithLogitsLoss() if num_classes == 1
                                     else nn.CrossEntropyLoss())
        self.criterion_unweighted_none = (
            nn.BCEWithLogitsLoss(reduction='none') if num_classes == 1
            else nn.CrossEntropyLoss(reduction='none'))

        # kept for backward compatibility (macro recall == balanced accuracy)
        self.val_balanced_acc = MulticlassRecall(num_classes=n_eff, average="macro")

        # ---- pr-auc / f1(pos) / mcc as torchmetrics objects --------------------
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
        # FRACTIONAL LOGGING NOTE: windowing shrinks the AveragePrecision score
        # buffer by roughly the val fraction, so drop_train_pr_auc=False is
        # often affordable again once align_train_metrics_to_val is on.
        self.train_metrics = _build_metrics(n_eff, prefix="train/",
                                            include_pr_auc=not drop_train_pr_auc)
        self.val_metrics   = _build_metrics(n_eff, prefix="val/",
                                            include_pr_auc=True)

    # ---- per step predictions helpers ----------------------------------------
    def _unpack(self, outputs, batch):
        """-> (logits_for_loss, targets_for_loss, subject_logits)"""
        _, seq_ids, _, _, labels, *_ = batch
        if not self.per_step:
            tgt = labels.float().unsqueeze(1) if self.num_classes == 1 else labels.long()
            return outputs, tgt, outputs

        tok_logits, last = outputs
        tok_labels = labels[seq_ids]                        # (n_tok,)
        tgt = (tok_labels.float().unsqueeze(1) if self.num_classes == 1
               else tok_labels.long())
        return tok_logits, tgt, last

    def _subject_mean_loss(self, logits, targets, seq_ids, B, crit=None):
        """Mean over subjects of the mean token loss within each subject."""
        crit = crit or self.criterion_none
        per_tok = crit(logits, targets).float().view(-1)                  # (n_tok,)
        counts  = torch.bincount(seq_ids, minlength=B)                    # (B,)
        per_subj = torch.zeros(B, dtype=per_tok.dtype, device=per_tok.device)
        per_subj.scatter_add_(0, seq_ids, per_tok)
        nz = counts > 0
        return (per_subj[nz] / counts[nz]).mean()

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
    
    # ----- Covariates helpers ---------------------
    def forward(self, frames, seq_ids, slot_ids, lengths, global_properties=None, local_properties = None):
        if global_properties is None and local_properties is None:
            return self.model(frames, seq_ids, slot_ids, lengths)
        else:
            return self.model(frames, seq_ids, slot_ids, lengths,
                          global_properties=global_properties, local_properties=local_properties)

    def _get_covariates(self, batch):
        global_properties = getattr(batch, "global_properties", None)   # None instead of an error
        local_properties  = getattr(batch, "local_properties", None)    # None instead of an error

        return {'global_properties': global_properties, 'local_properties': local_properties}

    # ---- loss + metrics --------------------------------------------------------
    def compute_loss_and_metrics(self, batch, stage):
        frames, seq_ids, slot_ids, lengths, labels, *_ = batch
        bsz = labels.size(0)

        outputs = self(frames, seq_ids, slot_ids, lengths, **self._get_covariates(batch))
        logits, targets, subj_logits = self._unpack(outputs, batch)

        if stage == "train" and not self._guard_finite(logits, "outputs"):
            return None

        loss = (self._subject_mean_loss(logits, targets, seq_ids, bsz)
                if self.per_step else self.criterion(logits, targets))
        if stage == "train" and not self._guard_finite(loss, "loss"):
            return None

        if self.num_classes == 1:
            preds = (subj_logits > 0.0).long().view(-1)
        else:
            preds = torch.argmax(subj_logits, dim=1)
        acc = (preds == labels.long().view(-1)).float().mean()

        self._update_class_counts(labels, stage, preds=preds)

        # ===== FRACTIONAL LOGGING: per-step train curves =====
        # loss/acc need none of the windowing machinery — on_step gives a dense
        # curve that already reads against the mid-epoch val points.
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log(f"{stage}/acc",  acc,  on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True, batch_size=bsz)

        scores = self._scores_for_metrics(subj_logits)
        tgt = labels.long().view(-1)
        mcoll = self.train_metrics if stage == "train" else self.val_metrics
        mcoll.update(scores, tgt)
        # ===== FRACTIONAL LOGGING: single ownership of the train collection =====
        # In aligned mode `_flush_train_window` calls compute()/reset() itself.
        # Handing the object to log_dict as well would let Lightning reset it at
        # epoch end underneath us, so the epoch value would reflect only the
        # post-flush tail. Val is unaffected: Lightning resets it per val run,
        # which is exactly the desired mid-epoch behaviour.
        if stage == "val" or not self.align_train_metrics_to_val:
            self.log_dict(mcoll, on_step=False, on_epoch=True, batch_size=bsz)

        if stage == "train":
            self.train_sample_count += bsz
            self._log_lrs()
            return loss
        else:
            # unweighted (proper) val loss — diagnostic only.
            loss_unweighted = (
                self._subject_mean_loss(logits, targets, seq_ids, bsz,
                                        crit=self.criterion_unweighted_none)
                if self.per_step else
                self.criterion_unweighted(logits, targets))
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
            resizing_factors, subject_ids, modalities, *_ = batch

        outputs = self(frames, seq_ids, slot_ids, lengths, **self._get_covariates(batch))
        tok_logits, subj_logits = outputs if self.per_step else (None, outputs)

        def _probs_preds(z):
            if self.num_classes == 1:
                p1 = torch.sigmoid(z).flatten()
                return torch.stack([1 - p1, p1], dim=1), (z > 0.0).long().flatten()
            return torch.softmax(z, dim=1), torch.argmax(z, dim=1)

        probs, preds = _probs_preds(subj_logits)
        out = {"probs": probs.detach().cpu(),
               "preds": preds.detach().cpu(),
               "labels": labels.detach().cpu().flatten(),
               "subject_ids": subject_ids}

        if self.per_step:
            tok_probs, tok_preds = _probs_preds(tok_logits)
            out.update(tok_probs=tok_probs.detach().cpu(),
                       tok_preds=tok_preds.detach().cpu(),
                       tok_seq_ids=seq_ids.detach().cpu(),
                       tok_slot_ids=slot_ids.detach().cpu())
        return out
