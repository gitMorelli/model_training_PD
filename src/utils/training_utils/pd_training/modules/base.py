"""Shared LightningModule for the flat and grouped regimes.

What lives here is what both regimes need inside the training loop itself:
forward, the optimizer/scheduler wiring, and per-class recall / balanced
accuracy reporting (including the windowed train series). Everything else is
elsewhere: param groups and schedules in `optim/`, metrics in `metrics/`,
diagnostics (parameter table, gradient check, LR logging, two-stage freezing)
in `callbacks/`, Trainer arguments in `trainer_factory`.

Subclasses implement `compute_loss_and_metrics(batch, stage)` and, inside it,
call `update_class_stats(labels, preds, stage)`.

Windowed train metrics
----------------------
With `eval_cfg.aligned`, every mid-epoch validation run first closes a train
"window" and writes its recall / balanced accuracy (plus the subclass's score
metrics) as "train_window/..." at the current global step. They are written
straight to the logger because Lightning mean-reduces repeated self.log calls
within an epoch, which would collapse the windows into one point. Epoch-level
train numbers are still reported under "train/..." at epoch end.
"""
from __future__ import annotations

import logging
import math
import warnings
from typing import Optional

import lightning as L
import torch
import torch.nn as nn

from ..config import DataShape, EvalConfig, OptimConfig, SchedulerConfig
from ..logging_utils import rank_zero_logger
from ..losses.class_weights import n_stat_classes
from ..metrics.class_stats import ClassStats, class_stats_payload, format_class_stats
from ..optim import (as_lightning_scheduler, build_param_groups, cosine_with_warmup,
                     resolve_total_steps, two_stage)

log = logging.getLogger(__name__)
info, _ = rank_zero_logger(__name__)


class PDModuleBase(L.LightningModule):

    def __init__(self, model: nn.Module, data_shape: DataShape, num_classes: int = 1,
                 optim_cfg: OptimConfig = OptimConfig(),
                 sched_cfg: SchedulerConfig = SchedulerConfig(),
                 eval_cfg: EvalConfig = EvalConfig(),
                 example_input_array=None):
        super().__init__()
        self.model = model
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.optim_cfg = optim_cfg
        self.sched_cfg = sched_cfg
        self.eval_cfg = eval_cfg
        if example_input_array is not None:
            self.example_input_array = example_input_array

        k = n_stat_classes(num_classes)
        # epoch-level per-class counts, one per stage. (Not an nn.ModuleDict:
        # the key "train" would clash with nn.Module.train.)
        self.train_class_stats = ClassStats(k)
        self.val_class_stats = ClassStats(k)
        # windowed train counts, cleared at every validation point
        self.train_window_stats = ClassStats(k) if eval_cfg.aligned else None

        self.two_stage_plan = None     # set by configure_optimizers for sched_cfg.kind == "two_stage"
        self.train_samples_seen = 0    # per epoch; read by the ParameterSummary callback
        self._val_run_idx = 0          # which validation run within the current epoch
        self._last_flush_step = 0      # global_step at the last train-window flush

    # ---- forward / steps ---------------------------------------------------------
    def forward(self, frames, seq_ids, slot_ids, lengths, **covariates):
        return self.model(frames, seq_ids, slot_ids, lengths, **covariates)

    def compute_loss_and_metrics(self, batch, stage: str):
        """Return the loss (or None to skip the step) for stage == "train";
        the return value is ignored for stage == "val"."""
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self.compute_loss_and_metrics(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.compute_loss_and_metrics(batch, "val")

    # ---- hooks for subclasses ---------------------------------------------------------
    def train_score_metrics(self):
        """Train-side MetricCollection to include in the windowed series, or
        None. In aligned mode the base class owns its compute()/reset(), so the
        subclass must not also pass it to self.log_dict."""
        return None

    def on_val_class_stats(self, payload: dict):
        """Called after each validation run's class stats are logged."""

    def describe(self) -> list:
        """Extra lines for the ParameterSummary callback."""
        return []

    # ---- helpers for subclasses ----------------------------------------------------
    def update_class_stats(self, labels: torch.Tensor, preds: Optional[torch.Tensor], stage: str):
        self.class_stats(stage).update(labels, preds)
        if stage == "train" and self.train_window_stats is not None:
            self.train_window_stats.update(labels, preds)

    def class_stats(self, stage: str) -> ClassStats:
        return self.train_class_stats if stage == "train" else self.val_class_stats

    def guard_finite(self, tensor: torch.Tensor, what: str) -> bool:
        if not torch.isfinite(tensor).all():
            log.warning("NaN/inf in %s at step %d. Skipping.", what, self.trainer.global_step)
            return False
        return True

    # ---- class-stat reporting ---------------------------------------------------------
    def _epoch_progress(self) -> float:
        try:
            done = self.trainer.fit_loop.epoch_loop.batch_progress.current.processed
            total = self.trainer.num_training_batches
            if not math.isfinite(total) or total <= 0:
                total = self.data_shape.batches_per_epoch
            return done / total if total else 0.0
        except (RuntimeError, AttributeError, ZeroDivisionError):
            return 0.0

    def _report_class_stats(self, metric: ClassStats, stage: str, key: str, *,
                            include_composition: bool = True,
                            direct_step: Optional[int] = None, suffix: str = ""):
        """Compute (a collective: every rank gets here), log, reset.
        Returns the logged payload, or None if the metric saw no samples."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*was called before the ``update``")
            n, correct = metric.compute()
        n, correct = n.tolist(), correct.tolist()
        metric.reset()
        if sum(n) == 0:
            return None

        payload, recalls, balanced_acc = class_stats_payload(n, correct, key, include_composition)
        if direct_step is not None:
            # straight to the logger at an explicit step: no epoch mean-reduction
            if self.logger is not None and self.trainer.is_global_zero:
                self.logger.log_metrics(payload, step=direct_step)
        else:
            # values are already summed over ranks, so every rank logs the same
            # number and callbacks can monitor them on any rank
            for name, v in payload.items():
                self.log(name, v, prog_bar=(stage == "val" and name.endswith("balanced_accuracy")))

        tag = f"[Epoch {self.current_epoch + 1}]"
        if stage == "val" and self.eval_cfg.val_check_interval is not None:
            tag += f"[val #{self._val_run_idx} @ {self._epoch_progress():.2f}]"
        info(f"{tag}{suffix} {format_class_stats(stage, n, recalls, balanced_acc)}")
        return payload

    def _flush_train_window(self, force: bool = False):
        """Emit train metrics accumulated since the last flush.

        Gated on global_step, which is identical on every rank. Never gate this
        on a locally computed quantity: compute() is a collective, so a rank
        that skips it hangs the job.
        """
        step = self.trainer.global_step
        if step <= self._last_flush_step:
            return                                     # no optimizer step since last flush
        if not force and step - self._last_flush_step < self.eval_cfg.min_window_steps:
            return
        self._last_flush_step = step

        self._report_class_stats(self.train_window_stats, "train", "train_window",
                                 include_composition=False, direct_step=step,
                                 suffix=f"[window @ {self._epoch_progress():.2f}]")
        scores = self.train_score_metrics()
        if scores is not None:
            out = scores.compute()
            if self.logger is not None and self.trainer.is_global_zero:
                self.logger.log_metrics(
                    {k.replace("train/", "train_window/", 1):
                        (v.item() if torch.is_tensor(v) else float(v)) for k, v in out.items()},
                    step=step)
            scores.reset()

    # ---- epoch hooks ------------------------------------------------------------------
    def on_train_epoch_start(self):
        self.train_samples_seen = 0
        self._val_run_idx = 0

    def on_validation_epoch_start(self):
        """Close the train window exactly at the validation point."""
        if self.train_window_stats is None:
            return
        # nothing to flush during the sanity check or a standalone trainer.validate()
        if self.trainer.sanity_checking or self.trainer.state.fn != "fit":
            return
        self._flush_train_window()

    def on_train_epoch_end(self):
        if self.train_window_stats is not None:
            # force: a short trailing window would otherwise leak into the next epoch
            self._flush_train_window(force=True)
            self._report_class_stats(self.train_class_stats, "train", "train", suffix="[epoch]")
        else:
            self._report_class_stats(self.train_class_stats, "train", "train")

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            # don't let the sanity-check batches pollute the first real run
            self.val_class_stats.reset()
            return
        self._val_run_idx += 1
        payload = self._report_class_stats(self.val_class_stats, "val", "val")
        if payload is not None:
            self.on_val_class_stats(payload)

    # ---- optimizers -----------------------------------------------------------------
    def _estimated_stepping_batches(self):
        try:
            return self.trainer.estimated_stepping_batches
        except (RuntimeError, AttributeError):   # no trainer attached
            return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(build_param_groups(self.model, self.optim_cfg))
        cfg = self.sched_cfg
        if cfg.kind == "none":
            return {"optimizer": optimizer}

        total_steps = resolve_total_steps(self._estimated_stepping_batches(),
                                          self.data_shape.total_steps)
        info(f"scheduler '{cfg.kind}': total_steps={total_steps}")
        if cfg.kind == "cosine":
            scheduler = cosine_with_warmup(optimizer, total_steps, cfg.warmup_fraction, cfg.eta_min)
        else:
            scheduler, self.two_stage_plan = two_stage(
                optimizer, self.model, total_steps,
                warmup_fraction=cfg.warmup_fraction, eta_min=cfg.eta_min,
                stage1_fraction=cfg.stage1_fraction, stage1_lr_names=cfg.stage1_lr_names)
        return {"optimizer": optimizer, "lr_scheduler": as_lightning_scheduler(scheduler)}
