"""Trainer arguments derived from the same configs the module uses.

    trainer = L.Trainer(**trainer_kwargs(data_shape, eval_cfg),
                        callbacks=default_callbacks(sched_cfg) + [ModelCheckpoint(...)],
                        devices=..., strategy=..., logger=...)
"""
from __future__ import annotations

import logging
from typing import List

import lightning as L

from .callbacks import GradientCheck, NamedLRLogger, ParameterSummary, TwoStageFreeze
from .config import DataShape, EvalConfig, SchedulerConfig

log = logging.getLogger(__name__)


def val_check_kwargs(data_shape: DataShape, eval_cfg: EvalConfig) -> dict:
    """Trainer kwargs for validation frequency.

    `val_check_interval` counts dataloader batches, not optimizer steps: with
    accumulate_grad_batches=4, 0.25 is still a quarter of the fetched batches.
    """
    v = eval_cfg.val_check_interval
    if v is None:
        return {}
    if isinstance(v, float):
        if not 0.0 < v <= 1.0:
            raise ValueError("float val_check_interval must be in (0, 1]")
        if v == 1.0:
            return {}                       # Trainer default
        if data_shape.iterable_train_dataset:
            # no __len__: Lightning can't resolve a fraction, so convert to batches
            n = max(1, int(round(v * data_shape.batches_per_epoch)))
            log.info("iterable train dataset: val_check_interval %s -> %d batches "
                     "(batches_per_epoch~%d)", v, n, data_shape.batches_per_epoch)
            return {"val_check_interval": n}
        return {"val_check_interval": v}
    v = int(v)
    # only drop the per-epoch check when the interval can exceed one epoch
    if v > data_shape.batches_per_epoch:
        return {"val_check_interval": v, "check_val_every_n_epoch": None}
    return {"val_check_interval": v}


def trainer_kwargs(data_shape: DataShape, eval_cfg: EvalConfig) -> dict:
    return {"max_epochs": data_shape.num_epochs,
            "accumulate_grad_batches": data_shape.accumulate_grad_batches,
            **val_check_kwargs(data_shape, eval_cfg)}


def default_callbacks(sched_cfg, *, summary_epochs=(0, 5), color=False,
                      gradient_check=True, lr_logging=True) -> List[L.Callback]:
    callbacks: List[L.Callback] = [ParameterSummary(summary_epochs, color=color)]
    if gradient_check:
        callbacks.append(GradientCheck())
    if lr_logging:
        callbacks.append(NamedLRLogger())
    if sched_cfg == "two_stage":
        callbacks.append(TwoStageFreeze())
    return callbacks
