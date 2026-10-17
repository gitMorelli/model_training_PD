"""Configuration dataclasses.

They are frozen so they can be used as default arguments safely, and they are
plain data so `save_hyperparameters` can store them in checkpoints.

`DataShape` and `EvalConfig` are read both by the LightningModule and by
`trainer_factory`, which is what keeps the Trainer arguments and the module's
own assumptions from drifting apart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union


@dataclass(frozen=True)
class DataShape:
    """How much data one epoch contains and how it is batched.

    `total_units` counts whatever the loader batches: subjects for the flat
    loader, groups for the grouped loader.
    """
    total_units: int
    batch_size: int = 32
    world_size: int = 1
    accumulate_grad_batches: int = 1
    num_epochs: int = 10
    # True when the train dataloader has no __len__. Lightning then cannot
    # resolve a fractional val_check_interval or estimate the number of steps,
    # so both are derived from `total_units` instead. True is the safe default:
    # the derived values are also valid for a sized loader.
    iterable_train_dataset: bool = True

    @property
    def batches_per_epoch(self) -> int:
        return math.ceil(self.total_units / (self.batch_size * self.world_size))

    @property
    def steps_per_epoch(self) -> int:
        """Optimizer steps per epoch."""
        return math.ceil(self.batches_per_epoch / self.accumulate_grad_batches)

    @property
    def total_steps(self) -> int:
        return self.num_epochs * self.steps_per_epoch


@dataclass(frozen=True)
class OptimGroup:
    """Explicit optimizer group: every trainable parameter whose lower-cased
    name contains one of `names` gets learning rate `lr`. First match wins."""
    names: Sequence[str]
    lr: float
    lr_name: str


@dataclass(frozen=True)
class OptimConfig:
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    # Parameter-to-group strategy, checked in this order:
    #   groups non-empty -> explicit groups (unmatched trainable params are an error)
    #   lora_tuning      -> timm layer-wise lr decay on the PEFT encoder + head
    #   otherwise        -> names containing any of `head_keys` are the head
    groups: Sequence[OptimGroup] = ()
    lora_tuning: bool = False
    lora_lr_decay: float = 0.75
    head_keys: Sequence[str] = ("classifier",)


@dataclass(frozen=True)
class SchedulerConfig:
    kind: str = "cosine"              # "cosine" | "two_stage" | "none"
    warmup_fraction: float = 0.1
    eta_min: float = 1e-6
    # two_stage only: train the groups named in `stage1_lr_names` for the first
    # `stage1_fraction` of steps, then everything.
    stage1_fraction: float = 0.2
    stage1_lr_names: Sequence[str] = ("lr_head",)

    def __post_init__(self):
        if self.kind not in ("cosine", "two_stage", "none"):
            raise ValueError(f"unknown scheduler kind {self.kind!r}; "
                             "expected 'cosine', 'two_stage' or 'none'")


@dataclass(frozen=True)
class LossConfig:
    loss_name: str = "standard"  # standard | silent
    use_balanced_weights: bool = True
    # 0.0 -> no balancing, 1.0 -> full inverse-frequency balancing
    balancing_factor: float = 1.0
    # informational only (whether the sampler already balances the data)
    balanced_data: bool = False

    #for silent loss
    pi: float = 0.1
    learnable_pi: bool = False


@dataclass(frozen=True)
class EvalConfig:
    """When validation runs and how train metrics are reported against it.

    `val_check_interval` is a Trainer argument; build the Trainer with
    `trainer_factory.trainer_kwargs(...)` so it uses this same value.
      None  -> validate at epoch end
      float -> fraction of an epoch, in (0, 1]
      int   -> every N dataloader batches (not optimizer steps)

    `align_train_metrics_to_val` emits a windowed train metric at every
    validation point ("train_window/..."), so train and val curves share an
    x-axis. It only has an effect when validation runs mid-epoch.
    """
    val_check_interval: Optional[Union[float, int]] = None
    align_train_metrics_to_val: bool = False
    min_window_steps: int = 50        # don't emit a window shorter than this many optimizer steps
    # AveragePrecision keeps every score in memory; skip it on train by default.
    # With windowing on, the buffer is much smaller, so False is often affordable.
    drop_train_pr_auc: bool = True

    @property
    def aligned(self) -> bool:
        return self.align_train_metrics_to_val and self.val_check_interval is not None
