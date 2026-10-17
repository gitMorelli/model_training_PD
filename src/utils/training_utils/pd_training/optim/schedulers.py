"""Learning-rate schedules. All of them step once per optimizer step."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def resolve_total_steps(estimated: Optional[float], fallback: int) -> int:
    """Trainer's estimate when it is usable, else `fallback`.

    The estimate is None without a trainer (unit tests) and inf for an
    iterable loader or max_epochs=-1; int(inf) would raise, so check first.
    """
    if estimated is None or not math.isfinite(estimated) or estimated <= 0:
        return int(fallback)
    return int(estimated)


def as_lightning_scheduler(scheduler) -> dict:
    return {"scheduler": scheduler, "interval": "step", "frequency": 1}


# ---- cosine with linear warmup ------------------------------------------------
def cosine_with_warmup(optimizer, total_steps: int, warmup_fraction: float, eta_min: float):
    warmup_steps = max(1, int(warmup_fraction * total_steps))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ---- two-stage: head first, then everything ------------------------------------
@dataclass
class TwoStagePlan:
    """What the TwoStageFreeze callback needs to freeze stage-2 parameters
    during stage 1."""
    switch_step: int
    stage2_params: List[nn.Parameter] = field(default_factory=list)
    stage2_batchnorms: List[nn.Module] = field(default_factory=list)


def _ramp(t, n):
    """Same shape as LinearLR(start_factor=1e-2)."""
    return 1e-2 + (1.0 - 1e-2) * min(t / n, 1.0)


def _cosine(t, n, base_lr, eta_min):
    """Multiplier form of CosineAnnealingLR(eta_min)."""
    floor = min(eta_min / base_lr, 1.0) if base_lr > 0 else 0.0
    progress = min(t / max(1, n), 1.0)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def two_stage(optimizer, model: nn.Module, total_steps: int, *,
              warmup_fraction: float, eta_min: float,
              stage1_fraction: float, stage1_lr_names: Sequence[str]):
    """Stage 1, steps [0, switch): only groups named in `stage1_lr_names`
    train (warmup, then constant). Stage 2, [switch, total): all groups,
    the others with their own warmup, everyone on a cosine decay.

    Returns (LambdaLR, TwoStagePlan).
    """
    switch = min(max(1, int(stage1_fraction * total_steps)), total_steps - 1)
    stage2_steps = total_steps - switch
    warm1 = max(1, int(warmup_fraction * switch))          # head warmup in stage 1
    warm2 = max(1, int(warmup_fraction * stage2_steps))    # backbone warmup in stage 2
    stage1_names = set(stage1_lr_names)

    available = {g["lr_name"] for g in optimizer.param_groups}
    if not stage1_names & available:
        raise ValueError(f"two_stage: no optimizer group has lr_name in "
                         f"{sorted(stage1_names)}; available: {sorted(available)}")

    def head_lambda(base_lr):
        def f(step):
            if step < warm1:
                return _ramp(step, warm1)
            if step < switch:
                return 1.0
            return _cosine(step - switch, stage2_steps, base_lr, eta_min)
        return f

    def backbone_lambda(base_lr):
        def f(step):
            if step < switch:
                return 0.0
            t = step - switch
            if t < warm2:
                return _ramp(t, warm2)
            return _cosine(t - warm2, stage2_steps - warm2, base_lr, eta_min)
        return f

    # one lambda per group, in order; g["lr"] is read before LambdaLR rescales it
    lambdas = [(head_lambda if g["lr_name"] in stage1_names else backbone_lambda)(g["lr"])
               for g in optimizer.param_groups]

    stage2_params = [p for g in optimizer.param_groups
                     if g["lr_name"] not in stage1_names for p in g["params"]]
    stage2_ids = {id(p) for p in stage2_params}
    stage2_bn = [m for m in model.modules()
                 if isinstance(m, nn.modules.batchnorm._BatchNorm)
                 and m.weight is not None and id(m.weight) in stage2_ids]
    plan = TwoStagePlan(switch, stage2_params, stage2_bn)

    log.info("two_stage: total_steps=%d, stage 1 = steps [0, %d) training %s, "
             "stage 2 = steps [%d, %d) training all; %d params / %d BN layers in stage 2 only",
             total_steps, switch, sorted(stage1_names), switch, total_steps,
             len(stage2_params), len(stage2_bn))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas), plan
