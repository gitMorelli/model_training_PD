"""Freezing for the two-stage schedule.

The two_stage scheduler already gives stage-2 groups a learning rate of 0
during stage 1. On its own that still (a) computes their gradients,
(b) lets AdamW accumulate moment estimates for them, and (c) updates
BatchNorm running statistics. This callback freezes those parameters and
puts their BatchNorm layers in eval mode until the switch step.

DDP note: parameters that stop receiving gradients need
`DDPStrategy(find_unused_parameters=True)`; static_graph=True is incompatible.
"""
from __future__ import annotations

import lightning as L

from ..logging_utils import rank_zero_logger

info, _ = rank_zero_logger(__name__)


class TwoStageFreeze(L.Callback):

    def __init__(self):
        self._frozen = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        plan = getattr(pl_module, "two_stage_plan", None)
        if plan is None:
            return
        if trainer.global_step < plan.switch_step:
            if not self._frozen:
                for p in plan.stage2_params:
                    p.requires_grad_(False)
                self._frozen = True
                info(f"two_stage: froze {len(plan.stage2_params)} params until step {plan.switch_step}")
            # Lightning puts the model back in train mode every epoch and after
            # each validation run, so re-apply this every batch.
            for bn in plan.stage2_batchnorms:
                bn.eval()
        elif self._frozen:
            for p in plan.stage2_params:
                p.requires_grad_(True)
            for bn in plan.stage2_batchnorms:
                bn.train()
            self._frozen = False
            info(f"two_stage: unfroze stage-2 params at step {trainer.global_step}")
