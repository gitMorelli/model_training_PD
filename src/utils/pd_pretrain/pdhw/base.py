"""Base LightningModule shared by all pre-training objectives.

Subclasses must:
  * build every trainable module inside `self.net` (an nn.Module / ModuleDict),
  * call `self.init_ema()` at the end of __init__,
  * implement `compute_loss(batch, net)` -> (loss, dict_of_logs),
  * implement `extract_features(image, modality)` -> (B, D) tensor,
  * optionally implement `preview(batch)` -> dict of image tensors in [-1, 1].

The base class handles optimisation (AdamW, warmup + cosine), EMA of `self.net`
(used for validation, previews and feature extraction), and logging.
"""
from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel


class PretrainModule(L.LightningModule):
    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        betas: tuple[float, float] = (0.9, 0.99),
        warmup_steps: int = 1000,
        min_lr_ratio: float = 0.01,
        ema_decay: float = 0.999,
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = tuple(betas)
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.ema_decay = ema_decay
        self.ema: AveragedModel | None = None

    # ------------------------------------------------------------------ EMA
    def init_ema(self) -> None:
        if self.ema_decay > 0:
            max_decay = self.ema_decay

            def ema_avg(avg_p, p, n_averaged):
                # warmup: decay = min(max_decay, (1+n)/(10+n)), so early EMA
                # weights track the model instead of staying near initialisation
                n = float(n_averaged)
                d = min(max_decay, (1.0 + n) / (10.0 + n))
                return d * avg_p + (1.0 - d) * p

            self.ema = AveragedModel(self.net, avg_fn=ema_avg, use_buffers=True)
            for p in self.ema.parameters():
                p.requires_grad_(False)

    @property
    def eval_net(self) -> nn.Module:
        """Network used for validation / previews / features (EMA if enabled)."""
        return self.ema.module if self.ema is not None else self.net

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update_parameters(self.net)

    # ------------------------------------------------------- to implement
    def compute_loss(self, batch: dict, net: nn.Module) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor, modality: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def preview(self, batch: dict) -> dict[str, torch.Tensor]:
        return {}

    # ------------------------------------------------------------- steps
    def training_step(self, batch, batch_idx):
        loss, logs = self.compute_loss(batch, self.net)
        bs = batch["image"].shape[0]
        self.log("train/loss", loss, prog_bar=True, batch_size=bs)
        for k, v in logs.items():
            self.log(f"train/{k}", v, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.compute_loss(batch, self.eval_net)
        bs = batch["image"].shape[0]
        self.log("val/loss", loss, prog_bar=True, batch_size=bs, sync_dist=True)
        for k, v in logs.items():
            self.log(f"val/{k}", v, batch_size=bs, sync_dist=True)
        return loss

    # --------------------------------------------------------- optimiser
    def _param_groups(self):
        decay, no_decay = [], []
        for name, p in self.net.named_parameters():
            if not p.requires_grad:
                continue
            # no weight decay on biases, norms, embeddings, positional/special tokens
            if p.ndim <= 1 or "embed" in name or "token" in name or "pos" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self._param_groups(), lr=self.lr, betas=self.betas)
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = min(self.warmup_steps, max(1, total // 10))
        min_ratio = self.min_lr_ratio

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total - warmup))
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    # ------------------------------------------------------------ utils
    @staticmethod
    def check_batch(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        x, c = batch["image"], batch["modality"]
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"batch['image'] must be (B,1,H,W), got {tuple(x.shape)}")
        return x, c.long()
