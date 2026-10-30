"""Preview callback: saves a grid of inputs / reconstructions / samples every N epochs.

Looking at these is the main sanity check that tremor-scale detail survives:
if reconstructions look smoother than inputs, the model is discarding PD signal.
"""
from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
from PIL import Image


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """[-1,1] with ink=+1  ->  uint8 with dark ink on white paper."""
    return ((1 - x.clamp(-1, 1)) * 127.5).round().byte().cpu().numpy()


def save_rows(rows: dict[str, torch.Tensor], path: Path, max_cols: int = 16, pad: int = 2) -> None:
    """One row per key, one column per sample."""
    tiles = []
    for v in rows.values():
        v = v[:max_cols, 0]
        n, h, w = v.shape
        row = np.full((h + pad, max_cols * (w + pad)), 200, np.uint8)
        for i, img in enumerate(to_uint8(v)):
            row[:h, i * (w + pad): i * (w + pad) + w] = img
        tiles.append(row)
    grid = np.concatenate(tiles, 0)
    scale = max(1, 256 // grid.shape[0] + 1) if grid.shape[0] < 256 else 1
    Image.fromarray(grid).resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST).save(path)


class PreviewCallback(L.Callback):
    """Save a preview grid every `every_n_val` validation runs.
 
    Works with epoch- or step-based validation (val_check_interval as an int,
    check_val_every_n_epoch: null). Files are named by optimizer step, so
    several validations in one epoch never overwrite each other.
 
    The preview batch is the first validation batch seen (stored once and reused
    for the whole run), so no extra iterator is opened on the val dataloader;
    with a WebDataset that would mean re-opening shards and starting workers.
    """
 
    def __init__(self, every_n_val: int = 1, num_samples: int = 16, out_dir: str | None = None):
        self.every_n_val = every_n_val
        self.num_samples = num_samples
        self.out_dir = out_dir
        self._batch = None
        self._n_val = 0
 
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self._batch is None and batch_idx == 0 and dataloader_idx == 0 and isinstance(batch, dict):
            self._batch = {k: v[: self.num_samples].detach().cpu() for k, v in batch.items() if torch.is_tensor(v)}
 
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._n_val += 1  # counted on every rank so all ranks agree
        if self._n_val % self.every_n_val or not trainer.is_global_zero or self._batch is None:
            return
        batch = {k: v.to(pl_module.device) for k, v in self._batch.items()}
        rows = pl_module.preview(batch)
        if not rows:
            return
        out = Path(self.out_dir) if self.out_dir else Path(trainer.log_dir or trainer.default_root_dir) / "previews"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"step{trainer.global_step:07d}.png"
        save_rows(rows, path, max_cols=self.num_samples)
        exp = getattr(trainer.logger, "experiment", None)
        if exp is not None and hasattr(exp, "add_image"):
            img = np.asarray(Image.open(path))
            exp.add_image("preview", img[None], trainer.global_step)
 
    # keep the counter and the fixed batch when resuming from a checkpoint
    def state_dict(self):
        return {"n_val": self._n_val, "batch": self._batch}
 
    def load_state_dict(self, state_dict):
        self._n_val = state_dict.get("n_val", 0)
        self._batch = state_dict.get("batch")