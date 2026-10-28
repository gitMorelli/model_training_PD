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
    def __init__(self, every_n_epochs: int = 5, num_samples: int = 16):
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples
        self._batch = None

    def _fixed_batch(self, trainer, pl_module):
        if self._batch is None:
            loaders = trainer.val_dataloaders or trainer.train_dataloader
            loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
            batch = next(iter(loader))
            self._batch = {k: v[: self.num_samples] for k, v in batch.items() if torch.is_tensor(v)}
        return {k: v.to(pl_module.device) for k, v in self._batch.items()}

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        if not trainer.is_global_zero:
            return
        rows = pl_module.preview(self._fixed_batch(trainer, pl_module))
        if not rows:
            return
        out = Path(trainer.log_dir or trainer.default_root_dir) / "previews"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"epoch{trainer.current_epoch + 1:04d}.png"
        save_rows(rows, path, max_cols=self.num_samples)
        exp = getattr(trainer.logger, "experiment", None)
        if exp is not None and hasattr(exp, "add_image"):
            img = np.asarray(Image.open(path))
            exp.add_image("preview", img[None], trainer.global_step)
