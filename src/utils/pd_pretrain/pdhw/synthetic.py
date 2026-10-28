"""Synthetic handwriting-like data, ONLY for smoke-testing the pipeline.

Each synthetic subject has a writing size and a tremor amplitude; 'PD' subjects
write smaller with more tremor. Symbols: loops/hooks (digit), V-strokes
(checkmark), looped cursive (text). Emits the same batch contract as your loader.
"""
from __future__ import annotations

import math

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _symbol(mod: int, rng: np.random.Generator, n: int = 240) -> np.ndarray:
    t = np.linspace(0, 1, n)
    if mod == 0:  # digit-like: loop, hooked line or S-curve
        kind = rng.integers(3)
        if kind == 0:
            a = 2 * math.pi * t * rng.uniform(0.9, 1.05)
            pts = np.stack([0.5 + 0.28 * np.sin(a), 0.5 - 0.38 * np.cos(a)], 1)
        elif kind == 1:
            pts = np.stack([0.5 + 0.1 * np.clip(0.25 - t, 0, None) * 8 - 0.1, 0.12 + 0.76 * t], 1)
        else:
            pts = np.stack([0.5 + 0.25 * np.sin(2 * math.pi * t), 0.12 + 0.76 * t], 1)
    elif mod == 1:  # checkmark
        k = np.array([[0.18, 0.52], [0.42, 0.8], [0.84, 0.18]]) + rng.normal(0, 0.03, (3, 2))
        s1 = int(n * 0.3)
        pts = np.concatenate([np.linspace(k[0], k[1], s1), np.linspace(k[1], k[2], n - s1)])
    else:  # cursive loops ("eeee")
        k = rng.uniform(5, 8)
        u = t * 2 * math.pi * k
        pts = np.stack([0.1 + 0.8 * t + 0.05 * np.sin(u), 0.5 - 0.15 * np.cos(u)], 1)
    return pts


def _render(pts: np.ndarray, size: int, width: float, pressure: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    grid = np.stack([xx.ravel(), yy.ravel()], 1)  # (HW, 2)
    p = pts * (size - 1)
    d2 = ((grid[:, None, :] - p[None]) ** 2).sum(-1)  # (HW, P)
    val = (np.exp(-d2 / (2 * width ** 2)) * pressure[None]).max(1)
    return val.reshape(size, size)


class SyntheticHandwriting(Dataset):
    def __init__(self, subjects: list[dict], samples_per_subject: int, image_size: int,
                 grid_n: int = 1, seed: int = 0):
        rng = np.random.default_rng(seed)
        cell = image_size // grid_n
        self.items = []
        for s in subjects:
            for _ in range(samples_per_subject):
                mod = int(rng.integers(3))
                img = np.zeros((image_size, image_size), np.float32)
                for gy in range(grid_n):
                    for gx in range(grid_n):
                        pts = _symbol(mod, rng)
                        # micrographia: shrink around the centre
                        pts = 0.5 + (pts - 0.5) * s["size"] * rng.uniform(0.9, 1.1)
                        # tremor: perpendicular oscillation along the stroke + jitter
                        tang = np.gradient(pts, axis=0)
                        normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
                        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
                        phase = rng.uniform(0, 2 * math.pi)
                        osc = np.sin(np.linspace(0, s["tremor_freq"], len(pts)) + phase)
                        pts = pts + normal * (s["tremor"] * osc)[:, None] + rng.normal(0, 0.003, pts.shape)
                        pressure = np.clip(s["pressure"] + rng.normal(0, 0.05, len(pts)), 0.3, 1.0)
                        img[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell] = _render(
                            pts, cell, width=max(0.6, 0.035 * cell), pressure=pressure)
                self.items.append({
                    "image": torch.from_numpy(img * 2 - 1)[None],
                    "modality": torch.tensor(mod),
                    "subject_id": torch.tensor(s["id"]),
                    "label": torch.tensor(s["pd"]),
                    "tremor": torch.tensor(s["tremor"], dtype=torch.float32),
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class SyntheticDataModule(L.LightningDataModule):
    """mode='single' -> one symbol per image; mode='grid' -> grid_n x grid_n symbols."""

    def __init__(self, image_size: int = 32, mode: str = "single", grid_n: int = 2,
                 n_subjects: int = 40, samples_per_subject: int = 24, val_frac: float = 0.2,
                 batch_size: int = 64, num_workers: int = 0, seed: int = 0):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        hp = self.hparams
        rng = np.random.default_rng(hp.seed)
        subjects = []
        for i in range(hp.n_subjects):
            pd = int(i % 2)
            subjects.append({
                "id": i, "pd": pd,
                "size": rng.uniform(0.55, 0.75) if pd else rng.uniform(0.7, 0.9),
                "tremor": rng.uniform(0.006, 0.02) if pd else rng.uniform(0.0, 0.006),
                "tremor_freq": rng.uniform(40, 70),
                "pressure": rng.uniform(0.6, 0.85) if pd else rng.uniform(0.75, 1.0),
            })
        order = rng.permutation(hp.n_subjects)
        n_val = max(2, int(hp.val_frac * hp.n_subjects))
        val_ids = set(order[:n_val].tolist())  # split by SUBJECT, never by sample
        grid_n = 1 if hp.mode == "single" else hp.grid_n
        self.train_set = SyntheticHandwriting([s for s in subjects if s["id"] not in val_ids],
                                              hp.samples_per_subject, hp.image_size, grid_n, hp.seed)
        self.val_set = SyntheticHandwriting([s for s in subjects if s["id"] in val_ids],
                                            hp.samples_per_subject, hp.image_size, grid_n, hp.seed + 1)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def predict_dataloader(self):
        return self.val_dataloader()
