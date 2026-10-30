"""Data interface between your dataset and the interpretability pipeline.

The pipeline never touches your Dataset directly. It asks an adapter for one
subject/sequence at a time, by a stable string id, and gets back a SequenceSample
holding exactly what ModularModel.forward needs for that sequence.

Implement `DataAdapter` (3 methods) or wrap an existing torch Dataset with
`TorchDatasetAdapter` (see examples/adapter_template.py).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch


@dataclass
class SequenceSample:
    """One sequence (subject), exactly as the model sees it at evaluation time.

    frames:            (T, k, C, H, W) preprocessed tensors (eval transforms only!)
    slot_ids:          (T,) long, time slot of each token, in [0, n_slots)
    label:             int or None
    view_mask:         (T, k) bool, False = crop missing/imputed. None = all present
    local_properties:  (T, n_local) or None
    global_properties: (n_global,) or None
    meta:              anything you want carried into the outputs (site, device, ...)
    """
    seq_id: str
    frames: torch.Tensor
    slot_ids: torch.Tensor
    label: Optional[int] = None
    view_mask: Optional[torch.Tensor] = None
    local_properties: Optional[torch.Tensor] = None
    global_properties: Optional[torch.Tensor] = None
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.frames.shape[0])

    @property
    def k(self) -> int:
        return int(self.frames.shape[1])

    def validate(self, n_slots: Optional[int] = None):
        f = self.frames
        if f.dim() != 5:
            raise ValueError(f'{self.seq_id}: frames must be (T, k, C, H, W), got {tuple(f.shape)}')
        if self.slot_ids.shape != (f.shape[0],):
            raise ValueError(f'{self.seq_id}: slot_ids must be (T,)')
        if self.view_mask is not None and tuple(self.view_mask.shape) != tuple(f.shape[:2]):
            raise ValueError(f'{self.seq_id}: view_mask must be (T, k)')
        if n_slots is not None and int(self.slot_ids.max()) >= n_slots:
            raise ValueError(f'{self.seq_id}: slot id >= n_slots ({n_slots})')
        if len(set(self.slot_ids.tolist())) != len(self.slot_ids):
            raise ValueError(f'{self.seq_id}: duplicated slot ids')


class DataAdapter(ABC):
    """Minimal interface. Subclass and implement ids() and load()."""

    #: optional human names for the k crop positions, e.g. ['A0', 'A1', 'B0']
    crop_names: Optional[Sequence[str]] = None

    @abstractmethod
    def ids(self) -> list[str]:
        """All sequence ids available (validation set)."""

    @abstractmethod
    def load(self, seq_id: str) -> SequenceSample:
        """Return one sequence. Must be deterministic (no random augmentation,
        no random frame sampling)."""

    def to_display(self, img: torch.Tensor) -> np.ndarray:
        """(C, H, W) model-input tensor -> (H, W) or (H, W, 3) float array in [0, 1]
        for plotting. Override to undo your normalization exactly; the default
        min-max rescales per image."""
        x = img.detach().float().cpu()
        if x.shape[0] == 1:
            x = x[0]
        elif x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        else:
            x = x.mean(0)
        x = x - x.min()
        return (x / (x.max() + 1e-8)).numpy()


class TorchDatasetAdapter(DataAdapter):
    """Wrap an existing (eval-mode) torch Dataset.

    dataset:    your Dataset; dataset[i] returns whatever your collate expects
    id_fn:      i -> stable string id (e.g. lambda i: dataset.subject_ids[i])
    to_sample:  (item, seq_id) -> SequenceSample
    to_display: optional (C, H, W) -> array in [0, 1]
    """

    def __init__(self, dataset, id_fn: Callable[[int], str],
                 to_sample: Callable[[object, str], SequenceSample],
                 to_display: Optional[Callable] = None,
                 crop_names: Optional[Sequence[str]] = None):
        self.dataset = dataset
        self._to_sample = to_sample
        self._to_display = to_display
        self.crop_names = crop_names
        self._index = {str(id_fn(i)): i for i in range(len(dataset))}
        if len(self._index) != len(dataset):
            raise ValueError('id_fn does not give unique ids')

    def ids(self):
        return list(self._index)

    def load(self, seq_id):
        return self._to_sample(self.dataset[self._index[str(seq_id)]], str(seq_id))

    def to_display(self, img):
        if self._to_display is not None:
            return self._to_display(img)
        return super().to_display(img)


def check_determinism(adapter: DataAdapter, ids: Sequence[str], atol: float = 0.0) -> list[str]:
    """Load each id twice; return ids whose frames differ (random augmentation left on?)."""
    bad = []
    for sid in ids:
        a, b = adapter.load(sid), adapter.load(sid)
        if a.frames.shape != b.frames.shape or not torch.allclose(a.frames, b.frames, atol=atol):
            bad.append(sid)
    return bad
