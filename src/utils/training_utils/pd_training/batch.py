"""Typed view of the flat collate output.

The collate produces
    (frames, seq_ids, slot_ids, lengths, labels, resizing_factors, subject_ids, modalities)
optionally as a namedtuple-like object that also carries `global_properties`
and `local_properties`. `PDBatch.from_collate` accepts either and gives the
fields names, so the rest of the code never unpacks by position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

_POSITIONAL = ("frames", "seq_ids", "slot_ids", "lengths", "labels",
               "resizing_factors", "subject_ids", "modalities")
_REQUIRED = 5  # frames .. labels


@dataclass
class PDBatch:
    frames: torch.Tensor
    seq_ids: torch.Tensor
    slot_ids: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor
    resizing_factors: Any = None
    subject_ids: Any = None
    modalities: Any = None
    global_properties: Optional[torch.Tensor] = None
    local_properties: Optional[torch.Tensor] = None
    view_mask: Optional[torch.Tensor] = None

    @classmethod
    def from_collate(cls, batch) -> "PDBatch":
        if isinstance(batch, cls):
            return batch
        values = tuple(batch)
        if len(values) < _REQUIRED:
            raise ValueError(f"expected at least {_REQUIRED} batch fields "
                             f"({', '.join(_POSITIONAL[:_REQUIRED])}), got {len(values)}")
        fields = dict(zip(_POSITIONAL, values))
        # Covariates are only present when the collate returns an object with
        # these attributes; a plain tuple has none, so they stay None.
        fields["global_properties"] = getattr(batch, "global_properties", None)
        fields["local_properties"] = getattr(batch, "local_properties", None)
        fields["view_mask"] = getattr(batch, "view_mask", None)
        return cls(**fields)

    @property
    def size(self) -> int:
        """Number of subjects in the batch."""
        return self.labels.size(0)

    def model_args(self):
        return self.frames, self.seq_ids, self.slot_ids, self.lengths

    def covariates(self) -> dict:
        """Keyword arguments for the model. Empty when there are no covariates,
        so models whose forward takes no covariate kwargs still work."""
        if self.global_properties is None and self.local_properties is None:
            return {}
        return {"global_properties": self.global_properties,
                "local_properties": self.local_properties,
                "view_mask": self.view_mask}


def make_example_input(k, n_slots, n_views_frames=2, C=3, H=224, W=224):
    """Positional forward() inputs for Lightning's model summary."""
    T = min(n_views_frames, n_slots)
    frames = torch.randn(T, k, C, H, W)
    seq_ids = torch.zeros(T, dtype=torch.long)
    slot_ids = torch.arange(T, dtype=torch.long)
    lengths = torch.tensor([T])
    return (frames, seq_ids, slot_ids, lengths)
