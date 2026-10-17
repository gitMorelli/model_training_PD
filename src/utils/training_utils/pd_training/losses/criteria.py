"""Classification loss with optional class weighting and per-subject averaging."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def subject_mean(per_token: torch.Tensor, seq_ids: torch.Tensor, num_subjects: int) -> torch.Tensor:
    """Mean over subjects of the mean token loss within each subject, so
    subjects with many tokens don't dominate."""
    per_token = per_token.float().view(-1)
    counts = torch.bincount(seq_ids, minlength=num_subjects)
    per_subj = torch.zeros(num_subjects, dtype=per_token.dtype, device=per_token.device)
    per_subj.scatter_add_(0, seq_ids, per_token)
    nz = counts > 0
    return (per_subj[nz] / counts[nz]).mean()


class ClassificationLoss(nn.Module):
    """BCE-with-logits for a single-logit head (num_classes == 1), else CE.

    Weights are derived from class counts, which are stored in the module's
    hyperparameters, so they are registered as non-persistent buffers: they
    follow the module across devices but are not written to checkpoints.

    Call with `seq_ids` and `num_subjects` for per-token logits; the loss is
    then averaged per subject first (see `subject_mean`).
    """

    def __init__(self, num_classes: int,
                 class_weights: Optional[torch.Tensor] = None,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.binary = num_classes == 1
        self.register_buffer("weight", None if self.binary else class_weights, persistent=False)
        self.register_buffer("pos_weight", pos_weight if self.binary else None, persistent=False)

    def _elementwise(self, logits, targets, reduction):
        if self.binary:
            return F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction=reduction)
        return F.cross_entropy(logits, targets, weight=self.weight, reduction=reduction)

    def forward(self, logits, targets, seq_ids=None, num_subjects=None):
        if seq_ids is None:
            return self._elementwise(logits, targets, "mean")
        return subject_mean(self._elementwise(logits, targets, "none"), seq_ids, num_subjects)
