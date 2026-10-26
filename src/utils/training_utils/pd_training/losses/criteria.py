"""Classification loss with optional class weighting and per-subject averaging."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def subject_mean(per_token: torch.Tensor, seq_ids: torch.Tensor, num_subjects: int) -> torch.Tensor:
    """Mean over subjects of the mean token loss within each subject, so
    subjects with many tokens don't dominate."""
    per_token = per_token.float().view(-1)
    counts = torch.bincount(seq_ids, minlength=num_subjects)
    per_subj = torch.zeros(num_subjects, dtype=per_token.dtype, device=per_token.device)
    per_subj.scatter_add_(0, seq_ids, per_token)
    nz = counts > 0
    return (per_subj[nz] / counts[nz]).mean()

def silent_prior(prevalence: float, silent_fraction: float) -> float:
    """pi = P(y=1 | no visible signs): the share of positives among normal-looking cases.
 
    `prevalence` is P(y=1) in the training data; `silent_fraction` is the share of
    positives whose condition leaves no visible trace in the input.
    """
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"prevalence must be in (0, 1), got {prevalence}")
    if not 0.0 < silent_fraction < 1.0:
        raise ValueError(f"silent_fraction must be in (0, 1), got {silent_fraction}")
    silent = prevalence * silent_fraction
    return silent / (silent + 1.0 - prevalence)


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

class SilentPositiveLoss(nn.Module):
    """BCE for a single-logit head when some positives are invisible in the input.
 
    The logit models s(x) = P(condition visibly present | x), and the observed
    label is modelled as
 
        P(y=1 | x) = s(x) + (1 - s(x)) * pi,    P(y=0 | x) = (1 - s(x)) * (1 - pi),
 
    with pi = P(y=1 | no visible signs) (see `silent_prior`). This assumes visible
    signs imply y=1 and that silent positives are distributed like negatives.
    Positives the model scores as normal then contribute a vanishing gradient
    instead of being pushed up, so the model is not driven to fit them through
    spurious cues. Computed in log space for numerical stability.
 
    pi is fixed by default and stored as a non-persistent buffer, like the class
    weights in `ClassificationLoss`. With `learnable_pi=True` it becomes a
    parameter initialised at `pi`; add this module's parameters to the optimizer.
    A per-sample `pi` tensor (broadcastable to `logits`) can also be passed to
    `forward`, e.g. from an out-of-fold estimate; for per-token logits index it
    with `seq_ids` first.
 
    Targets may be soft (floats in [0, 1]). `pos_weight` scales the positive term
    as in BCE-with-logits. Call with `seq_ids` and `num_subjects` for per-token
    logits, as with `ClassificationLoss`.
 
    At inference, sigmoid(logits) is s(x), not P(y=1 | x): use `label_prob` for
    calibrated probabilities. Rank metrics such as AUROC are unaffected, since
    P(y=1 | x) is monotone in s(x).
    """
 
    def __init__(self, pi: float, learnable_pi: bool = False,
                 pos_weight: Optional[torch.Tensor] = None, eps: float = 1e-6):
        super().__init__()
        self.binary = True
        if not 0.0 < pi < 1.0:
            raise ValueError(f"pi must be in (0, 1), got {pi}")
        logit_pi = torch.tensor(math.log(pi / (1.0 - pi)))
        if learnable_pi:
            self.logit_pi = nn.Parameter(logit_pi)
        else:
            self.register_buffer("logit_pi", logit_pi, persistent=False)
        self.register_buffer("pos_weight", pos_weight, persistent=False)
        self.eps = eps
 
    @property
    def pi(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_pi)
 
    def _log_pi(self, dtype, pi: Optional[torch.Tensor]):
        if pi is None:
            logit_pi = self.logit_pi.to(dtype)
            return F.logsigmoid(logit_pi), F.logsigmoid(-logit_pi)
        pi = pi.to(dtype).clamp(self.eps, 1.0 - self.eps)
        return torch.log(pi), torch.log1p(-pi)
 
    def _elementwise(self, logits, targets, pi=None):
        if logits.shape != targets.shape:
            raise ValueError(f"logits {tuple(logits.shape)} and targets "
                             f"{tuple(targets.shape)} must have the same shape")
        logits = logits.float()
        targets = targets.to(logits.dtype)
        log_pi, log_1m_pi = self._log_pi(logits.dtype, pi)
 
        log_s = F.logsigmoid(logits)        # log s
        log_1m_s = F.logsigmoid(-logits)    # log (1 - s)
        log_p1 = torch.logaddexp(log_s, log_1m_s + log_pi)   # log(s + (1 - s) pi)
        log_p0 = log_1m_s + log_1m_pi                         # log((1 - s)(1 - pi))
 
        pos_term = targets * log_p1
        if self.pos_weight is not None:
            pos_term = pos_term * self.pos_weight
        return -(pos_term + (1.0 - targets) * log_p0)
 
    def forward(self, logits, targets, seq_ids=None, num_subjects=None, pi=None):
        loss = self._elementwise(logits, targets, pi)
        if seq_ids is None:
            return loss.mean()
        return subject_mean(loss, seq_ids, num_subjects)
 
    def visible_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """s(x): probability that the condition is visibly present."""
        return torch.sigmoid(logits)
 
    def label_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """P(y=1 | x) = s + (1 - s) * pi: calibrated probability of the condition."""
        s = torch.sigmoid(logits)
        return s + (1.0 - s) * self.pi.to(s.dtype)