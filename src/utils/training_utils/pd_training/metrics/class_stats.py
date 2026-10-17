"""Per-class counts and hits as a torchmetrics Metric.

Replaces the hand-rolled `_stats` dicts and `_sum_across_ranks`: torchmetrics
sums the states across DDP ranks inside `compute()`. That makes `compute()` a
collective, so every rank must call it, never conditionally on local data.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torchmetrics import Metric


class ClassStats(Metric):
    """n[k] = samples with true label k; correct[k] = those also predicted k.
    recall_k = correct[k] / n[k]; balanced accuracy = mean recall."""

    full_state_update = False
    higher_is_better = None
    is_differentiable = False

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state("n", torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("correct", torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, preds: Optional[torch.Tensor] = None):
        k = self.num_classes
        t = target.detach().long().view(-1)
        if t.numel() and (t.min() < 0 or t.max() >= k):
            raise ValueError(f"labels contain values outside [0, {k})")
        self.n += torch.bincount(t, minlength=k)
        if preds is not None:
            p = preds.detach().long().view(-1)
            self.correct += torch.bincount(t[p == t], minlength=k)

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.n, self.correct


def class_stats_payload(n: List[int], correct: List[int], key: str,
                        include_composition: bool = True) -> Tuple[Dict[str, float], List[float], float]:
    """Turn counts into loggable scalars.

    -> (payload, recalls, balanced_acc). Recalls are NaN for classes with no
    samples; NaN values are left out of the payload. Composition (counts,
    fractions, class ratio) describes the data rather than the model, which is
    why the windowed train series leaves it out.
    """
    k, total = len(n), sum(n)
    recalls = [correct[i] / n[i] if n[i] > 0 else float("nan") for i in range(k)]
    payload: Dict[str, float] = {}

    if include_composition:
        for i in range(k):
            payload[f"class_counts/{key}_class_{i}_count"] = float(n[i])
            payload[f"class_counts/{key}_class_{i}_fraction"] = n[i] / total
        if k == 2:
            payload[f"class_counts/{key}_class_ratio"] = n[0] / n[1] if n[1] > 0 else float("inf")
        else:
            payload[f"class_counts/{key}_imbalance_ratio"] = (
                max(n) / min(n) if min(n) > 0 else float("inf"))

    for i, r in enumerate(recalls):
        if r == r:   # skip NaN
            payload[f"class_counts/{key}_recall_class_{i}"] = r

    balanced_acc = sum(recalls) / k if all(r == r for r in recalls) else float("nan")
    if balanced_acc == balanced_acc:
        payload[f"{key}/balanced_accuracy"] = balanced_acc
    return payload, recalls, balanced_acc


def format_class_stats(stage: str, n: List[int], recalls: List[float], balanced_acc: float) -> str:
    counts = " ".join(f"n{i}={c}" for i, c in enumerate(n))
    recs = " ".join(f"recall{i}={r:.4f}" for i, r in enumerate(recalls))
    return f"{stage}: {counts} {recs} balanced_acc={balanced_acc:.4f}"
