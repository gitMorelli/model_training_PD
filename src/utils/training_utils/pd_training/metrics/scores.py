"""Logits -> predictions / probabilities, and the score-based metric collection.

A single-logit BCE head (num_classes == 1) and a 2-logit CE head are both
treated as binary problems: predictions threshold the positive class at 0.5,
and the metrics are the Binary* torchmetrics.
"""
from __future__ import annotations

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAveragePrecision, BinaryF1Score, BinaryMatthewsCorrCoef,
    MulticlassAveragePrecision, MulticlassF1Score, MulticlassMatthewsCorrCoef,
)

from ..losses.class_weights import n_stat_classes


def logits_to_preds(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(N,) hard predictions. logit > 0 is the same test as sigmoid > 0.5."""
    if num_classes == 1:
        return (logits > 0.0).long().view(-1)
    return torch.argmax(logits, dim=1)


def probs_and_preds(logits: torch.Tensor, num_classes: int):
    """-> ((N, C) probabilities, (N,) predictions); C == 2 for a BCE head."""
    logits = logits.float()
    if num_classes == 1:
        p1 = torch.sigmoid(logits).flatten()
        return torch.stack([1 - p1, p1], dim=1), (logits > 0.0).long().flatten()
    return torch.softmax(logits, dim=1), torch.argmax(logits, dim=1)


def scores_for_metrics(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Scores in the shape the collection from `build_score_metrics` expects:
    binary -> (N,) positive-class probability; multiclass -> (N, C)."""
    logits = logits.float()
    if n_stat_classes(num_classes) == 2:
        if num_classes == 1:
            return torch.sigmoid(logits).reshape(-1)
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.softmax(logits, dim=1)


def build_score_metrics(k: int, prefix: str, include_pr_auc: bool = True) -> MetricCollection:
    """PR-AUC, F1 and MCC. Binary (k == 2): F1 and PR-AUC of class 1.
    Multiclass: macro averages, so "f1_pos" is macro-F1."""
    metrics = {}
    if k == 2:
        if include_pr_auc:
            metrics["pr_auc"] = BinaryAveragePrecision()
        metrics["f1_pos"] = BinaryF1Score()
        metrics["mcc"] = BinaryMatthewsCorrCoef()
    else:
        if include_pr_auc:
            metrics["pr_auc"] = MulticlassAveragePrecision(num_classes=k, average="macro")
        metrics["f1_pos"] = MulticlassF1Score(num_classes=k, average="macro")
        metrics["mcc"] = MulticlassMatthewsCorrCoef(num_classes=k)
    # fresh instances per call: metrics are stateful
    return MetricCollection(metrics, prefix=prefix)
