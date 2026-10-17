"""Class-count validation and balancing weights (pure functions)."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


def n_stat_classes(num_classes: int) -> int:
    """Number of distinct labels: 2 for a single-logit BCE head, else num_classes."""
    return 2 if num_classes <= 1 else num_classes


def validate_class_counts(class_counts: Sequence[float],
                          num_classes: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """-> (counts as float tensor, num_classes).

    num_classes == 1 means a single-logit BCE head and needs 2 counts
    [n_neg, n_pos]; num_classes >= 2 needs num_classes counts. If num_classes
    is None it is inferred from the number of counts (a CE head).
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if counts.ndim != 1 or counts.numel() < 2:
        raise ValueError("class_counts must be a 1-D sequence with >= 2 entries "
                         "(e.g. [n_class0, n_class1, ...]).")
    if num_classes is None:
        num_classes = counts.numel()
    expected = n_stat_classes(num_classes)
    if counts.numel() != expected:
        raise ValueError(f"Expected {expected} class counts for num_classes={num_classes}, "
                         f"got {counts.numel()}.")
    return counts, num_classes


def inverse_frequency_weights(counts: torch.Tensor, balancing_factor: float = 1.0) -> torch.Tensor:
    """sklearn-style w_c = total / (k * count_c), interpolated toward 1.0:
    balancing_factor 0 -> all ones, 1 -> full inverse frequency."""
    raw = counts.sum() / (counts.numel() * counts.clamp(min=1.0))
    return 1.0 + balancing_factor * (raw - 1.0)


def binary_pos_weight(counts: torch.Tensor, balancing_factor: float = 1.0) -> torch.Tensor:
    """BCE pos_weight = n_neg / n_pos (interpolated toward 1.0), shape (1,)."""
    if counts.numel() != 2:
        raise ValueError("pos_weight is only defined for 2 classes")
    raw = counts[0] / counts[1] if counts[1] > 0 else torch.tensor(1.0)
    return (1.0 + balancing_factor * (raw - 1.0)).reshape(1).to(torch.float32)
