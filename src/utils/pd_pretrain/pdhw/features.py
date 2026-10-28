"""Batch feature extraction and subject-level aggregation for the downstream PD classifier."""
from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from .base import PretrainModule
from .diffusion import DiffusionPretrainer


@torch.no_grad()
def extract_dataset(
    model: PretrainModule,
    loader: DataLoader,
    device: str | torch.device = "cpu",
    with_anomaly: bool = False,
    anomaly_kwargs: dict | None = None,
) -> dict:
    """Run `model.extract_features` over a loader.

    Returns {"features": (N, D) tensor, plus every other batch key (e.g. subject_id,
    label) concatenated / listed in the same order}. With a DiffusionPretrainer and
    with_anomaly=True, residual statistics are appended under "anomaly".
    """
    model = model.to(device).eval()
    out: dict[str, list] = defaultdict(list)
    for batch in loader:
        x = batch["image"].to(device)
        c = batch["modality"].to(device)
        out["features"].append(model.extract_features(x, c).cpu())
        if with_anomaly:
            if not isinstance(model, DiffusionPretrainer):
                raise TypeError("anomaly features are only available for DiffusionPretrainer")
            out["anomaly"].append(model.anomaly_features(x, c, **(anomaly_kwargs or {})).cpu())
        for k, v in batch.items():
            if k == "image":
                continue
            if torch.is_tensor(v):
                out[k].append(v.cpu())
            else:
                out[k].extend(v if isinstance(v, (list, tuple)) else [v])
    return {k: torch.cat(v) if v and torch.is_tensor(v[0]) else v for k, v in out.items()}


def aggregate_by_subject(features: torch.Tensor, subject_ids, stats=("mean", "std")) -> tuple[list, torch.Tensor]:
    """Per-subject [mean | std | ...] of sample-level features -> (subjects, (S, D*len(stats))).

    The PD label is per person, so classify subjects, not symbols, and keep every
    subject's samples inside a single CV fold.
    """
    ids = subject_ids.tolist() if torch.is_tensor(subject_ids) else list(subject_ids)
    groups: dict = defaultdict(list)
    for i, s in enumerate(ids):
        groups[s].append(i)
    subjects = sorted(groups)
    rows = []
    for s in subjects:
        f = features[groups[s]]
        parts = []
        for st in stats:
            if st == "mean":
                parts.append(f.mean(0))
            elif st == "std":
                parts.append(f.std(0, unbiased=False))
            elif st == "max":
                parts.append(f.max(0).values)
            else:
                raise ValueError(st)
        rows.append(torch.cat(parts))
    return subjects, torch.stack(rows)
