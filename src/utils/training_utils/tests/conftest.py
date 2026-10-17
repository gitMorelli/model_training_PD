import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from lightning.pytorch.loggers import Logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

K, C, H, W = 2, 1, 4, 4


class TinyHead(nn.Module):
    def __init__(self, dim, out_dim, per_step):
        super().__init__()
        self.fc = nn.Linear(dim, out_dim)
        self.mixer = SimpleNamespace(per_step=per_step)   # how the real model exposes it


class TinyModel(nn.Module):
    """Same call signature as the real model: frames (T, k, C, H, W) for T
    tokens, seq_ids mapping tokens to subjects."""

    def __init__(self, out_dim, per_step=False, batchnorm=False):
        super().__init__()
        layers = [nn.Linear(C * H * W, 8)]
        layers.append(nn.BatchNorm1d(8) if batchnorm else nn.LayerNorm(8))
        self.backbone = nn.Sequential(*layers)
        self.classifier = TinyHead(8, out_dim, per_step)
        self.per_step_mode = per_step

    def forward(self, frames, seq_ids, slot_ids, lengths):
        tok = self.backbone(frames.mean(1).flatten(1))
        B = lengths.numel()
        counts = torch.bincount(seq_ids, minlength=B).clamp(min=1).unsqueeze(1)
        subj = torch.zeros(B, tok.size(1), device=tok.device).index_add_(0, seq_ids, tok) / counts
        out = self.classifier.fc(subj)
        if self.per_step_mode:
            return self.classifier.fc(tok), out
        return out


def make_batches(n_batches, batch_size, n_labels, seed):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for b in range(n_batches):
        tokens_per = torch.randint(1, 4, (batch_size,), generator=g)
        seq_ids = torch.repeat_interleave(torch.arange(batch_size), tokens_per)
        T = seq_ids.numel()
        labels = (torch.arange(batch_size) + b) % n_labels        # every class in every batch
        frames = torch.randn(T, K, C, H, W, generator=g) + labels[seq_ids].view(-1, 1, 1, 1, 1) * 0.5
        slot_ids = torch.cat([torch.arange(int(t)) for t in tokens_per])
        batches.append((frames, seq_ids, slot_ids, tokens_per, labels,
                        torch.ones(batch_size), list(range(batch_size)), ["m"] * batch_size))
    return batches


class MemLogger(Logger):
    def __init__(self):
        super().__init__()
        self.records = {}

    @property
    def name(self):
        return "mem"

    @property
    def version(self):
        return 0

    def log_hyperparams(self, params, *args, **kwargs):
        pass

    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            self.records[(step, k)] = float(v)


@pytest.fixture
def mem_logger():
    return MemLogger()
