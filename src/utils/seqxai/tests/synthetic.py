"""Synthetic task with a KNOWN answer, to check that seqxai recovers it.

Subjects have 3-8 tokens at random slots in [0, 10); each token has k=4 crops:
modality 'pen' (2 crops), 'tablet' (1), 'scan' (1). 32x32 grey images of random strokes.

Planted signal: label-1 subjects have a bright 5x5 square in a 'pen' crop at slots >= 5.
Distractor: squares also appear in 'scan' crops for both classes (not predictive).
Global property 0 is weakly predictive, property 1 is noise. ~8% of crops are missing.

Expected findings: 'pen' dominates modality Shapley; late slots dominate token
importance for class 1; CAMs land on the square; the 'scan' square is ignored.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model_def import ModularModel  # noqa: E402

from seqxai import DataAdapter, SequenceSample  # noqa: E402

GROUPS = {'pen': 2, 'tablet': 1, 'scan': 1}
K, N_SLOTS, S = 4, 10, 32


def strokes(rng):
    img = rng.normal(0, 0.05, (S, S))
    for _ in range(rng.integers(2, 5)):
        y0, x0, y1, x1 = rng.integers(0, S, 4)
        n = 40
        ys, xs = np.linspace(y0, y1, n).astype(int), np.linspace(x0, x1, n).astype(int)
        img[ys, xs] += 0.6
    return img


def square(img, rng):
    y, x = rng.integers(4, S - 9, 2)
    img[y:y + 5, x:x + 5] = 2.0
    return (y, x)


def make_subject(i, rng):
    label = int(rng.random() < 0.5)
    T = int(rng.integers(3, 9))
    slots = np.sort(rng.choice(N_SLOTS, T, replace=False))
    frames = np.zeros((T, K, 1, S, S), np.float32)
    mask = rng.random((T, K)) > 0.08
    boxes = {}
    for t, sl in enumerate(slots):
        for j in range(K):
            frames[t, j, 0] = strokes(rng)
        if label == 1 and sl >= 5 and rng.random() < 0.85:
            j = int(rng.integers(0, 2))
            boxes[(t, j)] = square(frames[t, j, 0], rng)
            mask[t, j] = True
        if rng.random() < 0.4:
            boxes[(t, 3)] = square(frames[t, 3, 0], rng)
    frames[~mask] = 0
    glob = np.array([label * 0.6 + rng.normal(0, 1), rng.normal(0, 1)], np.float32)
    return dict(id=f'subj{i:04d}', label=label, frames=torch.tensor(frames),
                slots=torch.tensor(slots), mask=torch.tensor(mask), glob=torch.tensor(glob),
                boxes=boxes, site=['A', 'B', 'C'][i % 3])


def make_data(n=420, seed=0):
    rng = np.random.default_rng(seed)
    return [make_subject(i, rng) for i in range(n)]


def vision():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1), nn.Flatten())


def build(seq_model='attnpool', n_out=2):
    return ModularModel(vision(), 64, N_SLOTS, n_out, d_model=64, seq_model=seq_model,
                        view_agg='hierarchical', view_groups=GROUPS, n_global=2,
                        view_kwargs=dict(intra='attention', inter='attention', modality_dropout=0.2))


def collate(subs):
    frames = torch.cat([s['frames'] for s in subs])
    seq = torch.cat([torch.full((len(s['slots']),), b) for b, s in enumerate(subs)])
    return dict(frames=frames, seq_ids=seq, slot_ids=torch.cat([s['slots'] for s in subs]),
                lengths=torch.tensor([len(s['slots']) for s in subs]),
                global_properties=torch.stack([s['glob'] for s in subs]),
                view_mask=torch.cat([s['mask'] for s in subs]))


def train(model, data, epochs=25, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    y_all = torch.tensor([s['label'] for s in data])
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(data))
        tot = 0
        for a in range(0, len(data), 16):
            idx = perm[a:a + 16].tolist()
            b = collate([data[i] for i in idx])
            out = model(b['frames'], b['seq_ids'], b['slot_ids'], b['lengths'],
                        global_properties=b['global_properties'], view_mask=b['view_mask'])
            out = out[1] if isinstance(out, tuple) else out
            y = y_all[idx]
            loss = (nn.functional.cross_entropy(out, y) if out.shape[1] > 1 else
                    nn.functional.binary_cross_entropy_with_logits(out[:, 0], y.float()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0:
            print(f'epoch {ep + 1} loss {tot / len(data):.3f}')
    return model.eval()


@torch.no_grad()
def predict_csv(model, data, path):
    rows = []
    for s in data:
        b = collate([s])
        out = model(b['frames'], b['seq_ids'], b['slot_ids'], b['lengths'],
                    global_properties=b['global_properties'], view_mask=b['view_mask'])
        out = out[1] if isinstance(out, tuple) else out
        p = out.softmax(-1)[0, 1].item() if out.shape[1] > 1 else torch.sigmoid(out[0, 0]).item()
        rows.append(dict(subject=s['id'], y_true=s['label'], prob_pd=p))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


class SynthAdapter(DataAdapter):
    crop_names = ['pen0', 'pen1', 'tablet', 'scan']

    def __init__(self, data):
        self.d = {s['id']: s for s in data}

    def ids(self):
        return list(self.d)

    def load(self, sid):
        s = self.d[sid]
        return SequenceSample(sid, s['frames'], s['slots'], s['label'], s['mask'],
                              None, s['glob'], meta=dict(site=s['site']))

    def to_display(self, img):
        return np.clip(img[0].numpy() / 2.0, 0, 1)
