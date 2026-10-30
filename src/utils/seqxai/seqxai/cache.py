"""Stage 1: run the CNN once per sequence and cache everything lightweight.

Per sequence (one .pt file each, in <out>/cache/feats/):
    CNN features (T, k, F), slot ids, data view mask, properties, label, meta.
Global tables:
    <out>/cache/recomputed.csv   logits/probs from the cached-feature head path and from
                                 the full model.forward on frames, + the CSV probs and the
                                 differences (round-trip check)
    <out>/cache/embeddings.npz   pooled sequence embeddings z (pre-classifier) and the
                                 per-token embeddings q with their (seq, slot) index
Images are NOT stored: the adapter stays the single source of truth for pixels.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .adapter import DataAdapter
from .runner import HeadInputs, ModelRunner


def _fname(seq_id: str) -> str:
    safe = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in seq_id)[:80]
    return f'{safe}_{hashlib.md5(seq_id.encode()).hexdigest()[:8]}.pt'


def _identity(x):
    return x


class _LoadDS(torch.utils.data.Dataset):
    def __init__(self, adapter, ids):
        self.adapter, self.ids = adapter, list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.adapter.load(self.ids[i])


class FeatureCache:
    def __init__(self, root: str):
        self.root = root
        self.dir = os.path.join(root, 'feats')
        os.makedirs(self.dir, exist_ok=True)

    def path(self, seq_id):
        return os.path.join(self.dir, _fname(seq_id))

    def has(self, seq_id):
        return os.path.exists(self.path(seq_id))

    def save(self, item: HeadInputs, half: bool = False):
        cpu = lambda t: None if t is None else t.detach().cpu()
        f = item.feats.detach().cpu()
        torch.save(dict(seq_id=item.seq_id, feats=f.half() if half else f,
                        slot_ids=cpu(item.slot_ids), view_mask=cpu(item.view_mask),
                        local_properties=cpu(item.local_properties),
                        global_properties=cpu(item.global_properties),
                        label=item.label, meta=item.meta or {}), self.path(item.seq_id))

    def load(self, seq_id, device='cpu') -> HeadInputs:
        d = torch.load(self.path(seq_id), map_location=device, weights_only=False)
        return HeadInputs(d['seq_id'], d['feats'].float(), d['slot_ids'], d['view_mask'],
                          d['local_properties'], d['global_properties'], d['label'], d['meta'])

    def load_many(self, ids, device='cpu'):
        return [self.load(i, device) for i in ids]

    def ids(self):
        p = os.path.join(self.root, 'recomputed.csv')
        return pd.read_csv(p, dtype={'seq_id': str})['seq_id'].tolist() if os.path.exists(p) else []


def build_cache(runner: ModelRunner, adapter: DataAdapter, preds: Optional[pd.DataFrame],
                out_dir: str, ids: Optional[Sequence[str]] = None, num_workers: int = 0,
                half: bool = False, check_full_forward: bool = True, overwrite: bool = False,
                atol: float = 1e-3, verbose: bool = True) -> pd.DataFrame:
    """Compute and store features + predictions for every sequence; compare with the CSV."""
    cache = FeatureCache(os.path.join(out_dir, 'cache'))
    ids = list(ids if ids is not None else (preds.index if preds is not None else adapter.ids()))
    n_slots = runner.head.n_slots

    loader = torch.utils.data.DataLoader(_LoadDS(adapter, ids), batch_size=None,
                                         num_workers=num_workers, collate_fn=_identity)
    rows, Z, Q, Qidx = [], [], [], []
    for i, sample in enumerate(loader):
        sample.validate(n_slots)
        if cache.has(sample.seq_id) and not overwrite:
            item = cache.load(sample.seq_id, runner.device)
        else:
            item = runner.features(sample)
            cache.save(item, half=half)
            if half:
                item = cache.load(sample.seq_id, runner.device)
        lg = runner.logits([item])
        p = runner.probs(lg)[0].cpu().numpy()
        row = dict(seq_id=sample.seq_id, T=item.T, label=item.label,
                   **{f'logit_{c}': float(v) for c, v in enumerate(lg[0].cpu().numpy())},
                   **{f'rprob_{c}': float(v) for c, v in enumerate(p)},
                   rpred=int(p.argmax()))
        if check_full_forward:                      # the real model on the real frames
            with torch.no_grad():
                f = sample.frames.to(runner.device)
                N = f.shape[0]
                full = runner.model(
                    f, torch.zeros(N, dtype=torch.long, device=runner.device),
                    sample.slot_ids.to(runner.device).long(), torch.tensor([N], device=runner.device),
                    local_properties=None if sample.local_properties is None else sample.local_properties.to(runner.device),
                    global_properties=None if sample.global_properties is None else sample.global_properties[None].to(runner.device),
                    view_mask=None if sample.view_mask is None else sample.view_mask.to(runner.device))
                full = runner._final(full)
            row['full_vs_cache_maxdiff'] = float((runner.probs(full) - runner.probs(lg)).abs().max())
        for kk, v in (sample.meta or {}).items():
            if np.isscalar(v):
                row[f'meta_{kk}'] = v
        rows.append(row)
        Z.append(runner.pooled([item])[0].cpu().numpy())
        q = runner.token_embeddings(item).cpu().numpy()
        Q.append(q)
        Qidx.append(np.stack([np.full(item.T, i), item.slot_ids.cpu().numpy()], 1))
        if verbose and (i + 1) % 50 == 0:
            print(f'[cache] {i + 1}/{len(ids)}')

    df = pd.DataFrame(rows).set_index('seq_id')
    if preds is not None:
        common = df.index.intersection(preds.index)
        for c in range(runner.n_classes):
            df.loc[common, f'csv_prob_{c}'] = preds.loc[common, f'prob_{c}']
        diff = np.max(np.abs(df.loc[common, [f'rprob_{c}' for c in range(runner.n_classes)]].values
                             - df.loc[common, [f'csv_prob_{c}' for c in range(runner.n_classes)]].values), 1)
        df.loc[common, 'csv_vs_cache_maxdiff'] = diff
        if 'label' in preds:
            lab_mismatch = (preds.loc[common, 'label'] != df.loc[common, 'label']) & df.loc[common, 'label'].notna()
            df.loc[common, 'label_mismatch'] = lab_mismatch
    df.to_csv(os.path.join(cache.root, 'recomputed.csv'))
    np.savez_compressed(os.path.join(cache.root, 'embeddings.npz'),
                        ids=np.array(df.index, dtype=object), z=np.stack(Z),
                        q=np.concatenate(Q), q_index=np.concatenate(Qidx))

    report = dict(n=len(df))
    if 'full_vs_cache_maxdiff' in df:
        report['full_vs_cache_maxdiff'] = float(df['full_vs_cache_maxdiff'].max())
    if 'csv_vs_cache_maxdiff' in df:
        d = df['csv_vs_cache_maxdiff'].dropna()
        report.update(csv_vs_cache_maxdiff=float(d.max()) if len(d) else None,
                      csv_mismatches=int((d > atol).sum()),
                      csv_missing=int(len(ids) - len(d)))
    if 'label_mismatch' in df:
        report['label_mismatches'] = int(df['label_mismatch'].fillna(False).sum())
    with open(os.path.join(cache.root, 'roundtrip.json'), 'w') as fh:
        json.dump(report, fh, indent=2)
    if verbose:
        print('[cache] round-trip check:', report)
        if report.get('csv_mismatches'):
            print(f'[cache] WARNING: {report["csv_mismatches"]} sequences differ from the CSV by '
                  f'> {atol}. Check transforms, frame sampling, checkpoint, dropout/eval mode.')
    return df


def load_embeddings(out_dir: str) -> dict:
    d = np.load(os.path.join(out_dir, 'cache', 'embeddings.npz'), allow_pickle=True)
    return {k: d[k] for k in d.files}
