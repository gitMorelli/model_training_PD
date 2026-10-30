"""Thin driver around ModularModel / SequenceModularHead.

Everything in the pipeline goes through `ModelRunner`, which only uses public pieces of
your model: `model._features`, `classifier.encode_tokens`, `classifier.mix`,
`classifier.forward_prefixes`, `classifier.build_view_mask` and
`classifier.view_pool.modality_view_mask`. So the analyses stay exact with respect to
the real forward pass: cached-feature head calls reproduce model.forward() (checked in
the cache stage).

Key idea: the CNN processes each crop independently (in eval mode), so its features
can be computed once and cached. Every temporal / modality / crop analysis then only
re-runs the cheap head, and pixel-level methods only re-run the CNN on the crops being
explained.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .adapter import SequenceSample


@dataclass
class HeadInputs:
    """One sequence in feature space: what the head needs, with the CNN already run."""
    seq_id: str
    feats: torch.Tensor                    # (T, k, F) CNN features (zeros for skipped crops)
    slot_ids: torch.Tensor                 # (T,)
    view_mask: Optional[torch.Tensor]      # (T, k) bool from the data (None = all present)
    local_properties: Optional[torch.Tensor] = None   # (T, n_local)
    global_properties: Optional[torch.Tensor] = None  # (n_global,)
    label: Optional[int] = None
    meta: Optional[dict] = None

    @property
    def T(self):
        return int(self.feats.shape[0])

    @property
    def k(self):
        return int(self.feats.shape[1])

    def base_mask(self) -> torch.Tensor:
        if self.view_mask is None:
            return torch.ones(self.T, self.k, dtype=torch.bool, device=self.feats.device)
        return self.view_mask.bool()

    def select_tokens(self, idx) -> 'HeadInputs':
        """Keep only tokens `idx` (removing the others from the sequence)."""
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.feats.device)
        return HeadInputs(self.seq_id, self.feats[idx], self.slot_ids[idx],
                          None if self.view_mask is None else self.view_mask[idx],
                          None if self.local_properties is None else self.local_properties[idx],
                          self.global_properties, self.label, self.meta)

    def with_slots(self, slot_ids) -> 'HeadInputs':
        return HeadInputs(self.seq_id, self.feats, torch.as_tensor(slot_ids, device=self.feats.device),
                          self.view_mask, self.local_properties, self.global_properties,
                          self.label, self.meta)

    def to(self, device):
        mv = lambda t: None if t is None else t.to(device)
        return HeadInputs(self.seq_id, self.feats.to(device), self.slot_ids.to(device),
                          mv(self.view_mask), mv(self.local_properties), mv(self.global_properties),
                          self.label, self.meta)


@dataclass
class Batch:
    feats: torch.Tensor          # (N*k, F)
    N: int
    k: int
    seq_ids: torch.Tensor        # (N,)
    slot_ids: torch.Tensor       # (N,)
    lengths: torch.Tensor        # (B,)
    view_mask: Optional[torch.Tensor]   # (N, k) or None
    local_properties: Optional[torch.Tensor]
    global_properties: Optional[torch.Tensor]  # (B, n_global)

    @property
    def B(self):
        return int(self.lengths.numel())


def grad_context(device):
    """cuDNN RNNs refuse backward() in eval mode; disable cuDNN while taking gradients."""
    if torch.device(device).type == 'cuda':
        return torch.backends.cudnn.flags(enabled=False)
    return contextlib.nullcontext()


class ModelRunner:
    def __init__(self, model, device='cpu', crop_names: Optional[Sequence[str]] = None,
                 max_tokens_per_call: int = 20_000):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.head = model.classifier
        self.max_tokens = max_tokens_per_call
        self.n_out = int(self.head.head.out_features)
        self.n_classes = 2 if self.n_out == 1 else self.n_out
        self.per_step = bool(getattr(self.head.mixer, 'per_step', False))
        self._crop_names = list(crop_names) if crop_names is not None else None
        self._k = None

    # ------------------------------------------------------------------ modalities
    @property
    def modality_aware(self) -> bool:
        return bool(getattr(self.head, 'modality_aware', False))

    @property
    def modality_names(self) -> list[str]:
        if not self.modality_aware:
            return ['all']
        names = list(self.head.view_pool.names)
        allowed = self.head.modalities
        if allowed is None:
            return names
        keep = {names[m] if not isinstance(m, str) else m for m in allowed}
        return [n for n in names if n in keep]

    def modality_crop_mask(self, modalities, k: int) -> torch.Tensor:
        """(k,) bool: crops belonging to the given modality subset."""
        if not self.modality_aware or modalities is None:
            return torch.ones(k, dtype=torch.bool, device=self.device)
        if len(modalities) == 0:
            return torch.zeros(k, dtype=torch.bool, device=self.device)
        m = self.head.view_pool.modality_view_mask(list(modalities), self.device)
        return m.reshape(-1).bool()

    def crop_modality(self, k: int) -> list[str]:
        if not self.modality_aware:
            return ['all'] * k
        out = ['?'] * k
        for name in self.head.view_pool.names:
            for j in self.modality_crop_mask([name], k).nonzero().flatten().tolist():
                out[j] = name
        return out

    def crop_names(self, k: int) -> list[str]:
        if self._crop_names is not None:
            return self._crop_names
        mods = self.crop_modality(k)
        seen, out = {}, []
        for m in mods:
            out.append(f'{m}[{seen.get(m, 0)}]')
            seen[m] = seen.get(m, 0) + 1
        return out

    # ------------------------------------------------------------------ targets
    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        if self.n_out == 1:
            p1 = torch.sigmoid(logits[:, 0])
            return torch.stack([1 - p1, p1], 1)
        return logits.softmax(-1)

    def score(self, logits: torch.Tensor, target) -> torch.Tensor:
        """Explained quantity: log-odds of the target class vs the rest,
            logit_c - logsumexp(logit_j, j != c)      (= logit_1 - logit_0 for 2 logits,
                                                        = +/- logit for a 1-output head)
        Not the raw logit_c: softmax ignores shifts common to all logits, so logit_c alone
        can drop while P(c) rises, which makes signed attributions meaningless. Not the
        probability either: its saturation flattens gradients."""
        target = torch.as_tensor(target, device=logits.device).long().reshape(-1)
        if target.numel() == 1 and logits.shape[0] > 1:
            target = target.expand(logits.shape[0])
        if self.n_out == 1:
            return logits[:, 0] * (2 * target.float() - 1)
        own = logits.gather(1, target[:, None])[:, 0]
        others = logits.scatter(1, target[:, None], float('-inf'))
        return own - torch.logsumexp(others, 1)

    # ------------------------------------------------------------------ CNN
    @torch.no_grad()
    def features(self, sample: SequenceSample, chunk_tokens: int = 64) -> HeadInputs:
        """Run the CNN exactly as ModularModel.forward does (same masking/skip rules)."""
        frames = sample.frames
        T, k = frames.shape[:2]
        vm = None if sample.view_mask is None else sample.view_mask.to(self.device).bool()
        mask = self.head.build_view_mask(vm, self.head.modalities, T, k, self.device)
        out = []
        for a in range(0, T, chunk_tokens):
            fr = frames[a:a + chunk_tokens].to(self.device)
            m = None if mask is None else mask[a:a + chunk_tokens]
            out.append(self.model._features(fr, m).float())
        feats = torch.cat(out).view(T, k, -1)
        mv = lambda t: None if t is None else t.to(self.device)
        return HeadInputs(sample.seq_id, feats, sample.slot_ids.to(self.device).long(), vm,
                          mv(sample.local_properties), mv(sample.global_properties),
                          sample.label, sample.meta)

    def vision(self, images: torch.Tensor) -> torch.Tensor:
        """(n, C, H, W) -> (n, F) features, gradients allowed."""
        return self.model.vision_model(images.to(self.device))

    # ------------------------------------------------------------------ batching
    def make_batch(self, items: Sequence[HeadInputs],
                   masks: Optional[Sequence[Optional[torch.Tensor]]] = None,
                   feats: Optional[Sequence[torch.Tensor]] = None) -> Batch:
        """Stack sequences into the model's flat token layout.

        masks: optional extra (T, k) bool masks per item (AND-ed with the data mask);
               used for crop / modality occlusion.
        feats: optional replacement (T, k, F) features per item (e.g. requiring grad).
        Masked crops get zero features when the model skips masked views, exactly as
        ModularModel._features does.
        """
        dev = self.device
        k = items[0].k
        f_list, m_list, s_list, sl_list, loc_list, glob_list = [], [], [], [], [], []
        any_mask = False
        for b, it in enumerate(items):
            f = it.feats if feats is None else feats[b]
            m = it.base_mask()
            if masks is not None and masks[b] is not None:
                m = m & masks[b].to(m.device).bool()
            allowed = self.head.build_view_mask(None, self.head.modalities, 1, k, dev)
            if allowed is not None:
                m = m & allowed
            if not bool(m.all()):
                any_mask = True
                if getattr(self.model, 'skip_masked_views', True):
                    f = f * m[..., None].to(f.dtype)
            f_list.append(f.reshape(it.T * k, -1))
            m_list.append(m)
            s_list.append(torch.full((it.T,), b, dtype=torch.long, device=dev))
            sl_list.append(it.slot_ids.long())
            loc_list.append(it.local_properties)
            glob_list.append(it.global_properties)
        N = sum(it.T for it in items)
        local = None if loc_list[0] is None else torch.cat(loc_list).to(dev)
        glob = None if glob_list[0] is None else torch.stack(glob_list).to(dev)
        return Batch(torch.cat(f_list).to(dev), N, k, torch.cat(s_list),
                     torch.cat(sl_list).to(dev),
                     torch.tensor([it.T for it in items], device=dev),
                     torch.cat(m_list).to(dev) if any_mask else None, local, glob)

    def _chunks(self, items, *extra):
        """Yield index ranges keeping each head call under max_tokens."""
        a, n = 0, len(items)
        while a < n:
            b, tok = a, 0
            while b < n and (tok + items[b].T <= self.max_tokens or b == a):
                tok += items[b].T
                b += 1
            yield a, b
            a = b

    # ------------------------------------------------------------------ head calls
    def _final(self, logits):
        return logits[1] if isinstance(logits, tuple) else logits

    def head_forward(self, batch: Batch, return_attn: bool = False):
        out = self.head(batch.feats, batch.N, batch.k, batch.seq_ids, batch.slot_ids,
                        batch.lengths, local_properties=batch.local_properties,
                        global_properties=batch.global_properties, view_mask=batch.view_mask,
                        return_view_attn=return_attn, return_slot_attn=return_attn)
        if return_attn:
            logits, va, sa = out
            return self._final(logits), va, sa
        return self._final(out)

    @torch.no_grad()
    def logits(self, items: Sequence[HeadInputs], masks=None) -> torch.Tensor:
        """(B, n_out) final logits for a list of (possibly modified) sequences."""
        out = []
        for a, b in self._chunks(items):
            batch = self.make_batch(items[a:b], None if masks is None else masks[a:b])
            out.append(self.head_forward(batch))
        return torch.cat(out)

    def encode(self, batch: Batch) -> torch.Tensor:
        """Per-token embeddings q (N, d_model): output of view pooling + properties +
        seq_in_norm, i.e. the input of the slot mixer."""
        q, _ = self.head.encode_tokens(batch.feats, batch.N, batch.k, batch.seq_ids,
                                       batch.local_properties, batch.global_properties,
                                       batch.view_mask)
        return q

    def mix(self, q: torch.Tensor, batch: Batch) -> torch.Tensor:
        logits, _ = self.head.mix(q, batch.seq_ids, batch.slot_ids, batch.B,
                                  batch.global_properties)
        return self._final(logits)

    @torch.no_grad()
    def pooled(self, items: Sequence[HeadInputs]) -> torch.Tensor:
        """Sequence embedding right before the final Linear (B, out_dim): last step for
        per-step mixers. Captured with a forward hook on classifier.head."""
        store = []
        h = self.head.head.register_forward_hook(lambda m, i, o: store.append(i[0].detach()))
        try:
            outs = []
            for a, b in self._chunks(items):
                store.clear()
                batch = self.make_batch(items[a:b])
                self.head_forward(batch)
                z = store[-1]
                if z.dim() == 3:                       # per-step: take each subject's last step
                    rows, cols, L, order = self.head._layout(
                        batch.seq_ids, batch.slot_ids, batch.B, self.head.mixer.needs_compact)
                    last = torch.zeros(batch.B, dtype=torch.long, device=z.device).scatter_reduce(
                        0, rows, cols, 'amax', include_self=True)
                    z = z[torch.arange(batch.B, device=z.device), last]
                outs.append(z)
            return torch.cat(outs)
        finally:
            h.remove()

    @torch.no_grad()
    def token_embeddings(self, item: HeadInputs) -> torch.Tensor:
        return self.encode(self.make_batch([item]))

    @torch.no_grad()
    def attentions(self, item: HeadInputs):
        """Raw (view_attn, slot_attn) as returned by the model, for one sequence."""
        _, va, sa = self.head_forward(self.make_batch([item]), return_attn=True)
        return va, sa

    def token_layout(self, item: HeadInputs):
        """(rows, cols) of each token in the mixer's (B, L) buffer, in token order."""
        b = self.make_batch([item])
        rows, cols, L, order = self.head._layout(b.seq_ids, b.slot_ids, 1,
                                                 self.head.mixer.needs_compact)
        if order is None:
            return cols
        tok_cols = torch.empty_like(cols)
        tok_cols[order] = cols
        return tok_cols

    @torch.no_grad()
    def prefix_logits(self, items: Sequence[HeadInputs]) -> list[torch.Tensor]:
        """For each sequence, (T, n_out) logits seeing only tokens with slot <= token's slot."""
        out = []
        for a, b in self._chunks(items):
            batch = self.make_batch(items[a:b])
            lg = self.head.forward_prefixes(batch.feats, batch.N, batch.k, batch.seq_ids,
                                            batch.slot_ids, batch.lengths,
                                            local_properties=batch.local_properties,
                                            global_properties=batch.global_properties,
                                            view_mask=batch.view_mask)
            start = 0
            for it in items[a:b]:
                out.append(lg[start:start + it.T])
                start += it.T
        return out

    def feature_grad(self, item: HeadInputs, target, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        """d score / d CNN-features (T, k, F), and the score. Basis for chained pixel
        attributions, TCAV and SAE-latent attribution."""
        f = item.feats.detach().clone().requires_grad_(True)
        with torch.enable_grad(), grad_context(self.device):
            batch = self.make_batch([item], None if mask is None else [mask], feats=[f])
            s = self.score(self.head_forward(batch), target)
            g, = torch.autograd.grad(s.sum(), f)
        return g.detach(), s.detach()


def resolve_target(runner: ModelRunner, item: HeadInputs, mode='pred', logits=None) -> int:
    """'pred' (predicted class), 'label' (true class) or an int."""
    if isinstance(mode, (int, np.integer)):
        return int(mode)
    if mode == 'label':
        if item.label is None:
            raise ValueError(f'{item.seq_id}: no label for target="label"')
        return int(item.label)
    if logits is None:
        logits = runner.logits([item])
    return int(runner.probs(logits).argmax(1)[0])
