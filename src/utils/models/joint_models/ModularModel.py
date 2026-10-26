import torch
import torch.nn as nn

from src.utils.models.aggregators.view_aggregation import ModalityViewAggregator, build_view_pool
from src.utils.models.sequence_models.mixers import MIXERS
from .helpers import _property_encoder
# MIXERS and _property_encoder come from your existing module, e.g.:
# from .mixers import MIXERS, _property_encoder


class SequenceModularHead(nn.Module):
    """Everything after the CNN: view aggregation + slot mixing + classification.
    All params land under `classifier.*` in the parent's named_parameters().

    `d_model` is the single width: CNN features are projected straight to it and all
    pooling happens there.

    ---- view aggregation (k crops per token -> 1 vector) ----
    Without `view_groups` the k crops are treated as an unordered set (old behaviour):
        view_agg in {'mean', 'max', 'lse', 'attention', 'pma', 'transformer', 'concat'}
        options go in view_kwargs (e.g. dict(heads=4) for 'attention').

    With `view_groups` (crops of each modality concatenated along k, in that order) the
    aggregator is modality-aware: per-modality LayerNorm, learned crop-position and
    modality embeddings, masking of missing modalities, and modality/crop dropout.
        view_agg='hierarchical'  -> intra-modality pool, then fusion across modalities
        any other view_agg       -> that pool applied flat over all k crops
    All ModalityViewAggregator options go in view_kwargs.

    Examples
    groups = {'modA': n1, 'modB': n2, 'modC': n3}

    # old behaviour, unchanged parameters (old checkpoints load)
    h = Head(feat_dim, n_slots, n_classes, view_agg='attention')

    # hierarchical: gated attention within modality, learned 'missing' tokens + concat across
    h = Head(feat_dim, n_slots, n_classes, view_agg='hierarchical', view_groups=groups,
             view_kwargs=dict(intra='attention', inter='concat', modality_dropout=0.3))

    # different pool per modality: the 2 modB crops fused position-specifically
    h = Head(..., view_agg='hierarchical', view_groups=groups,
             view_kwargs=dict(intra={'modA': 'attention', 'modB': 'concat', 'modC': 'attention'},
                              intra_kwargs={'modA': dict(heads=4), 'modC': dict(heads=4)},
                              inter='concat', modality_dropout=0.3))

    # hierarchical with attention across modalities (interpretable modality weights)
    h = Head(..., view_agg='hierarchical', view_groups=groups,
             view_kwargs=dict(intra='attention', intra_kwargs=dict(heads=4),
                              inter='attention', modality_dropout=0.3, view_dropout=0.1))

    # flat joint model: 2 self-attention layers over all crops, then PMA pooling
    h = Head(..., view_agg='transformer', view_groups=groups,
             view_kwargs=dict(pool_kwargs=dict(heads=4, n_sab=2), modality_dropout=0.3))

    ---- slot mixing (unchanged) ----
    seq_model in MIXERS: 'mean', 'pool', 'gru', 'conv', 'transformer', ...
    """

    def __init__(self, feat_dim, n_slots, n_classes,
                 d_model=256, seq_model='gru', view_agg='attention',
                 use_slot_pos=True, dropout=0.1,
                 n_local=0, n_global=0, global_fusion='both',
                 view_groups=None, modalities=None, view_kwargs=None,
                 **mixer_kwargs):
        super().__init__()
        if seq_model not in MIXERS:
            raise ValueError(f'seq_model must be one of {sorted(MIXERS)}')
        if global_fusion not in ('early', 'late', 'both'):
            raise ValueError("global_fusion must be 'early', 'late' or 'both'")

        self.proj = nn.Linear(feat_dim, d_model) if feat_dim != d_model else nn.Identity()
        self.modalities = modalities

        # ---- view aggregation ----
        self.view_agg = view_agg
        view_kwargs = dict(view_kwargs or {})
        if view_groups is not None:
            if view_agg == 'hierarchical':
                view_kwargs['mode'] = 'hierarchical'
            else:
                view_kwargs.update(mode='flat', pool=view_agg)
            self.view_pool = ModalityViewAggregator(d_model, view_groups, **view_kwargs)
        else:
            if view_agg == 'hierarchical':
                raise ValueError("view_agg='hierarchical' needs view_groups")
            self.view_pool = build_view_pool(view_agg, d_model, **view_kwargs)
        self.modality_aware = view_groups is not None

        # ---- property encoders ----
        self.n_local, self.n_global, self.global_fusion = n_local, n_global, global_fusion
        self.local_enc = _property_encoder(n_local, d_model, dropout) if n_local else None
        self.global_early = (_property_encoder(n_global, d_model, dropout)
                             if n_global and global_fusion in ('early', 'both') else None)

        self.seq_in_norm = nn.LayerNorm(d_model)          # scale-match before + slot_pos
        self.n_slots = n_slots
        self.use_slot_pos = use_slot_pos
        if use_slot_pos:
            self.slot_pos = nn.Embedding(n_slots, d_model)
            nn.init.normal_(self.slot_pos.weight, std=0.02)

        cfg = dict(dropout=dropout, **mixer_kwargs)
        self.seq_model = seq_model
        self.mixer = MIXERS[seq_model](d_model, cfg)
        self.separate_values = use_slot_pos and self.mixer.wants_raw_values

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.mixer.out_dim)
        self.global_late = (_property_encoder(n_global, self.mixer.out_dim, dropout)
                            if n_global and global_fusion in ('late', 'both') else None)
        self.head = nn.Linear(self.mixer.out_dim, n_classes)

    # ---- view mask -----------------------------------------------------------

    def build_view_mask(self, view_mask, modalities, N, k, device):
        """Combine an explicit (N, k) mask (e.g. imputed crops = False) with a modality
        subset (names or indices). Returns None when every crop is present."""
        if modalities is not None:
            if not self.modality_aware:
                raise ValueError('`modalities` needs a head built with view_groups')
            keep = self.view_pool.modality_view_mask(modalities, device).expand(N, k)
            view_mask = keep if view_mask is None else view_mask.bool() & keep
        return None if view_mask is None else view_mask.bool()

    # ---- slot layout ---------------------------------------------------------

    def _layout(self, seq_ids, slot_ids, B, compact):
        """Row/col indices for scattering tokens into a (B, L, D) buffer."""
        if not compact:
            return seq_ids, slot_ids, self.n_slots, None
        order = torch.argsort(seq_ids * self.n_slots + slot_ids)   # subject-major, in time
        rows = seq_ids[order]
        counts = torch.bincount(rows, minlength=B)
        offs = torch.cumsum(counts, 0) - counts
        cols = torch.arange(rows.numel(), device=rows.device) - offs[rows]
        L = int(counts.max()) if rows.numel() else 1
        return rows, cols, max(L, 1), order

    @staticmethod
    def _scatter(q, rows, cols, order, B, L):
        buf = q.new_zeros(B, L, q.size(-1))
        buf[rows, cols] = q if order is None else q[order]
        return buf

    # ---- forward -------------------------------------------------------------

    def encode_tokens(self, feats, N, k, seq_ids, local_properties=None,
                      global_properties=None, view_mask=None):
        """Everything that is per-token (independent of which other tokens are visible)."""
        x = self.proj(feats).view(N, k, -1)
        mask = self.build_view_mask(view_mask, self.modalities, N, k, x.device)
        if self.modality_aware:
            q, view_attn = self.view_pool(x, mask, group_ids=seq_ids)
        else:
            q, view_attn = self.view_pool(x, mask)
        if self.local_enc is not None:
            q = q + self.local_enc(local_properties.to(q.dtype))
        if self.global_early is not None:
            q = q + self.global_early(global_properties.to(q.dtype))[seq_ids]
        return self.seq_in_norm(q), view_attn

    def mix(self, q, seq_ids, slot_ids, B, global_properties=None):
        """Slot mixing + classification. global_properties: (B, n_global)."""
        dev = q.device
        rows, cols, L, order = self._layout(seq_ids, slot_ids, B, self.mixer.needs_compact)
        q_pos = q + self.slot_pos(slot_ids) if self.use_slot_pos else q
        buf = self._scatter(q_pos, rows, cols, order, B, L)
        values = self._scatter(q, rows, cols, order, B, L) if self.separate_values else None
        pad_mask = torch.ones(B, L, dtype=torch.bool, device=dev)
        pad_mask[rows, cols] = False
        lens = torch.bincount(seq_ids, minlength=B)

        pooled, slot_attn = self.mixer(buf, pad_mask, lens, values=values)
        z = self.norm(pooled)
        if self.global_late is not None:
            g = self.global_late(global_properties.to(z.dtype))
            z = z + (g.unsqueeze(1) if z.dim() == 3 else g)
        z = self.drop(z)
        with torch.autocast(device_type=z.device.type, enabled=False):
            logits = self.head(z.float())

        if getattr(self.mixer, 'per_step', False):
            n_tok = seq_ids.numel()
            idx = order if order is not None else torch.arange(n_tok, device=dev)
            tok_logits = logits.new_zeros(n_tok, logits.size(-1))
            tok_logits[idx] = logits[rows, cols]
            last_col = torch.zeros(B, dtype=torch.long, device=dev).scatter_reduce(
                0, rows, cols, 'amax', include_self=True)
            last = logits[torch.arange(B, device=dev), last_col]
            logits = (tok_logits, last)
        return logits, slot_attn

    def forward(self, feats, N, k, seq_ids, slot_ids, lengths,
                local_properties=None, global_properties=None,
                view_mask=None, return_view_attn=False, return_slot_attn=False):
        if self.local_enc is not None and local_properties is None:
            raise ValueError('head was built with n_local > 0 but local_properties is None')
        if self.n_global and global_properties is None:
            raise ValueError('head was built with n_global > 0 but global_properties is None')
        q, view_attn = self.encode_tokens(feats, N, k, seq_ids, local_properties,
                                          global_properties, view_mask)
        logits, slot_attn = self.mix(q, seq_ids, slot_ids, lengths.size(0), global_properties)
        if not (return_view_attn or return_slot_attn):
            return logits
        out = [logits]
        if return_view_attn:
            out.append(view_attn)
        if return_slot_attn:
            out.append(slot_attn)
        return tuple(out)

    @torch.no_grad()
    def forward_prefixes(self, feats, N, k, seq_ids, slot_ids, lengths,
                         local_properties=None, global_properties=None,
                         view_mask=None, max_cost=200_000):
        """logits[i] = prediction when the model sees only the tokens of subject seq_ids[i]
        with time <= slot_ids[i] (its causal prefix, current step included).
        Returns (N, n_classes) in the same order as seq_ids. Exact for any mixer."""
        q, _ = self.encode_tokens(feats, N, k, seq_ids, local_properties,
                                  global_properties, view_mask)
        dev, B = q.device, lengths.size(0)

        # fast path: the mixer gives per-step outputs (hence is causal) -> one pass is enough
        if getattr(self.mixer, 'per_step', False):
            return self.mix(q, seq_ids, slot_ids, B, global_properties)[0][0]

        order = torch.argsort(seq_ids * self.n_slots + slot_ids, stable=True)
        subj = seq_ids[order]
        counts = torch.bincount(subj, minlength=B)
        offs = torch.cumsum(counts, 0) - counts
        rank = torch.arange(N, device=dev) - offs[subj]           # position within subject
        sizes = rank + 1                                           # prefix length per virtual subject

        # chunk virtual subjects by memory cost (compact: #tokens, else padded n_slots)
        cost = sizes if self.mixer.needs_compact else torch.full_like(sizes, self.n_slots)
        ends = torch.cumsum(cost, 0)

        res, a = [], 0
        while a < N:
            base = int(ends[a - 1]) if a else 0
            b = max(int(torch.searchsorted(ends, base + max_cost, right=True)), a + 1)
            sz = sizes[a:b]
            vid = torch.repeat_interleave(torch.arange(b - a, device=dev), sz)  # virtual subject id
            start = torch.cumsum(sz, 0) - sz
            within = torch.arange(vid.numel(), device=dev) - start[vid]       # 0..rank
            member = order[offs[subj[a:b]][vid] + within]                     # original token idx
            g = global_properties[subj[a:b]] if global_properties is not None else None
            logits, _ = self.mix(q[member], vid, slot_ids[member], b - a, g)
            if isinstance(logits, tuple):         # per-step mixer: the prefix's last step
                logits = logits[1]
            res.append(logits)
            a = b

        res = torch.cat(res)
        out = torch.empty_like(res)
        out[order] = res                          # back to input token order
        return out


class ModularModel(nn.Module):
    """skip_masked_views: don't run the CNN on crops that are masked out (imputed blanks,
    or modalities excluded at inference); their features are set to zero instead."""

    def __init__(self, vision_model, feat_dim, n_slots, n_classes,
                 skip_masked_views=True, **head_kwargs):
        super().__init__()
        self.vision_model = vision_model
        self.feat_dim = feat_dim
        self.skip_masked_views = skip_masked_views
        self.classifier = SequenceModularHead(
            feat_dim, n_slots, n_classes, **head_kwargs)

    def _features(self, frames, mask):
        flat = frames.flatten(0, 1)                                # (N*k, C, H, W)
        if mask is None or not self.skip_masked_views or bool(mask.all()):
            return self.vision_model(flat)                         # (N*k, feat_dim)
        idx = mask.flatten().nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return frames.new_zeros(flat.size(0), self.feat_dim)
        out = self.vision_model(flat[idx])
        feats = out.new_zeros(flat.size(0), out.size(-1))
        feats[idx] = out
        return feats

    def forward(self, frames, seq_ids, slot_ids, lengths,
                local_properties=None, global_properties=None,
                view_mask=None, return_view_attn=False, return_slot_attn=False):
        N, k = frames.shape[:2]
        mask = self.classifier.build_view_mask(view_mask, self.classifier.modalities, N, k, frames.device)
        feats = self._features(frames, mask)
        return self.classifier(feats, N, k, seq_ids, slot_ids, lengths,
                               local_properties=local_properties,
                               global_properties=global_properties,
                               view_mask=mask,
                               return_view_attn=return_view_attn,
                               return_slot_attn=return_slot_attn)

    @torch.no_grad()
    def forward_prefixes(self, frames, seq_ids, slot_ids, lengths,
                         local_properties=None, global_properties=None,
                         view_mask=None, modalities=None, max_cost=200_000):
        if self.training:
            raise RuntimeError('call model.eval() first')
        N, k = frames.shape[:2]
        mods = modalities if modalities is not None else self.classifier.modalities
        mask = self.classifier.build_view_mask(view_mask, mods, N, k, frames.device)
        feats = self._features(frames, mask)                       # CNN runs ONCE
        return self.classifier.forward_prefixes(
            feats, N, k, seq_ids, slot_ids, lengths,
            local_properties=local_properties, global_properties=global_properties,
            view_mask=mask, max_cost=max_cost)
    
    def get_modalities_names(self):
        """Return the names of the modalities used by the model, if available."""
        return self.classifier.view_pool.names

    def subset_name(self, subset):
        """'all' for None, else modality names joined in view_groups order, e.g. 'A+C'."""
        if subset is None:
            return 'all'
        names = self.classifier.view_pool.names
        idx = sorted({names.index(m) if isinstance(m, str) else int(m) for m in subset})
        return '+'.join(names[i] for i in idx)

    def forward_subsets(self, frames, seq_ids, slot_ids, lengths, subsets,
                        local_properties=None, global_properties=None, view_mask=None,
                        return_view_attn=False, return_slot_attn=False,
                        return_token_present=False):
        """Predictions for several modality subsets from ONE pass of the vision model.
 
        Call in eval mode (model.eval(), under torch.no_grad()): the CNN features of a crop
        don't depend on the subset, so they are computed once on the union of crops any
        subset needs, and only the (cheap) head is run once per subset. Each result is
        identical to forward(..., modalities=subset).
 
        subsets: list of subsets, each a list of modality names/indices, or None for all,
                 e.g. [None, ['A'], ['B'], ['C'], ['A', 'C']].
        Returns
          outputs: {name: what forward() returns for that subset}   (name from subset_name)
          present: {name: (B,) bool} True if the subject has at least one real crop in the
                   subset. Subjects with False are predicted from properties alone; you may
                   want to report metrics with and without them.
          tok_present (only if return_token_present=True):
                   {name: (N,) bool} True if that token (slot) has at least one real crop in
                   the subset; same order as seq_ids / per-step token logits.
        """
        if self.training:
            raise RuntimeError('forward_subsets is for evaluation: call model.eval() first')
        N, k = frames.shape[:2]
        dev, B = frames.device, lengths.size(0)
        build = self.classifier.build_view_mask
        masks = {self.subset_name(s): build(view_mask, s, N, k, dev) for s in subsets}
 
        if any(m is None for m in masks.values()):
            union = None
        else:
            union = torch.stack(list(masks.values())).any(0)
        feats = self._features(frames, union)
 
        outputs, present, tok_present = {}, {}, {}
        for name, m in masks.items():
            outputs[name] = self.classifier(
                feats, N, k, seq_ids, slot_ids, lengths,
                local_properties=local_properties, global_properties=global_properties,
                view_mask=m, return_view_attn=return_view_attn,
                return_slot_attn=return_slot_attn)
            tok = (torch.ones(N, dtype=torch.bool, device=dev) if m is None else m.any(1))
            present[name] = torch.zeros(B, dtype=torch.bool, device=dev).index_put_(
                (seq_ids,), tok, accumulate=True)
            tok_present[name] = tok
        if return_token_present:
            return outputs, present, tok_present
        return outputs, present

    @torch.no_grad()
    def forward_prefixes_subsets(self, frames, seq_ids, slot_ids, lengths, subsets,
                                 local_properties=None, global_properties=None,
                                 view_mask=None, max_cost=200_000):
        """Causal prefix logits (see SequenceModularHead.forward_prefixes) for several
        modality subsets from ONE pass of the vision model.
        Returns
          outputs:        {name: (N, n_classes) prefix logits, same order as seq_ids}
          present:        {name: (B,) bool} subject has >= 1 real crop in the subset
          prefix_present: {name: (N,) bool} the token's prefix has >= 1 real crop
        """
        if self.training:
            raise RuntimeError('call model.eval() first')
        N, k = frames.shape[:2]
        dev, B = frames.device, lengths.size(0)
        build = self.classifier.build_view_mask
        masks = {self.subset_name(s): build(view_mask, s, N, k, dev) for s in subsets}
        union = (None if any(m is None for m in masks.values())
                 else torch.stack(list(masks.values())).any(0))
        feats = self._features(frames, union)                     # CNN runs once

        # subject-major, time-sorted layout for the cumulative 'present' flags
        order = torch.argsort(seq_ids * self.classifier.n_slots + slot_ids, stable=True)
        s = seq_ids[order]
        counts = torch.bincount(s, minlength=B)
        offs = torch.cumsum(counts, 0) - counts

        outputs, present, prefix_present = {}, {}, {}
        for name, m in masks.items():
            outputs[name] = self.classifier.forward_prefixes(
                feats, N, k, seq_ids, slot_ids, lengths,
                local_properties=local_properties, global_properties=global_properties,
                view_mask=m, max_cost=max_cost)
            tok = torch.ones(N, dtype=torch.bool, device=dev) if m is None else m.any(1)
            present[name] = torch.zeros(B, dtype=torch.bool, device=dev).index_put_(
                (seq_ids,), tok, accumulate=True)
            f = tok[order].long()
            c = torch.cumsum(f, 0)
            pp = torch.empty_like(tok)
            pp[order] = (c - (c - f)[offs[s]]) > 0                # running count within subject
            prefix_present[name] = pp
        return outputs, present, prefix_present

    
'''
present[name] is a (B,) bool that says whether a subject has at least one real crop in that subset. A subject with none gets a prediction from properties alone. 
Report metrics on all subjects and on just the covered ones, otherwise e.g. “B only” looks worse than it is.
'''

'''
view_mask -> which crops are real
Shape and type: (N, k) bool, the same first two dimensions as frames (frames is (N, k, C, H, W)).
N is the number of tokens in the batch, i.e. all slots of all subjects flattened, the same rows that seq_ids / slot_ids index.
k = n1 + 2 + n3, in the same order as view_groups: A crops first, then B, then C.
Meaning: True means a real crop. False means a crop you filled with a blank image.
What happens to masked crops: they aren’t sent through the CNN and are zeroed, and every pooler ignores them. If a whole modality is masked for a token, that modality is treated as missing.
Why you need it: without it, the model sees your blank images as real images of that modality. It then learns that “blank” is a meaningful pattern, which is different from “absent”.
None means every crop is real (the old behaviour).
'''

'''
modalities: which modalities to use on this call
Format: a list of modality names or indices, e.g. ['A', 'C'] or [0, 2]. It applies to the whole batch.
Effect: it builds a mask that hides every crop of the other modalities, then ANDs it with view_mask. So a crop is used only if it’s real and its modality is selected.
None means all modalities.
Use: leave it None in training (modality dropout handles the robustness there) and set it during validation.
'''