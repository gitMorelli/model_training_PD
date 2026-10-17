import torch
import torch.nn as nn
from src.utils.models.aggregators.models import GatedAttentionPool
from src.utils.models.sequence_models.mixers import MIXERS
from .helpers import _property_encoder

#Time-sequence models
class SequenceFlexibleClassifierHead(nn.Module):
    """Everything after the CNN: view aggregation + slot mixing + classification.
    All params land under `classifier.*` in the parent's named_parameters().

    `d_model` is now the single width: the CNN features are projected straight to it
    and pooling happens there, so there is no separate sequence-side dimension.


    h = SequenceClassifierHead(feat_dim, n_slots, n_classes, d_model=256, seq_model='mean', use_slot_pos=False) #masked mean
    h = SequenceClassifierHead(feat_dim, n_slots, n_classes, d_model=256, seq_model='pool', use_slot_pos=False) #attention pooling
    h = SequenceClassifierHead(feat_dim, n_slots, n_classes, d_model=256, seq_model='pool') #attention pooling with slot pos
    h = SequenceClassifierHead(feat_dim, n_slots, n_classes, d_model=256, seq_model='gru', bidirectional=True) #bidirectional GRU
    h = SequenceClassifierHead(feat_dim, n_slots, n_classes, d_model=256, seq_model='conv', n_blocks=3, kernel=5) #temporal convolution
    h = SequenceClassifierHead(feat_dim, n_slots, n_classes, d_model=256, seq_model='transformer', n_layers=4)
    """

    def __init__(self, feat_dim, n_slots, n_classes,
                 d_model=256, seq_model='gru', view_agg='attention',
                 use_slot_pos=True, dropout=0.1,
                 n_local=0, n_global=0, global_fusion='both',
                 **mixer_kwargs):
        super().__init__()
        if seq_model not in MIXERS:
            raise ValueError(f'seq_model must be one of {sorted(MIXERS)}')
        if global_fusion not in ('early', 'late', 'both'):
            raise ValueError("global_fusion must be 'early', 'late' or 'both'")

        self.proj = nn.Linear(feat_dim, d_model) if feat_dim != d_model else nn.Identity()
        self.view_agg = view_agg
        self.view_pool = GatedAttentionPool(d_model) if view_agg == 'attention' else None

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

    # ---- slot layout -------------------------------------------------------

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

    # ---- forward -----------------------------------------------------------

    def forward(self, feats, N, k, seq_ids, slot_ids, lengths,
                local_properties=None, global_properties=None,
                return_view_attn=False, return_slot_attn=False):
        if self.local_enc is not None and local_properties is None:
            raise ValueError('head was built with n_local > 0 but local_properties is None')
        if self.n_global and global_properties is None:
            raise ValueError('head was built with n_global > 0 but global_properties is None')

        x = self.proj(feats).view(N, k, -1)                        # (N, k, d_model)
        if self.view_pool is not None:
            q, view_attn = self.view_pool(x)
        else:
            q, view_attn = x.mean(1), None                         # (N, d_model)

        # ---- per-token conditioning (still in input/frame order) ----
        if self.local_enc is not None:
            q = q + self.local_enc(local_properties.to(q.dtype))        # (N, d_model)
        if self.global_early is not None:
            g = self.global_early(global_properties.to(q.dtype))        # (B, d_model)
            q = q + g[seq_ids]                                     # broadcast to tokens
        q = self.seq_in_norm(q)

        B, dev = lengths.size(0), q.device
        compact = self.mixer.needs_compact
        rows, cols, L, order = self._layout(seq_ids, slot_ids, B, compact)

        q_pos = q + self.slot_pos(slot_ids) if self.use_slot_pos else q
        buf = self._scatter(q_pos, rows, cols, order, B, L)
        values = self._scatter(q, rows, cols, order, B, L) if self.separate_values else None

        pad_mask = torch.ones(B, L, dtype=torch.bool, device=dev)
        pad_mask[rows, cols] = False
        lens = torch.bincount(seq_ids, minlength=B)

        pooled, slot_attn = self.mixer(buf, pad_mask, lens, values=values)
        z = self.norm(pooled)                                      # (B, D) or (B, L, D) if per_step
        if self.global_late is not None:
            g = self.global_late(global_properties.to(z.dtype))         # (B, D)
            z = z + (g.unsqueeze(1) if z.dim() == 3 else g)
        logits = self.head(self.drop(z))

        if getattr(self.mixer, 'per_step', False):
            n_tok = seq_ids.numel()
            idx = order if order is not None else torch.arange(n_tok, device=dev)
            tok_logits = logits.new_zeros(n_tok, logits.size(-1))
            tok_logits[idx] = logits[rows, cols]
            last = logits[torch.arange(B, device=dev), (lens - 1).clamp(min=0)]
            logits = (tok_logits, last)

        if not (return_view_attn or return_slot_attn):
            return logits
        out = [logits]
        if return_view_attn:
            out.append(view_attn)
        if return_slot_attn:
            out.append(slot_attn)
        return tuple(out)
class FlexibleSequenceQuestionnaireModel(nn.Module):
    def __init__(self, vision_model, feat_dim, n_slots, n_classes, **head_kwargs):
        super().__init__()
        self.vision_model = vision_model
        self.classifier = SequenceFlexibleClassifierHead(
            feat_dim, n_slots, n_classes, **head_kwargs)

    def forward(self, frames, seq_ids, slot_ids, lengths,
                local_properties=None, global_properties=None,
                return_view_attn=False, return_slot_attn=False):
        N, k = frames.shape[:2]
        feats = self.vision_model(frames.flatten(0, 1))            # (N*k, feat_dim)
        return self.classifier(feats, N, k, seq_ids, slot_ids, lengths,
                               local_properties=local_properties, global_properties=global_properties,
                               return_view_attn=return_view_attn,
                               return_slot_attn=return_slot_attn)
