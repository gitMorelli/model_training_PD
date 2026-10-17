import torch
import torch.nn as nn

from src.utils.models.aggregators.models import GatedAttentionPool
from src.utils.models.sequence_models.mixers import MIXERS
from .helpers import _property_encoder

#unordered models
class SetClassifierHead(nn.Module):
    """Order-invariant head over a variable number of timesteps per subject.

        logits = rho( pool_{t in available} phi(x_t) )

    Pooling is done by segment-reduction over seq_ids, so every timestep
    handed to the head is used: there is no capacity cap, no padding buffer,
    and no slot_ids. Subjects may have any count >= 1, and counts may differ
    within a batch and between train and inference.

    Pooled features fed to rho:
      * mean of phi(x_t) over the subject's timesteps
      * std across timesteps (optional); zero when the subject has one timestep
      * normalized count (optional)
    """

    def __init__(self, feat_dim, n_classes,
                 d_model=512, view_agg='attention', dropout=0.1,
                 ff_mult=2, use_spread=True, use_count_feature=True,
                 count_norm=8.0, use_attention_pool=False,**kwargs):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model) if feat_dim != d_model else nn.Identity()
        self.view_agg = view_agg
        self.view_pool = GatedAttentionPool(d_model) if view_agg == 'attention' else None

        self.use_attention_pool = use_attention_pool
        if self.use_attention_pool:
            self.time_attention_pool = SegmentedGatedAttentionPool(d_model)
            use_spread = False  # spread is redundant with attention pooling
            use_count_feature = False  # count feature is redundant with attention pooling

        # shared per-timestep encoder (phi)
        self.phi = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.use_spread = use_spread
        self.use_count_feature = use_count_feature
        self.count_norm = float(count_norm)   # scaling only, NOT a capacity limit

        rho_in = d_model \
            + (d_model if use_spread else 0) \
            + (1 if use_count_feature else 0)

        # set-level classifier (rho)
        self.rho = nn.Sequential(
            nn.LayerNorm(rho_in),
            nn.Linear(rho_in, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, n_classes),
        )

    def forward(self, feats, N, k, seq_ids, lengths, return_view_attn=False, local_properties = None, global_properties = None):
        # feats: (N*k, feat_dim) straight from the CNN
        # N = sum(T_i) = total timesteps in the batch
        # seq_ids: (N,) which subject each timestep belongs to
        x = self.proj(feats).view(N, k, -1)                 # (N, k, d_model)
        if self.view_pool is not None:
            q_repr, view_attn = self.view_pool(x)           # (N, d_model)
        else:
            q_repr, view_attn = x.mean(1), None

        B   = lengths.size(0)
        D   = q_repr.size(-1)
        dev = q_repr.device
        seq = seq_ids.long()

        assert int(seq.max()) < B, \
            f"seq_id {int(seq.max())} out of range for batch of {B} subjects"

        h = self.phi(q_repr)                                # (N, D)

        # segment-reduce over subjects: every timestep in the batch contributes
        n = torch.zeros(B, 1, device=dev, dtype=h.dtype).index_add_(
            0, seq, torch.ones(seq.numel(), 1, device=dev, dtype=h.dtype))
        assert (n.squeeze(1) > 0).all(), "a subject in the batch has no timesteps"

        if self.use_attention_pool:
            # Use attention pooling over the time dimension for each subject
            mean, _ = self.time_attention_pool(h, seq, B)  # (B, D)
        else:
            s = torch.zeros(B, D, device=dev, dtype=h.dtype).index_add_(0, seq, h)
            mean = s / n

        parts = [mean]
        if self.use_spread:
            # E[h^2] - E[h]^2, in one pass; clamp guards fp cancellation
            sq = torch.zeros(B, D, device=dev, dtype=h.dtype).index_add_(0, seq, h * h)
            var = (sq / n - mean ** 2).clamp(min=0)
            parts.append(torch.sqrt(var + 1e-8))            # ~zero when n == 1
        if self.use_count_feature:
            parts.append(((n - 1.0) / (self.count_norm - 1.0)).clamp(0, 1))

        logits = self.rho(torch.cat(parts, dim=-1))
        return (logits, view_attn) if return_view_attn else logits
class SetQuestionnaireModel(nn.Module):
    def __init__(self, vision_model, feat_dim, n_classes, **head_kwargs):
        super().__init__()
        self.vision_model = vision_model                    # -> cnn.*
        self.classifier = SetClassifierHead(                # -> classifier.*
            feat_dim, n_classes, **head_kwargs)

    def forward(self, frames, seq_ids, slot_ids, lengths, local_properties=None, global_properties=None, return_view_attn=False):
        N, k  = frames.shape[:2]
        feats = self.vision_model(frames.flatten(0, 1))     # (N*k, feat_dim)
        return self.classifier(feats, N, k, seq_ids, lengths, return_view_attn, 
            local_properties=local_properties,global_properties=global_properties)
