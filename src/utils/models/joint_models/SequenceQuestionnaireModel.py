import torch
import torch.nn as nn

from src.utils.models.aggregators.models import GatedAttentionPool
from src.utils.models.sequence_models.mixers import MIXERS
from .helpers import _property_encoder

class SequenceClassifierHead(nn.Module):
    """Everything after the CNN: view aggregation + slot transformer + classification.
    All params land under `classifier.*` in the parent's named_parameters()."""
    def __init__(self, feat_dim, n_slots, n_classes,
                 d_model=512, n_heads=8, n_layers=4, ff_mult=4,
                 view_agg='attention', dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model) if feat_dim != d_model else nn.Identity()

        self.view_agg  = view_agg
        self.view_pool = GatedAttentionPool(d_model) if view_agg == 'attention' else None

        self.n_slots  = n_slots
        self.slot_pos = nn.Embedding(n_slots, d_model)
        self.cls      = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        #The * 0.02 is the standard transformer init scale — it's initializer_range=0.02 from BERT and GPT-2, inherited by most implementations since

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff_mult * d_model,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, feats, N, k, seq_ids, slot_ids, lengths, return_view_attn=False):
        # N total number of frames (can come from different subjects), k modality dimensions
        # feats: (N*k, feat_dim) straight from the CNN
        x = self.proj(feats).view(N, k, -1)                 # (N, k, d_model)

        if self.view_pool is not None:
            q_repr, view_attn = self.view_pool(x)           # (N, d_model)
        else:
            q_repr, view_attn = x.mean(1), None #use a simple mean instead of global average pooling

        B   = lengths.size(0) #number of subjects in the batch
        D   = q_repr.size(-1) #dimensionality of the pooled features
        dev = q_repr.device

        buffer = q_repr.new_zeros(B, self.n_slots, D) #new_zeros creates a zero tensor that keeps the type and device of q_repr
        buffer[seq_ids, slot_ids] = q_repr
        pad_mask = torch.ones(B, self.n_slots, dtype=torch.bool, device=dev)
        pad_mask[seq_ids, slot_ids] = False

        pos    = self.slot_pos(torch.arange(self.n_slots, device=dev))
        buffer = buffer + pos.unsqueeze(0)

        cls  = self.cls.expand(B, -1, -1) # (1,1,D) -> (B, 1, D) — expand is like repeat but without copying memory
        seq  = torch.cat([cls, buffer], dim=1)              # (B, 1+Q, D)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev),
                          pad_mask], dim=1)

        out    = self.transformer(seq, src_key_padding_mask=mask)
        logits = self.head(self.norm(out[:, 0]))
        return (logits, view_attn) if return_view_attn else logits
class SequenceQuestionnaireModel(nn.Module):
    def __init__(self, vision_model, feat_dim, n_slots, n_classes, **head_kwargs):
        super().__init__()
        self.vision_model = vision_model                               # -> cnn.*
        self.classifier = SequenceClassifierHead(           # -> classifier.*
            feat_dim, n_slots, n_classes, **head_kwargs)

    def forward(self, frames, seq_ids, slot_ids, lengths, return_view_attn=False):
        N, k  = frames.shape[:2]
        feats = self.vision_model(frames.flatten(0, 1))             # (N*k, feat_dim)  <- only heavy step
        return self.classifier(feats, N, k, seq_ids, slot_ids, lengths, return_view_attn)
