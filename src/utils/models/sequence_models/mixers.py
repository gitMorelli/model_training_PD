import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanMixer(nn.Module):
    """Rung 1: masked mean over valid slots. Zero parameters."""
    needs_compact, wants_raw_values = False, True

    def __init__(self, d):
        super().__init__()
        self.out_dim = d

    def forward(self, x, pad_mask, lengths, values=None):
        keep = (~pad_mask).unsqueeze(-1).to(x.dtype)
        val = values if values is not None else x
        return (val * keep).sum(1) / keep.sum(1).clamp(min=1), None
class AttnPoolMixer(nn.Module):
    """Rungs 2-3: learned masked weighted average. No temporal interaction."""
    needs_compact, wants_raw_values = False, True

    def __init__(self, d, hidden=None, gated=True):
        super().__init__()
        hidden = hidden or d // 2
        self.norm = nn.LayerNorm(d)
        self.gated = gated
        if gated:
            self.v = nn.Linear(d, hidden)
            self.u = nn.Linear(d, hidden)
            self.w = nn.Linear(hidden, 1)
        else:
            self.w = nn.Linear(d, 1)
        self.out_dim = d

    def forward(self, x, pad_mask, lengths, values=None):
        h = self.norm(x)
        s = (self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h)))
             if self.gated else self.w(h)).squeeze(-1)
        a = s.masked_fill(pad_mask, float('-inf')).softmax(-1).nan_to_num(0.0)
        val = values if values is not None else x
        return (val * a.unsqueeze(-1)).sum(1), a
class ConvMixer(nn.Module):
    """Rung 4a: dilated depthwise-separable residual blocks + attention readout."""
    needs_compact, wants_raw_values = False, False

    def __init__(self, d, n_blocks=3, k=5, mult=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(
                norm=nn.LayerNorm(d),
                dw=nn.Conv1d(d, d, k, padding=(2 ** i) * (k - 1) // 2,
                             dilation=2 ** i, groups=d),
                pw1=nn.Linear(d, mult * d),
                pw2=nn.Linear(mult * d, d),
                drop=nn.Dropout(dropout),
            )) for i in range(n_blocks)])
        self.pool_score = nn.Linear(d, 1)
        self.out_dim = d

    def forward(self, x, pad_mask, lengths, values=None):
        keep = (~pad_mask).unsqueeze(-1).to(x.dtype)
        for b in self.blocks:
            h = b['norm'](x) * keep                        # never leak into padding
            h = b['dw'](h.transpose(1, 2)).transpose(1, 2)
            h = b['pw2'](F.gelu(b['pw1'](h)))
            x = x + b['drop'](h) * keep
        s = self.pool_score(x).squeeze(-1).masked_fill(pad_mask, float('-inf'))
        a = s.softmax(-1).nan_to_num(0.0)
        return (x * a.unsqueeze(-1)).sum(1), a
class GRUMixer(nn.Module):
    """Rung 4b: packed bi-GRU, final hidden state as the summary. Needs compact layout."""
    needs_compact, wants_raw_values = True, False

    def __init__(self, d, hidden=None, n_layers=1, dropout=0.1, bidirectional=True):
        super().__init__()
        hidden = hidden or d // 2
        self.rnn = nn.GRU(d, hidden, num_layers=n_layers, batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.dirs = 2 if bidirectional else 1
        self.hidden = hidden
        self.out_dim = hidden * self.dirs

    def forward(self, x, pad_mask, lengths, values=None):
        B = x.size(0)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h = h.view(-1, self.dirs, B, self.hidden)[-1]      # last layer: (dirs, B, H)
        summary = torch.cat(list(h), dim=-1)               # (B, H*dirs)
        return summary * (lengths > 0).unsqueeze(-1).to(summary.dtype), None

class CausalGRUMixer(nn.Module):
    needs_compact, wants_raw_values = True, False

    def __init__(self, d, hidden=None, n_layers=1, dropout=0.1):
        super().__init__()
        hidden = hidden or d // 2
        self.rnn = nn.GRU(d, hidden, num_layers=n_layers, batch_first=True,
                          bidirectional=False,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.dirs = 1
        self.hidden = hidden
        self.out_dim = hidden * self.dirs
        self.per_step = True  # flag to indicate that this mixer produces per-step outputs

    def forward(self, x, pad_mask, lengths, values=None):
        B, L = x.shape[:2]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)

        seq, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=L)          # (B, L, H)
        return seq * (~pad_mask).unsqueeze(-1).to(seq.dtype), None

class TransformerMixer(nn.Module):
    """Rung 5: your original encoder + CLS readout."""
    needs_compact, wants_raw_values = False, False

    def __init__(self, d, n_heads=8, n_layers=4, ff_mult=4, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d, n_heads, dim_feedforward=ff_mult * d, dropout=dropout,
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.out_dim = d

    def forward(self, x, pad_mask, lengths, values=None):
        B, dev = x.size(0), x.device
        seq = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), pad_mask], dim=1)
        return self.enc(seq, src_key_padding_mask=mask)[:, 0], None

class CausalTransformerMixer(nn.Module):
    """Rung 5b: causal (decoder-only style) encoder. Output at step t sees only steps <= t.

    Per-step outputs, same contract as CausalGRUMixer: returns (B, L, d), padded slots zeroed.
    Works with any padding layout (right-padded or holes), so no compact layout needed.
    """
    needs_compact, wants_raw_values = False, False

    def __init__(self, d, n_heads=8, n_layers=4, ff_mult=4, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d, n_heads, dim_feedforward=ff_mult * d, dropout=dropout,
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d),
                                         enable_nested_tensor=False)
        self.n_heads = n_heads
        self.out_dim = d
        self.per_step = True

    @staticmethod
    def _attn_mask(pad_mask, n_heads):
        """Bool mask (B*heads, L, L), True = blocked. Blocks future keys and padded keys.

        The diagonal is always left open so no query row is fully masked (a padded query
        whose earlier keys are all padding would otherwise give NaN). That only affects
        padded queries, whose outputs are zeroed anyway; valid queries never see pad keys.
        """
        B, L = pad_mask.shape
        dev = pad_mask.device
        future = torch.ones(L, L, dtype=torch.bool, device=dev).triu(1)      # (L, L)
        m = future.unsqueeze(0) | pad_mask.unsqueeze(1)                       # (B, L, L)
        m = m & ~torch.eye(L, dtype=torch.bool, device=dev)
        return m.repeat_interleave(n_heads, dim=0)                            # (B*h, L, L)

    def forward(self, x, pad_mask, lengths, values=None):
        mask = self._attn_mask(pad_mask, self.n_heads)
        out = self.enc(x, mask=mask)                                          # (B, L, d)
        return out * (~pad_mask).unsqueeze(-1).to(out.dtype), None
    
MIXERS = {
    'mean':        lambda d, c: MeanMixer(d),
    'pool':        lambda d, c: AttnPoolMixer(d, gated=c.get('gated', True)),
    'conv':        lambda d, c: ConvMixer(d, n_blocks=c.get('n_blocks', 3),
                                          k=c.get('kernel', 5), mult=c.get('ff_mult', 2),
                                          dropout=c.get('dropout', 0.1)),
    'gru':         lambda d, c: GRUMixer(d, hidden=c.get('hidden', None),
                                         n_layers=c.get('n_layers', 1),
                                         dropout=c.get('dropout', 0.1),
                                         bidirectional=c.get('bidirectional', True)),
    'causal_gru': lambda d, c: CausalGRUMixer(d, hidden=c.get('hidden', None),
                                                n_layers=c.get('n_layers', 1),
                                                dropout=c.get('dropout', 0.1)),
    'transformer': lambda d, c: TransformerMixer(d, n_heads=c.get('n_heads', 8),
                                                 n_layers=c.get('n_layers', 4),
                                                 ff_mult=c.get('ff_mult', 4),
                                                 dropout=c.get('dropout', 0.1)),
    'causal_transformer': lambda d, c: CausalTransformerMixer(
        d, n_heads=c.get('n_heads', 8), n_layers=c.get('n_layers', 4),
        ff_mult=c.get('ff_mult', 4), dropout=c.get('dropout', 0.1)),
}
