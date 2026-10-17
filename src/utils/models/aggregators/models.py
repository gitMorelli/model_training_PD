from torch import nn
import torch

#Multiple instance learning modules
class GatedAttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
    def forward(self, x):           
        #x: (B, n, d_model)                        
        #torch.tanh(self.V(x)) -> (B, n, hidden)
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x)))
        #a: (B, n, 1)
        alpha = torch.softmax(a, dim=1)
        return (alpha * x).sum(1), alpha.squeeze(-1) #(B,d_model), (B,n)
class SegmentedGatedAttentionPool(nn.Module):
    """Gated attention (Ilse et al., ABMIL) over ragged segments.

    h:   (N, D)  per-timestep features
    seq: (N,)    segment id in [0, B)
    ->   (B, D) pooled, (N,) per-timestep weights (sum to 1 within each segment)
    """
    def __init__(self, d_model, d_attn=None):
        super().__init__()
        d_attn = d_attn or d_model // 2
        self.V = nn.Linear(d_model, d_attn)
        self.U = nn.Linear(d_model, d_attn)
        self.w = nn.Linear(d_attn, 1)

    def forward(self, h, seq, B):
        dev, dt = h.device, h.dtype
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        scores = self.w(a)                                            # (N, 1)

        # segmented softmax with per-segment max subtraction
        m = torch.zeros(B, 1, device=dev, dtype=dt).index_reduce_(
            0, seq, scores, 'amax', include_self=False)
        e = torch.exp(scores - m[seq])
        Z = torch.zeros(B, 1, device=dev, dtype=dt).index_add_(0, seq, e)
        w = e / Z[seq].clamp_min(torch.finfo(dt).tiny)

        pooled = torch.zeros(B, h.size(-1), device=dev, dtype=dt
                             ).index_add_(0, seq, w * h)
        return pooled, w.squeeze(1)
