"""View aggregation for multi-modal crops: (N, k, d) -> (N, d).

Every pooler has the same interface
    pooled, attn = pool(x, mask=None)
    x:    (N, n, d)     view features
    mask: (N, n) bool   True = view present (None = all present)
and is safe when a row has no valid views at all: it returns zeros, never NaN.

Two levels of use:
  * a plain pooler over all k views (modality-agnostic, as before);
  * ModalityViewAggregator, which knows the layout `view_groups` (crops of each modality
    concatenated along k, in a fixed order) and adds per-modality normalisation,
    position/modality embeddings, masking, and modality/crop dropout. It can pool
    either 'flat' (all views at once) or 'hierarchical' (within modality, then across).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# masking helpers
# ---------------------------------------------------------------------------

def _masked_softmax(logits, mask, dim):
    """Softmax over `dim` ignoring entries where mask is False.
    Rows with no valid entry come out as all zeros (no NaN in forward or backward)."""
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    w = torch.softmax(logits, dim) * mask
    return w / w.sum(dim, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)


def _effective(mask):
    """(mask_eff, any_valid). Fully-masked rows are unmasked in mask_eff so the inner
    computation stays finite; callers zero those rows at the end using any_valid."""
    any_ = mask.any(1)
    return mask | ~any_.unsqueeze(1), any_


def _zero_empty(out, any_):
    return torch.where(any_.view(-1, *([1] * (out.dim() - 1))), out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# poolers
# ---------------------------------------------------------------------------

class MeanPool(nn.Module):
    """Masked mean."""
    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(1), None
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp_min(1), None


class MaxPool(nn.Module):
    """Masked max over views."""
    def forward(self, x, mask=None):
        if mask is None:
            return x.amax(1), None
        out = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).amax(1)
        return _zero_empty(out, mask.any(1)), None


class LSEPool(nn.Module):
    """Smooth max  t * log(mean(exp(x / t)))  with a learnable (per-channel) temperature.
    t -> 0 gives max, t -> inf gives mean. Works with signed features (unlike GeM)."""
    def __init__(self, dim, init_t=1.0, per_channel=True):
        super().__init__()
        n = dim if per_channel else 1
        self.raw_t = nn.Parameter(torch.full((n,), math.log(math.expm1(init_t))))

    @property
    def temperature(self):
        return F.softplus(self.raw_t) + 1e-3

    def forward(self, x, mask=None):
        t = self.temperature.to(x.dtype)
        z = x / t
        if mask is None:
            return t * (torch.logsumexp(z, 1) - math.log(x.size(1))), None
        m_eff, any_ = _effective(mask)
        z = z.masked_fill(~m_eff.unsqueeze(-1), torch.finfo(z.dtype).min)
        cnt = m_eff.sum(1, keepdim=True).to(x.dtype)
        out = t * (torch.logsumexp(z, 1) - cnt.log())
        return _zero_empty(out, any_), None


class GatedAttentionPool(nn.Module):
    """Gated attention pooling (Ilse et al. 2018), now multi-head and maskable.

    heads=1 is exactly the original module (same parameter names/shapes, so old
    checkpoints load). With heads=H the feature dim is split into H slices and each
    slice gets its own softmax over views; heads=dim gives channel-wise attention
    (every feature picks its own view mix).
    attn: (N, n) if heads == 1 else (N, H, n).
    """
    def __init__(self, dim, hidden=128, heads=1, temperature=1.0, learn_temperature=False):
        super().__init__()
        if dim % heads:
            raise ValueError(f'dim ({dim}) must be divisible by heads ({heads})')
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, heads)
        self.heads = heads
        if learn_temperature:
            self.log_t = nn.Parameter(torch.tensor(math.log(temperature)))
        else:
            self.log_t = None
            self._t = float(temperature)

    def forward(self, x, mask=None):
        N, n, d = x.shape
        H = self.heads
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x)))      # (N, n, H)
        a = a / (self.log_t.exp() if self.log_t is not None else self._t)
        if mask is None:
            alpha = torch.softmax(a, dim=1)
        else:
            alpha = _masked_softmax(a, mask.unsqueeze(-1), dim=1)
        pooled = (alpha.unsqueeze(-1) * x.view(N, n, H, d // H)).sum(1).reshape(N, d)
        attn = alpha.squeeze(-1) if H == 1 else alpha.transpose(1, 2)
        return pooled, attn


class PMAPool(nn.Module):
    """Learned-query cross-attention pooling (Set Transformer PMA).

    n_sab > 0 adds that many pre-norm self-attention layers over the views first, so
    views (and, in flat mode, modalities) can interact before pooling.
    n_seeds > 1 uses several queries whose outputs are concatenated and projected.
    attn: (N, n) if heads == 1 else (N, H, n), averaged over seeds.
    """
    def __init__(self, dim, heads=4, n_seeds=1, n_sab=0, dropout=0.1, ff_mult=2):
        super().__init__()
        if dim % heads:
            raise ValueError(f'dim ({dim}) must be divisible by heads ({heads})')
        self.heads, self.n_seeds = heads, n_seeds
        self.sab = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, heads, ff_mult * dim, dropout,
                                       activation='gelu', batch_first=True, norm_first=True)
            for _ in range(n_sab)])
        self.seeds = nn.Parameter(torch.randn(n_seeds, dim) * 0.02)
        self.kv_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(n_seeds * dim, dim)

    def forward(self, x, mask=None):
        N, n, d = x.shape
        H, S, dh = self.heads, self.n_seeds, d // self.heads
        m_eff, any_ = _effective(mask) if mask is not None else (None, None)
        pad = ~m_eff if m_eff is not None else None
        for layer in self.sab:
            x = layer(x, src_key_padding_mask=pad)
        kv = self.kv_norm(x)
        q = self.q(self.seeds).view(S, H, dh)
        k = self.k(kv).view(N, n, H, dh)
        v = self.v(kv).view(N, n, H, dh)
        logits = torch.einsum('shc,nkhc->nhsk', q, k) / math.sqrt(dh)    # (N, H, S, n)
        if m_eff is None:
            attn = torch.softmax(logits, -1)
        else:
            attn = _masked_softmax(logits, m_eff[:, None, None, :], -1)
        out = torch.einsum('nhsk,nkhc->nshc', attn, v).reshape(N, S * d)
        out = self.o(out)
        if any_ is not None:
            out = _zero_empty(out, any_)
            attn = attn * any_.view(-1, 1, 1, 1)
        attn = attn.mean(2)                                                # (N, H, n)
        return out, (attn.squeeze(1) if H == 1 else attn)


class ConcatPool(nn.Module):
    """Position-specific fusion: concat the n views and project back to dim.
    Each position has its own weights, so order matters fully. Missing views are
    replaced by a learned per-position 'missing' token (requires a fixed n)."""
    def __init__(self, dim, n_views, hidden=None, dropout=0.1):
        super().__init__()
        self.n_views = n_views
        self.missing = nn.Parameter(torch.randn(n_views, dim) * 0.02)
        if hidden:
            self.proj = nn.Sequential(nn.Linear(n_views * dim, hidden), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, dim))
        else:
            self.proj = nn.Linear(n_views * dim, dim)

    def forward(self, x, mask=None):
        if x.size(1) != self.n_views:
            raise ValueError(f'ConcatPool built for {self.n_views} views, got {x.size(1)}')
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), x, self.missing.to(x.dtype).expand_as(x))
        return self.proj(x.flatten(1)), None


VIEW_POOLS = {
    'mean': MeanPool,
    'max': MaxPool,
    'lse': LSEPool,
    'attention': GatedAttentionPool,
    'pma': PMAPool,
    'transformer': PMAPool,    # PMA preceded by self-attention (n_sab defaults to 2)
    'concat': ConcatPool,
}


def build_view_pool(name, dim, n_views=None, **kw):
    if name not in VIEW_POOLS:
        raise ValueError(f'view pool must be one of {sorted(VIEW_POOLS)}, got {name!r}')
    if name in ('mean', 'max'):
        if kw:
            raise ValueError(f'{name!r} pool takes no arguments, got {sorted(kw)}')
        return VIEW_POOLS[name]()
    if name == 'concat':
        if n_views is None:
            raise ValueError("'concat' pool needs n_views")
        return ConcatPool(dim, n_views, **kw)
    if name == 'transformer':
        kw.setdefault('n_sab', 2)
    return VIEW_POOLS[name](dim, **kw)


# ---------------------------------------------------------------------------
# modality-aware aggregator
# ---------------------------------------------------------------------------

class ModalityViewAggregator(nn.Module):
    """Pools the k = sum(view_groups) crops of each token into one d_model vector.

    view_groups: dict {modality_name: n_crops} (ordered) or a sequence of ints.
                 Crops are expected concatenated along k in that order.
    mode:        'flat'         -> one `pool` over all k views (+ modality/position embeddings)
                 'hierarchical' -> `intra` pool inside each modality, then `inter` pool
                                   over the M modality tokens. `intra` can be one pool
                                   name, or a dict {modality: pool name} / a list; and
                                   intra_kwargs can be shared or {modality: kwargs}.
    view_pos:    learned embedding per crop slot (encodes modality + order within modality).
    modality_emb: learned embedding per modality (added to views in flat mode, to the
                 modality tokens in hierarchical mode).
    modality_norm: a separate LayerNorm per modality on the crop features.
    modality_dropout / view_dropout: training-only probabilities of hiding a whole
                 modality / a single crop. A token never loses all of its available views.
    dropout_level: 'subject' -> modality dropout is drawn once per sequence (all slots of a
                 subject lose the same modality, as at inference on a modality subset);
                 'token' -> drawn independently per token.

    Returns (pooled (N, d), info dict). info holds the attention weights:
      flat:         {'views': attn, 'view_mask'}
      hierarchical: {'intra': {name: attn}, 'inter': attn, 'combined': (N, k) or None,
                     'view_mask', 'modality_mask'}
    'combined' = intra weight x modality weight, available when both levels give
    single-head attention.
    """

    def __init__(self, d_model, view_groups, mode='hierarchical',
                 pool='attention', pool_kwargs=None,
                 intra='attention', inter='concat', intra_kwargs=None, inter_kwargs=None,
                 view_pos=True, modality_emb=True, modality_norm=True,
                 modality_dropout=0.0, view_dropout=0.0, dropout_level='subject'):
        super().__init__()
        if mode not in ('flat', 'hierarchical'):
            raise ValueError("mode must be 'flat' or 'hierarchical'")
        if dropout_level not in ('subject', 'token'):
            raise ValueError("dropout_level must be 'subject' or 'token'")
        if isinstance(view_groups, dict):
            self.names, sizes = list(view_groups), [int(v) for v in view_groups.values()]
        else:
            sizes = [int(v) for v in view_groups]
            self.names = [f'm{i}' for i in range(len(sizes))]
        if not sizes or min(sizes) < 1:
            raise ValueError('every modality needs at least one crop')

        self.mode, self.sizes, self.M, self.k = mode, sizes, len(sizes), sum(sizes)
        offs = [0]
        for s in sizes:
            offs.append(offs[-1] + s)
        self.slices = [slice(a, b) for a, b in zip(offs[:-1], offs[1:])]
        self.register_buffer('mod_of_view',
                             torch.repeat_interleave(torch.arange(self.M), torch.tensor(sizes)),
                             persistent=False)
        self.modality_dropout, self.view_dropout = modality_dropout, view_dropout
        self.dropout_level = dropout_level

        self.mod_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in sizes]) if modality_norm else None
        self.view_pos = nn.Parameter(torch.randn(self.k, d_model) * 0.02) if view_pos else None
        self.mod_emb = nn.Parameter(torch.randn(self.M, d_model) * 0.02) if modality_emb else None

        if mode == 'flat':
            self.pool = build_view_pool(pool, d_model, n_views=self.k, **(pool_kwargs or {}))
        else:
            pools = self._per_modality(intra, 'intra')
            kws = intra_kwargs or {}
            if kws and set(kws) <= set(self.names):          # {name: kwargs} per modality
                kws = [kws.get(n, {}) for n in self.names]
            else:                                            # shared kwargs
                kws = [kws] * self.M
            self.intra = nn.ModuleList([
                build_view_pool(p, d_model, n_views=n, **kw) for p, n, kw in zip(pools, sizes, kws)])
            self.tok_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in sizes])
            self.inter = build_view_pool(inter, d_model, n_views=self.M, **(inter_kwargs or {}))

    def _per_modality(self, spec, what):
        if isinstance(spec, str):
            return [spec] * self.M
        if isinstance(spec, dict):
            missing = set(self.names) - set(spec)
            if missing:
                raise ValueError(f'{what}: no pool given for modalities {sorted(missing)}')
            return [spec[n] for n in self.names]
        spec = list(spec)
        if len(spec) != self.M:
            raise ValueError(f'{what}: expected {self.M} pools, got {len(spec)}')
        return spec

    # ---- masks ---------------------------------------------------------------

    def modality_view_mask(self, modalities, device=None):
        """(k,) bool mask keeping only the given modalities (names or indices)."""
        keep = torch.zeros(self.M, dtype=torch.bool, device=device)
        for m in modalities:
            if isinstance(m, str):
                if m not in self.names:
                    raise ValueError(f'unknown modality {m!r}; known: {self.names}')
                m = self.names.index(m)
            keep[m] = True
        return keep[self.mod_of_view.to(device)]

    def _drop(self, mask, group_ids):
        if not self.training or (self.modality_dropout <= 0 and self.view_dropout <= 0):
            return mask
        N, dev = mask.size(0), mask.device
        new = mask
        if self.modality_dropout > 0:
            if self.dropout_level == 'subject' and group_ids is not None:
                G = int(group_ids.max()) + 1 if group_ids.numel() else 1
                keep = (torch.rand(G, self.M, device=dev) >= self.modality_dropout)[group_ids]
            else:
                keep = torch.rand(N, self.M, device=dev) >= self.modality_dropout
            new = new & keep[:, self.mod_of_view]
        if self.view_dropout > 0:
            new = new & (torch.rand(N, self.k, device=dev) >= self.view_dropout)
        lost_all = ~new.any(1) & mask.any(1)          # never leave a token with nothing
        return torch.where(lost_all.unsqueeze(1), mask, new)

    # ---- forward -------------------------------------------------------------

    def forward(self, x, mask=None, group_ids=None):
        N, k, _ = x.shape
        if k != self.k:
            raise ValueError(f'expected {self.k} views {dict(zip(self.names, self.sizes))}, got {k}')
        if mask is None:
            mask = torch.ones(N, k, dtype=torch.bool, device=x.device)
        mask = self._drop(mask.bool(), group_ids)

        if self.mod_norm is not None:
            x = torch.cat([ln(x[:, s]) for ln, s in zip(self.mod_norm, self.slices)], 1)
        if self.view_pos is not None:
            x = x + self.view_pos.to(x.dtype)
        if self.mode == 'flat' and self.mod_emb is not None:
            x = x + self.mod_emb.to(x.dtype)[self.mod_of_view]
        x = x * mask.unsqueeze(-1).to(x.dtype)               # missing / imputed crops -> 0

        if self.mode == 'flat':
            z, attn = self.pool(x, mask)
            return z, {'views': attn, 'view_mask': mask}

        tokens, intra_attn = [], {}
        for name, s, pool, ln in zip(self.names, self.slices, self.intra, self.tok_norm):
            zm, am = pool(x[:, s], mask[:, s])
            tokens.append(ln(zm))
            intra_attn[name] = am
        T = torch.stack(tokens, 1)                            # (N, M, d)
        if self.mod_emb is not None:
            T = T + self.mod_emb.to(T.dtype)
        mod_mask = torch.stack([mask[:, s].any(1) for s in self.slices], 1)
        T = T * mod_mask.unsqueeze(-1).to(T.dtype)
        z, inter_attn = self.inter(T, mod_mask)

        combined = None
        if (inter_attn is not None and inter_attn.dim() == 2
                and all(a is not None and a.dim() == 2 for a in intra_attn.values())):
            combined = torch.cat([intra_attn[n] * inter_attn[:, i:i + 1]
                                  for i, n in enumerate(self.names)], 1)
        return z, {'intra': intra_attn, 'inter': inter_attn, 'combined': combined,
                   'view_mask': mask, 'modality_mask': mod_mask}