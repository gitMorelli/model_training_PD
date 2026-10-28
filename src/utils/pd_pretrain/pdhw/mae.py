"""Masked autoencoder (ViT) with ink-aware masking.

Plain random masking fails on handwriting: most patches are blank paper, so most
masked patches are trivially predicted as 'empty'. Here ink and empty patches
are masked at separate rates, and the loss is concentrated on masked ink patches.
Because the number of visible patches then varies per sample, the encoder takes
a key-padding mask instead of fixed-length visible sets.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import NUM_MODALITIES
from .base import PretrainModule


def sincos_pos_embed(dim: int, grid: int) -> torch.Tensor:
    """Fixed 2-D sin-cos positional embedding, (grid*grid, dim)."""
    assert dim % 4 == 0
    y, x = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                          torch.arange(grid, dtype=torch.float32), indexing="ij")
    omega = 1.0 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)))
    out = []
    for coord in (y.flatten(), x.flatten()):
        arg = coord[:, None] * omega[None]
        out += [arg.sin(), arg.cos()]
    return torch.cat(out, dim=1)


def make_transformer(dim: int, depth: int, heads: int, mlp_ratio: float, dropout: float):
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio), dropout=dropout,
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class _MAENet(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, dec_dim, dec_depth, dec_heads,
                 mlp_ratio, dropout):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.p = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid ** 2
        pdim = patch_size * patch_size

        # encoder
        self.patch_embed = nn.Linear(pdim, dim)
        self.register_buffer("pos", sincos_pos_embed(dim, self.grid)[None], persistent=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.modality_token = nn.Embedding(NUM_MODALITIES, dim)
        self.encoder = make_transformer(dim, depth, heads, mlp_ratio, dropout)
        self.enc_norm = nn.LayerNorm(dim)

        # decoder
        self.dec_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("dec_pos", sincos_pos_embed(dec_dim, self.grid)[None], persistent=False)
        self.decoder = make_transformer(dec_dim, dec_depth, dec_heads, mlp_ratio, dropout)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, pdim)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.modality_token.weight, std=0.02)

    # --------------------------------------------------------------- patches
    def patchify(self, x):  # (B,1,H,W) -> (B,L,p*p)
        B, _, H, W = x.shape
        p = self.p
        x = x.reshape(B, H // p, p, W // p, p).permute(0, 1, 3, 2, 4)
        return x.reshape(B, (H // p) * (W // p), p * p)

    def unpatchify(self, t):  # (B,L,p*p) -> (B,1,H,W)
        B, L, _ = t.shape
        g, p = self.grid, self.p
        t = t.reshape(B, g, g, p, p).permute(0, 1, 3, 2, 4)
        return t.reshape(B, 1, g * p, g * p)

    # ---------------------------------------------------------------- encode
    def encode(self, x, c, mask=None):
        """mask: (B,L) bool, True = hidden. Returns (special tokens, visible tokens, ids, pad)."""
        B = x.shape[0]
        tok = self.patch_embed(self.patchify(x)) + self.pos
        if mask is None:
            mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=x.device)
        # stable sort puts visible patches first, in their original order
        ids = torch.sort(mask.to(torch.uint8), dim=1, stable=True).indices
        n_vis = (~mask).sum(1)
        max_vis = int(n_vis.max())
        ids_vis = ids[:, :max_vis]
        vis = torch.gather(tok, 1, ids_vis[..., None].expand(-1, -1, tok.shape[-1]))
        pad = torch.arange(max_vis, device=x.device)[None] >= n_vis[:, None]  # True = padding

        special = torch.cat([self.cls_token.expand(B, -1, -1), self.modality_token(c)[:, None]], 1)
        seq = torch.cat([special, vis], 1)
        key_pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=x.device), pad], 1)
        h = self.enc_norm(self.encoder(seq, src_key_padding_mask=key_pad))
        return h[:, :2], h[:, 2:], ids_vis, pad

    def decode(self, special, vis, ids_vis, pad):
        B, L = special.shape[0], self.num_patches
        v = self.dec_embed(vis)
        full = self.mask_token.expand(B, L, -1).clone()
        # scatter visible tokens back; padded slots are routed to a dummy index L
        idx = torch.where(pad, torch.full_like(ids_vis, L), ids_vis)
        full = torch.cat([full, torch.zeros_like(full[:, :1])], 1)
        full.scatter_(1, idx[..., None].expand(-1, -1, v.shape[-1]), v)
        full = full[:, :L] + self.dec_pos
        seq = torch.cat([self.dec_embed(special), full], 1)
        h = self.dec_norm(self.decoder(seq))
        return self.dec_pred(h[:, 2:])  # (B, L, p*p)


class MaskedAE(PretrainModule):
    """Ink-aware MAE.

    Args:
        image_size: side of the (square) input. Must match the dataloader.
        patch_size: 4 is a good default for 32/64 px inputs, 8 for 128 px.
        mask_ratio_ink / mask_ratio_empty: fraction of ink / empty patches hidden.
        ink_patch_threshold: fraction of ink pixels for a patch to count as 'ink'.
        empty_loss_weight: loss weight on masked empty patches (ink patches = 1).
        norm_pix_loss: normalise each target patch by its own mean/std (MAE paper);
            emphasises local stroke structure over absolute intensity.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        dec_dim: int = 128,
        dec_depth: int = 2,
        dec_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mask_ratio_ink: float = 0.6,
        mask_ratio_empty: float = 0.5,
        ink_patch_threshold: float = 0.02,
        empty_loss_weight: float = 0.05,
        norm_pix_loss: bool = True,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.save_hyperparameters()
        self.net = _MAENet(image_size, patch_size, dim, depth, heads, dec_dim, dec_depth,
                           dec_heads, mlp_ratio, dropout)
        self.init_ema()

    # ----------------------------------------------------------------- mask
    def ink_patches(self, x, net):
        frac = (net.patchify(x) > 0).float().mean(-1)
        return frac > self.hparams.ink_patch_threshold  # (B, L) bool

    def sample_mask(self, ink: torch.Tensor) -> torch.Tensor:
        hp = self.hparams
        r = torch.rand(ink.shape, device=ink.device)
        mask = torch.where(ink, r < hp.mask_ratio_ink, r < hp.mask_ratio_empty)
        # keep at least one visible patch per sample, preferring an ink patch
        all_hidden = torch.where(ink.any(1), (mask | ~ink).all(1), mask.all(1))
        if all_hidden.any():
            score = torch.rand_like(r) + ink.float()  # ink patches win if present
            keep = score.argmax(1)
            rows = all_hidden.nonzero().squeeze(1)
            mask[rows, keep[rows]] = False
        return mask

    # ----------------------------------------------------------------- loss
    def compute_loss(self, batch, net):
        x, c = self.check_batch(batch)
        ink = self.ink_patches(x, net)
        mask = self.sample_mask(ink)
        special, vis, ids_vis, pad = net.encode(x, c, mask)
        pred = net.decode(special, vis, ids_vis, pad)

        target = net.patchify(x)
        if self.hparams.norm_pix_loss:
            mu = target.mean(-1, keepdim=True)
            var = target.var(-1, keepdim=True)
            target = (target - mu) / (var + 1e-4).sqrt()
        err = ((pred - target) ** 2).mean(-1)  # (B, L)
        w = mask.float() * torch.where(ink, 1.0, self.hparams.empty_loss_weight)
        loss = (err * w).sum() / w.sum().clamp_min(1.0)
        logs = {
            "masked_ink_frac": (mask & ink).float().sum() / ink.float().sum().clamp_min(1),
            "ink_patch_frac": ink.float().mean(),
        }
        return loss, logs

    # ------------------------------------------------------------- features
    @torch.no_grad()
    def extract_features(self, image, modality):
        """Unmasked encoder: [CLS | mean over ink tokens | mean over all tokens] -> (B, 3*dim)."""
        net = self.eval_net.eval()
        special, tok, _, _ = net.encode(image, modality.long(), None)
        ink = self.ink_patches(image, net).float()[..., None]
        ink_mean = (tok * ink).sum(1) / ink.sum(1).clamp_min(1)
        ink_mean = torch.where(ink.sum(1) > 0, ink_mean, tok.mean(1))
        return torch.cat([special[:, 0], ink_mean, tok.mean(1)], dim=1)

    @torch.no_grad()
    def preview(self, batch):
        x, c = self.check_batch(batch)
        net = self.eval_net.eval()
        ink = self.ink_patches(x, net)
        mask = self.sample_mask(ink)
        pred = net.decode(*net.encode(x, c, mask))
        if self.hparams.norm_pix_loss:  # undo per-patch normalisation using true stats
            t = net.patchify(x)
            pred = pred * (t.var(-1, keepdim=True) + 1e-4).sqrt() + t.mean(-1, keepdim=True)
        m = mask[..., None].float()
        masked_in = net.unpatchify(net.patchify(x) * (1 - m) + (-1.0) * m)
        pasted = net.unpatchify(net.patchify(x) * (1 - m) + pred.clamp(-1, 1) * m)
        return {"input": x, "masked": masked_in, "recon": pasted}
