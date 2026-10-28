"""Denoising autoencoder with structured (stroke-preserving) corruption."""
from __future__ import annotations

import torch
import torch.nn as nn

from . import NUM_MODALITIES
from .base import PretrainModule
from .blocks import dilate, ink_mask, masked_spatial_mean
from .cnn import ConvDecoder, ConvEncoder
from .losses import ReconLoss


class StructuredCorruption(nn.Module):
    """Corruptions that remove *information* without teaching the model to smooth strokes.

    * block erasure  : random rectangles set to background
    * stroke dropout : small squares erased at random ink locations (breaks strokes)
    * background noise: low-amplitude noise / specks on background pixels only

    Deliberately absent: blur, strong pixel noise on strokes, rescaling, rotation.
    Those would teach the network that tremor-scale detail is noise to remove.
    """

    def __init__(
        self,
        p_block: float = 0.5,
        block_frac: tuple[float, float] = (0.1, 0.25),
        n_stroke_erasures: int = 2,
        stroke_erase_frac: float = 0.08,
        bg_noise_std: float = 0.05,
        speck_prob: float = 0.002,
    ):
        super().__init__()
        self.p_block = p_block
        self.block_frac = block_frac
        self.n_stroke_erasures = n_stroke_erasures
        self.stroke_erase_frac = stroke_erase_frac
        self.bg_noise_std = bg_noise_std
        self.speck_prob = speck_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        dev = x.device
        out = x.clone()
        ys = torch.arange(H, device=dev).view(1, H, 1)
        xs = torch.arange(W, device=dev).view(1, 1, W)

        # 1) block erasure
        lo, hi = self.block_frac
        bh = (torch.empty(B, device=dev).uniform_(lo, hi) * H).long().clamp_min(1)
        bw = (torch.empty(B, device=dev).uniform_(lo, hi) * W).long().clamp_min(1)
        y0 = (torch.rand(B, device=dev) * (H - bh + 1)).long()
        x0 = (torch.rand(B, device=dev) * (W - bw + 1)).long()
        block = (
            (ys >= y0.view(B, 1, 1)) & (ys < (y0 + bh).view(B, 1, 1))
            & (xs >= x0.view(B, 1, 1)) & (xs < (x0 + bw).view(B, 1, 1))
        )
        use = (torch.rand(B, device=dev) < self.p_block).view(B, 1, 1)
        erase = block & use

        # 2) stroke dropout: erase squares centred on randomly chosen ink pixels
        ink = ink_mask(x).view(B, -1)
        has_ink = ink.sum(1) > 0
        probs = torch.where(has_ink[:, None], ink, torch.ones_like(ink))
        s = max(1, int(round(self.stroke_erase_frac * min(H, W))))
        for _ in range(self.n_stroke_erasures):
            idx = torch.multinomial(probs, 1).squeeze(1)
            cy, cx = (idx // W).view(B, 1, 1), (idx % W).view(B, 1, 1)
            erase |= ((ys - cy).abs() <= s // 2) & ((xs - cx).abs() <= s // 2)

        out = torch.where(erase.unsqueeze(1), torch.full_like(out, -1.0), out)

        # 3) background-only noise and specks (clutter), strokes left untouched
        bg = 1.0 - dilate(ink_mask(x), 1)
        if self.bg_noise_std > 0:
            out = out + bg * self.bg_noise_std * torch.randn_like(out).abs()
        if self.speck_prob > 0:
            specks = (torch.rand_like(out) < self.speck_prob).to(out.dtype) * bg
            out = torch.where(specks > 0, torch.ones_like(out), out)
        return out.clamp(-1, 1)


class _DAENet(nn.Module):
    def __init__(self, channels, blocks_per_stage, latent_channels, cond_dim, dropout):
        super().__init__()
        self.encoder = ConvEncoder(channels, blocks_per_stage, latent_channels, dropout)
        self.modality_embed = nn.Embedding(NUM_MODALITIES, cond_dim) if cond_dim else None
        self.decoder = ConvDecoder(latent_channels, channels, blocks_per_stage, cond_dim, dropout)

    def forward(self, x, c):
        z = self.encoder(x)
        cond = self.modality_embed(c) if self.modality_embed is not None else None
        return self.decoder(z, cond), z


class DenoisingAE(PretrainModule):
    """Encoder -> spatial latent -> (modality-conditioned) decoder; reconstruct clean from corrupted.

    Args:
        channels: encoder widths per stage; spatial latent size is H / 2**(len(channels)-1).
        latent_channels: channels of the bottleneck. Together with the number of stages
            this sets the bottleneck size; keep it well below H*W.
        cond_dim: modality embedding size for the decoder (0 = unconditional).
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (32, 64, 128),
        blocks_per_stage: int = 2,
        latent_channels: int = 8,
        cond_dim: int = 64,
        dropout: float = 0.0,
        corruption: dict | None = None,
        fg_weight: float = 5.0,
        grad_weight: float = 1.0,
        fft_weight: float = 0.0,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.save_hyperparameters()
        self.net = _DAENet(channels, blocks_per_stage, latent_channels, cond_dim, dropout)
        self.corrupt = StructuredCorruption(**(corruption or {}))
        self.recon_loss = ReconLoss(fg_weight=fg_weight, grad_weight=grad_weight, fft_weight=fft_weight)
        self.init_ema()

    def compute_loss(self, batch, net):
        x, c = self.check_batch(batch)
        x_tilde = self.corrupt(x)
        x_hat, _ = net(x_tilde, c)
        loss, parts = self.recon_loss(x_hat, x)
        return loss, parts

    @torch.no_grad()
    def extract_features(self, image, modality):
        """Latent of the *clean* image: [global mean | ink-region mean] -> (B, 2*latent_channels)."""
        net = self.eval_net.eval()
        z = net.encoder(image)
        return torch.cat([z.mean(dim=(2, 3)), masked_spatial_mean(z, ink_mask(image))], dim=1)

    @torch.no_grad()
    def preview(self, batch):
        x, c = self.check_batch(batch)
        x_tilde = self.corrupt(x)
        net = self.eval_net.eval()
        return {"input": x, "corrupted": x_tilde, "recon": net(x_tilde, c)[0], "recon_clean": net(x, c)[0]}
