"""Conditional VAE: the decoder knows WHAT was written (modality), so z only has to
encode HOW it was written. An adversary additionally pushes modality information
out of z: the adversary learns to predict modality from a detached z, and the
encoder is trained to make the adversary's prediction uniform (bounded
'confusion' loss). Plain gradient reversal is avoided on purpose: maximising
cross-entropy is unbounded, and the encoder learns to make the adversary
confidently WRONG by inflating mu, which keeps modality fully decodable."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import NUM_MODALITIES
from .base import PretrainModule
from torch.func import functional_call
from .cnn import ConvDecoder, ConvEncoder
from .losses import ReconLoss


class _CVAENet(nn.Module):
    def __init__(self, image_size, channels, blocks_per_stage, z_dim, cond_dim, dropout):
        super().__init__()
        self.encoder = ConvEncoder(channels, blocks_per_stage, out_channels=32, dropout=dropout)
        self.h = image_size // self.encoder.downsample_factor
        flat = 32 * self.h * self.h
        self.to_stats = nn.Linear(flat, 2 * z_dim)

        self.modality_embed = nn.Embedding(NUM_MODALITIES, cond_dim)
        # z also modulates every decoder block through FiLM, so style is not
        # bottlenecked through the initial spatial map alone
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim + z_dim, cond_dim), nn.SiLU(),
                                      nn.Linear(cond_dim, cond_dim))
        self.from_z = nn.Linear(z_dim, 32 * self.h * self.h)
        self.decoder = ConvDecoder(32, channels, blocks_per_stage, cond_dim=cond_dim, dropout=dropout)

        self.adversary = nn.Sequential(nn.Linear(z_dim, 128), nn.SiLU(), nn.Linear(128, NUM_MODALITIES))

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        mu, logvar = self.to_stats(h).chunk(2, dim=1)
        return mu, logvar.clamp(-10, 10)

    def decode(self, z, c):
        cond = self.cond_mlp(torch.cat([self.modality_embed(c), z], dim=1))
        h = self.from_z(z).view(z.shape[0], 32, self.h, self.h)
        return self.decoder(h, cond)


class ConditionalVAE(PretrainModule):
    """
    Loss (per pixel scale):
        recon + beta(t) * KL_freebits / n_pixels
              + adv_weight * CE(adversary(mu.detach()), modality)        # trains adversary only
              + confusion_weight(t) * CE(uniform, adversary_frozen(mu))  # trains encoder only

    Args:
        image_size: side of the square input.
        z_dim: latent size. 64-128 is a reasonable range for single symbols.
        beta: KL weight at the end of warmup. The recon term is a per-pixel mean
            rescaled by n_pixels, so beta=1 is the standard ELBO with a Laplace
            likelihood. Lower beta -> sharper reconstructions (keeps tremor).
        free_bits: minimum KL (nats) per latent dim before it is penalised;
            prevents posterior collapse.
        adv_weight: weight of the adversary's own cross-entropy.
        confusion_weight / confusion_warmup_steps: strength (and ramp) of the term
            pushing the encoder to make the adversary output uniform.
    """

    def __init__(
        self,
        image_size: int = 32,
        channels: tuple[int, ...] = (32, 64, 128),
        blocks_per_stage: int = 2,
        z_dim: int = 64,
        cond_dim: int = 64,
        dropout: float = 0.0,
        beta: float = 0.5,
        kl_warmup_steps: int = 5000,
        free_bits: float = 0.1,
        adv_weight: float = 0.1,
        confusion_weight: float = 1.0,
        confusion_warmup_steps: int = 5000,
        fg_weight: float = 5.0,
        grad_weight: float = 1.0,
        fft_weight: float = 0.0,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.save_hyperparameters()
        self.net = _CVAENet(image_size, channels, blocks_per_stage, z_dim, cond_dim, dropout)
        self.recon_loss = ReconLoss(fg_weight=fg_weight, grad_weight=grad_weight, fft_weight=fft_weight)
        self.init_ema()

    def _ramp(self, warmup: int) -> float:
        if not self.training or warmup <= 0:
            return 1.0
        return min(1.0, self.global_step / warmup)

    def compute_loss(self, batch, net):
        hp = self.hparams
        x, c = self.check_batch(batch)
        n_pix = x.shape[-1] * x.shape[-2]

        mu, logvar = net.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        x_hat = net.decode(z, c)
        recon, parts = self.recon_loss(x_hat, x)

        kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean(0)  # (z_dim,) batch-avg
        kl = kl_dim.clamp_min(hp.free_bits).sum()
        beta_t = hp.beta * self._ramp(hp.kl_warmup_steps)

        # adversary: learns to read modality from a detached z
        logits = net.adversary(mu.detach())
        adv = F.cross_entropy(logits, c)
        # encoder: make the (frozen) adversary output uniform -> bounded, minimum log(3)
        frozen = {k: v.detach() for k, v in net.adversary.named_parameters()}
        logits_enc = functional_call(net.adversary, frozen, (mu,))
        confusion = -F.log_softmax(logits_enc, dim=1).mean(1).mean()
        conf_w = hp.confusion_weight * self._ramp(hp.confusion_warmup_steps)

        loss = recon + beta_t * kl / n_pix + hp.adv_weight * adv + conf_w * confusion
        logs = {
            **parts,
            "kl": kl_dim.sum(),
            "kl_active_dims": (kl_dim > 0.01).float().sum(),
            "beta": torch.tensor(beta_t),
            "adv_ce": adv,
            "confusion": confusion,  # -> log(3) = 1.099 when z carries no modality info
            # should drift towards chance (1/3) as z sheds modality information
            "adv_acc": (logits.argmax(1) == c).float().mean(),
        }
        return loss, logs

    @torch.no_grad()
    def extract_features(self, image, modality):
        """Posterior mean mu -> (B, z_dim). Modality is not used by the encoder."""
        return self.eval_net.eval().encode(image)[0]

    @torch.no_grad()
    def reconstruction_error(self, image, modality):
        """Per-sample reconstruction loss from mu; a useful extra scalar feature."""
        net = self.eval_net.eval()
        x_hat = net.decode(net.encode(image)[0], modality.long())
        return self.recon_loss(x_hat, image, reduce=False)[0]

    @torch.no_grad()
    def preview(self, batch):
        x, c = self.check_batch(batch)
        net = self.eval_net.eval()
        mu, _ = net.encode(x)
        # same z, decoded as each of the 3 modalities: shows what z carries
        swaps = {f"as_mod{k}": net.decode(mu, torch.full_like(c, k)) for k in range(NUM_MODALITIES)}
        return {"input": x, "recon": net.decode(mu, c), **swaps}
