"""Class-conditional diffusion model (diffusers UNet2DModel) used as a feature extractor
and as a normative model for anomaly residuals.

* Training: v-prediction, cosine schedule, min-SNR-gamma weighting, optional ink
  weighting, modality dropped 10% of the time (null class) so the model also
  works unconditionally.
* extract_features: noise the image at a few timesteps, one forward pass each,
  pool mid/up-block activations; averaged over seeded noise draws.
* anomaly_maps / anomaly_features: partially noise to t0, DDIM-denoise back, and
  measure |x - x_hat|. Train on HEALTHY subjects only for this use.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

from . import NUM_MODALITIES
from .base import PretrainModule
from .blocks import dilate, ink_mask, masked_spatial_mean


def build_unet(image_size, block_out_channels, layers_per_block, attn_levels, dropout,
               norm_num_groups, attention_head_dim) -> UNet2DModel:
    n = len(block_out_channels)
    down = tuple("AttnDownBlock2D" if i in attn_levels else "DownBlock2D" for i in range(n))
    up = tuple("AttnUpBlock2D" if (n - 1 - i) in attn_levels else "UpBlock2D" for i in range(n))
    return UNet2DModel(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        block_out_channels=tuple(block_out_channels),
        layers_per_block=layers_per_block,
        down_block_types=down,
        up_block_types=up,
        dropout=dropout,
        norm_num_groups=norm_num_groups,
        attention_head_dim=attention_head_dim,
        resnet_time_scale_shift="scale_shift",  # AdaGN conditioning
        num_class_embeds=NUM_MODALITIES + 1,  # last index = null (unconditional)
    )


class DiffusionPretrainer(PretrainModule):
    """
    Args:
        image_size: side of the square input (32, 64 or 128).
        block_out_channels: UNet widths per level; len-1 downsamplings.
            32 px: (64, 128, 256). 64 px: (64, 128, 256, 256). 128 px: (64, 128, 192, 256, 256).
        attn_levels: indices of levels that get self-attention (keep to low resolutions).
        cond_drop_prob: probability of replacing the modality with the null class.
        snr_gamma: min-SNR-gamma loss weighting (Hang et al. 2023); 0 disables.
        fg_weight: extra loss weight on ink pixels. Per-pixel weights do not change the
            optimum of the denoising objective, only where capacity is spent.
        feature_timesteps: noise levels (of 1000) used by extract_features.
        feature_draws: noise draws averaged per timestep (features are deterministic
            given feature_seed).
    """

    def __init__(
        self,
        image_size: int = 32,
        block_out_channels: Sequence[int] = (64, 128, 256),
        layers_per_block: int = 2,
        attn_levels: Sequence[int] = (2,),
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        attention_head_dim: int = 32,
        num_train_timesteps: int = 1000,
        cond_drop_prob: float = 0.1,
        snr_gamma: float = 5.0,
        fg_weight: float = 3.0,
        feature_timesteps: Sequence[int] = (50, 150, 300),
        feature_draws: int = 4,
        feature_seed: int = 0,
        feature_use_modality: bool = True,
        **base_kwargs,
    ):
        base_kwargs.setdefault("ema_decay", 0.9995)
        super().__init__(**base_kwargs)
        self.save_hyperparameters()
        self.net = build_unet(image_size, block_out_channels, layers_per_block, set(attn_levels),
                              dropout, norm_num_groups, attention_head_dim)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
            clip_sample=False,
        )
        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod.clone(), persistent=False)
        self.null_class = NUM_MODALITIES
        self.init_ema()

    # ------------------------------------------------------------- training
    def compute_loss(self, batch, net):
        hp = self.hparams
        x0, c = self.check_batch(batch)
        B = x0.shape[0]
        t = torch.randint(0, hp.num_train_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.scheduler.add_noise(x0, noise, t)
        target = self.scheduler.get_velocity(x0, noise, t)

        if self.training and hp.cond_drop_prob > 0:
            drop = torch.rand(B, device=x0.device) < hp.cond_drop_prob
            c = torch.where(drop, torch.full_like(c, self.null_class), c)

        pred = net(xt, t, class_labels=c).sample
        w = 1.0 + (hp.fg_weight - 1.0) * dilate(ink_mask(x0), 1)
        mse = ((pred - target) ** 2 * w).sum((1, 2, 3)) / w.sum((1, 2, 3))

        if hp.snr_gamma > 0:
            ac = self.alphas_cumprod[t]
            snr = ac / (1 - ac)
            weight = snr.clamp(max=hp.snr_gamma) / (snr + 1)  # min-SNR for v-prediction
            loss = (mse * weight).mean()
        else:
            loss = mse.mean()
        return loss, {"mse": mse.mean()}

    # ------------------------------------------------------------- features
    def _feature_layers(self, net):
        return [("mid", net.mid_block)] + [(f"up{i}", b) for i, b in enumerate(net.up_blocks)]

    @torch.no_grad()
    def extract_features(self, image, modality, timesteps: Sequence[int] | None = None,
                         draws: int | None = None):
        """-> (B, D). For each timestep and hooked layer: [global mean | ink-region mean]."""
        hp = self.hparams
        timesteps = list(timesteps or hp.feature_timesteps)
        draws = draws or hp.feature_draws
        net = self.eval_net.eval()
        c = modality.long() if hp.feature_use_modality else torch.full_like(modality.long(), self.null_class)
        ink = ink_mask(image)

        acts: dict[str, torch.Tensor] = {}
        handles = [
            layer.register_forward_hook(lambda m, i, o, name=name: acts.__setitem__(name, o))
            for name, layer in self._feature_layers(net)
        ]
        g = torch.Generator(device=image.device).manual_seed(hp.feature_seed)
        feats = []
        try:
            for t_val in timesteps:
                t = torch.full((image.shape[0],), t_val, device=image.device, dtype=torch.long)
                acc = None
                for _ in range(draws):
                    noise = torch.randn(image.shape, generator=g, device=image.device, dtype=image.dtype)
                    net(self.scheduler.add_noise(image, noise, t), t, class_labels=c)
                    pooled = torch.cat(
                        [torch.cat([a.mean((2, 3)), masked_spatial_mean(a, ink)], 1) for a in acts.values()], 1)
                    acc = pooled if acc is None else acc + pooled
                feats.append(acc / draws)
        finally:
            for h in handles:
                h.remove()
        return torch.cat(feats, dim=1)

    # ------------------------------------------------------ DDIM utilities
    def _ddim(self, num_inference_steps: int) -> DDIMScheduler:
        sched = DDIMScheduler.from_config(self.scheduler.config)
        sched.set_timesteps(num_inference_steps, device=self.device)
        return sched

    def _model_out(self, net, x, t, c, guidance_scale: float):
        if guidance_scale == 1.0:
            return net(x, t, class_labels=c).sample
        both = net(torch.cat([x, x]), t, class_labels=torch.cat([c, torch.full_like(c, self.null_class)])).sample
        cond, uncond = both.chunk(2)
        return uncond + guidance_scale * (cond - uncond)

    @torch.no_grad()
    def sample(self, modality: torch.Tensor, num_inference_steps: int = 50, guidance_scale: float = 1.0,
               seed: int | None = None) -> torch.Tensor:
        net = self.eval_net.eval()
        s = self.hparams.image_size
        g = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        x = torch.randn((modality.shape[0], 1, s, s), generator=g, device=self.device)
        sched = self._ddim(num_inference_steps)
        for t in sched.timesteps:
            x = sched.step(self._model_out(net, x, t, modality.long(), guidance_scale), t, x).prev_sample
        return x.clamp(-1, 1)

    @torch.no_grad()
    def anomaly_maps(self, image, modality, t0: int = 200, num_inference_steps: int = 50,
                     draws: int = 1, seed: int = 0, guidance_scale: float = 1.0):
        """Partial noising to t0 then DDIM denoising back. Returns (x_hat, |x - x_hat|),
        both averaged over `draws` seeded noise draws."""
        net = self.eval_net.eval()
        sched = self._ddim(num_inference_steps)
        ts = sched.timesteps[sched.timesteps <= t0]
        if len(ts) == 0:
            raise ValueError(f"t0={t0} is below the first DDIM timestep; raise t0 or the step count")
        g = torch.Generator(device=image.device).manual_seed(seed)
        c = modality.long()
        recon = torch.zeros_like(image)
        for _ in range(draws):
            noise = torch.randn(image.shape, generator=g, device=image.device, dtype=image.dtype)
            x = sched.add_noise(image, noise, ts[:1].repeat(image.shape[0]))
            for t in ts:
                x = sched.step(self._model_out(net, x, t, c, guidance_scale), t, x).prev_sample
            recon += x.clamp(-1, 1)
        recon /= draws
        return recon, (image - recon).abs()

    @torch.no_grad()
    def anomaly_features(self, image, modality, t0s: Sequence[int] = (100, 200, 300),
                         num_inference_steps: int = 50, draws: int = 2, seed: int = 0) -> torch.Tensor:
        """Residual statistics per t0 over the stroke region (ink of input OR reconstruction):
        [mean, q90, q99, global mean] -> (B, 4*len(t0s))."""
        out = []
        for t0 in t0s:
            recon, res = self.anomaly_maps(image, modality, t0, num_inference_steps, draws, seed)
            region = dilate(torch.maximum(ink_mask(image), ink_mask(recon)), 1).bool()
            stats = []
            for r, m in zip(res.flatten(1), region.flatten(1)):
                v = r[m] if m.any() else r
                stats.append(torch.stack([v.mean(), torch.quantile(v, 0.9), torch.quantile(v, 0.99), r.mean()]))
            out.append(torch.stack(stats))
        return torch.cat(out, dim=1)

    @torch.no_grad()
    def preview(self, batch):
        x, c = self.check_batch(batch)
        n = min(8, x.shape[0])
        x, c = x[:n], c[:n]
        recon, res = self.anomaly_maps(x, c, t0=200, num_inference_steps=25)
        samples = self.sample(torch.arange(n, device=x.device) % NUM_MODALITIES, num_inference_steps=25, seed=0)
        return {"input": x, "denoised_t200": recon, "residual_x2": (res * 2 - 1).clamp(-1, 1), "samples": samples}
