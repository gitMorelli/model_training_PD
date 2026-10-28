# Handwriting pre-training for PD classification

Four self-supervised objectives on one shared base, in PyTorch + Lightning
(diffusion via `diffusers`). Each trained model exposes the same
`extract_features(image, modality) -> (B, D)` call, so the downstream PD
classifier treats them identically.

| Model | File | What it learns | Features |
|---|---|---|---|
| Denoising AE | `pdhw/dae.py` | Reconstruct clean image from structurally corrupted input | pooled spatial latent (global + ink-region) |
| Masked AE (ViT) | `pdhw/mae.py` | Reconstruct hidden patches; ink-aware masking | CLS + ink-token mean + all-token mean |
| Conditional VAE | `pdhw/cvae.py` | Decoder gets the modality, so `z` encodes *how* not *what*; an adversary + bounded confusion loss removes modality from `z` | posterior mean `mu` |
| Diffusion | `pdhw/diffusion.py` | v-prediction denoiser, class-conditional, min-SNR weighting | pooled mid/up-block activations at several noise levels; optional anomaly residuals |

## Batch contract

Your DataLoader must yield dicts with

```
image     FloatTensor (B, 1, H, W) in [-1, 1], ink = +1, background = -1   (square, H = W)
modality  LongTensor  (B,)  0 = digit, 1 = checkmark, 2 = text
...       any other keys (subject_id, label, ...) are passed through by feature extraction
```

If your images are dark ink on white paper in [0, 1]: `image = 1 - 2 * x`.

## Install

```
pip install -r requirements.txt
```

## Train

```
python train.py fit --config configs/dae.yaml
python train.py fit --config configs/mae.yaml
python train.py fit --config configs/cvae.yaml
python train.py fit --config configs/diffusion.yaml
```

Plug in your data module by editing the `data:` section of a config:

```yaml
data:
  class_path: my_project.data.HandwritingDataModule
  init_args: {image_size: 64, mode: grid, batch_size: 128}
```

Every config is set up for 32 px single symbols. For grids, change the
size-dependent arguments together:

| Input | DAE / CVAE `channels` | MAE `image_size` / `patch_size` | Diffusion `block_out_channels` / `attn_levels` |
|---|---|---|---|
| 32 px single | `[32, 64, 128]` | 32 / 4 | `[64, 128, 256]` / `[2]` |
| 64 px grid | `[32, 64, 128, 128]` | 64 / 4 | `[64, 128, 256, 256]` / `[3]` |
| 128 px grid | `[32, 64, 128, 128, 128]` | 128 / 8 | `[64, 128, 192, 256, 256]` / `[3, 4]` |

CVAE and MAE also need `image_size` set to the data size.

`precision: bf16-mixed` is the default; use `32-true` on CPU or GPUs without bf16.

Preview grids are written to `<log_dir>/previews/` every 5 epochs. Rows are,
per model: DAE input / corrupted / reconstruction / clean reconstruction;
MAE input / masked / reconstruction; CVAE input / reconstruction / the same `z`
decoded as each modality; diffusion input / partially denoised / residual / samples.

## Extract features

```
python extract_features.py --model_class pdhw.DenoisingAE --ckpt runs/dae/.../last.ckpt \
    --data.class_path my_project.data.HandwritingDataModule --data.init_args.image_size 32 \
    --split predict --out features_dae.pt

# diffusion with residual statistics from partial denoising
python extract_features.py --model_class pdhw.DiffusionPretrainer --ckpt ... --anomaly true ...
```

The output holds sample-level `features`, every passed-through batch key, and,
when batches carry `subject_id`, per-subject `[mean | std]` aggregates in
`subject_features`. Or from Python:

```python
from pdhw import DenoisingAE
from pdhw.features import extract_dataset, aggregate_by_subject
model = DenoisingAE.load_from_checkpoint(ckpt)
out = extract_dataset(model, loader, device="cuda")
subjects, X = aggregate_by_subject(out["features"], out["subject_id"])
```

## Metrics worth watching

* **DAE / CVAE**: `grad` term. If it plateaus while `l1` keeps falling, the
  model is getting shapes right but smoothing the stroke detail.
* **MAE**: `masked_ink_frac` should sit near `mask_ratio_ink`.
* **CVAE**: `adv_acc` should fall toward chance (0.33) and `confusion` toward
  log 3 = 1.10 as `z` sheds modality information; `kl` should not explode; `kl_active_dims` should stay well above zero (no posterior collapse).
* **Diffusion**: `mse` (unweighted by SNR) is the comparable number across runs.

## Design decisions that protect the PD signal

* **No scale normalisation anywhere.** Micrographia is a size effect.
* **No blur, rotation, flips or heavy pixel noise** in the DAE corruption: it
  erases blocks and stroke segments and adds clutter to background only.
* **Foreground-weighted L1 + image-gradient L1** reconstruction loss, so blank
  paper does not dominate and tremor-scale edges are penalised.
* **Nearest-upsample + conv** decoders (no transposed convs), so checkerboard
  artefacts don't look like tremor.
* **GroupNorm, EMA weights, seeded noise**: features are deterministic and
  identical between runs of extraction.

## Smoke test

```
python tests/smoke_test.py            # all models at 32 / 64 / 128 px on synthetic data
```

`pdhw/synthetic.py` generates toy handwriting where synthetic 'PD' subjects
write smaller, with more tremor and lighter pressure. It exists only to test
the pipeline; don't draw conclusions about real data from it.
