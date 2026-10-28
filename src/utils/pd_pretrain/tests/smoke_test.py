"""End-to-end smoke test: every model x {32 single, 64 grid, 128 grid}.

Trains a few steps, saves a preview, extracts features, checks shapes,
determinism and checkpoint round-trip.   Run:  python tests/smoke_test.py
"""
import sys
import tempfile
import time
from pathlib import Path

import lightning as L
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdhw import ConditionalVAE, DenoisingAE, DiffusionPretrainer, MaskedAE  # noqa: E402
from pdhw.callbacks import PreviewCallback  # noqa: E402
from pdhw.features import aggregate_by_subject, extract_dataset  # noqa: E402
from pdhw.synthetic import SyntheticDataModule  # noqa: E402

SMALL = dict(warmup_steps=2)
CASES = {
    32: dict(mode="single", grid_n=1),
    64: dict(mode="grid", grid_n=2),
    128: dict(mode="grid", grid_n=4),
}


def make(name, size):
    stages = {32: (16, 32, 64), 64: (16, 32, 64, 64), 128: (16, 32, 64, 64, 64)}[size]
    if name == "dae":
        return DenoisingAE(channels=stages, blocks_per_stage=1, **SMALL)
    if name == "mae":
        return MaskedAE(image_size=size, patch_size=4 if size <= 64 else 8, dim=64, depth=2, heads=2,
                        dec_dim=32, dec_depth=1, dec_heads=2, **SMALL)
    if name == "cvae":
        return ConditionalVAE(image_size=size, channels=stages, blocks_per_stage=1, z_dim=16,
                              kl_warmup_steps=2, confusion_warmup_steps=2, **SMALL)
    if name == "diffusion":
        ch = {32: (32, 32, 64), 64: (32, 32, 64, 64), 128: (32, 32, 64, 64, 64)}[size]
        return DiffusionPretrainer(image_size=size, block_out_channels=ch, layers_per_block=1,
                                   attn_levels=(len(ch) - 1,), norm_num_groups=8, attention_head_dim=16,
                                   feature_draws=2, **SMALL)
    raise ValueError(name)


def run(name, size, tmp):
    L.seed_everything(0, verbose=False)
    dm = SyntheticDataModule(image_size=size, n_subjects=6, samples_per_subject=4, batch_size=8, **CASES[size])
    model = make(name, size)
    trainer = L.Trainer(max_steps=4, max_epochs=1, limit_val_batches=1, accelerator="cpu", logger=False,
                        enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp,
                        callbacks=[PreviewCallback(every_n_epochs=1, num_samples=8)],
                        enable_checkpointing=False, num_sanity_val_steps=0)
    t0 = time.time()
    trainer.fit(model, dm)
    loss = trainer.callback_metrics["val/loss"].item()
    assert torch.isfinite(torch.tensor(loss)), "non-finite val loss"

    out = extract_dataset(model, dm.val_dataloader())
    f = out["features"]
    assert f.ndim == 2 and f.shape[0] == len(dm.val_set) and torch.isfinite(f).all()
    batch = next(iter(dm.val_dataloader()))
    f1 = model.extract_features(batch["image"], batch["modality"])
    f2 = model.extract_features(batch["image"], batch["modality"])
    assert torch.allclose(f1, f2), "features not deterministic"
    subjects, sf = aggregate_by_subject(f, out["subject_id"])

    ckpt = Path(tmp) / f"{name}{size}.ckpt"
    trainer.save_checkpoint(ckpt)
    reloaded = type(model).load_from_checkpoint(ckpt, map_location="cpu")
    assert torch.allclose(reloaded.extract_features(batch["image"], batch["modality"]), f1, atol=1e-5)

    extra = ""
    if name == "diffusion":
        a = model.anomaly_features(batch["image"][:4], batch["modality"][:4], t0s=(100, 200),
                                   num_inference_steps=10, draws=1)
        assert a.shape == (4, 8) and torch.isfinite(a).all()
        extra = f" anomaly={tuple(a.shape)}"
    previews = sorted((Path(tmp) / "previews").glob("*.png"))
    assert previews, "no preview written"
    print(f"OK {name:9s} {size:3d}px  val_loss={loss:.4f}  feat_dim={f.shape[1]:5d}  "
          f"subjects={len(subjects)}x{sf.shape[1]}{extra}  ({time.time() - t0:.1f}s)")
    return previews[-1]


if __name__ == "__main__":
    names = sys.argv[1:] or ["dae", "mae", "cvae", "diffusion"]
    keep = Path(__file__).resolve().parents[1] / "smoke_previews"
    keep.mkdir(exist_ok=True)
    for name in names:
        for size in CASES:
            with tempfile.TemporaryDirectory() as tmp:
                png = run(name, size, tmp)
                (keep / f"{name}_{size}.png").write_bytes(png.read_bytes())
