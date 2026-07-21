#!/usr/bin/env python3
"""
Visual test harness for SyntheticTransform.

For each progressive transform it takes one initial image and renders it at
n timesteps (t = 0 .. 1), saving a labelled "strip" PNG per transform plus one
combined grid PNG. Built for headless / SLURM runs: forces the Agg matplotlib
backend, opens no windows, writes everything under --outdir.

Examples
--------
    # generated sample image, all transforms, 6 frames each
    python test_synthetic_transform.py --outdir ./synthtest --n 6

    # your own black-on-white image, with the class's default jitter
    python test_synthetic_transform.py --image sample.png --n 6 --jitter 0.15

    # only a couple of transforms
    python test_synthetic_transform.py --transforms progressive_slant progressive_tremor
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")            # must be set before pyplot import (headless)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

# If your module is named differently, change this import.
from src.utils.image_processing import SyntheticTransform, ALL_SYNTHETIC_TRANSFORMS

ALL_TRANSFORMS = ALL_SYNTHETIC_TRANSFORMS  # convenience alias

def make_sample_image(size=(320, 110), text="handwriting"):
    """A black-on-white text image, which is what to_ink() expects."""
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    font = None
    for cand in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(cand):
            font = ImageFont.truetype(cand, 54)
            break
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size[0] - tw) / 2 - bbox[0], (size[1] - th) / 2 - bbox[1]),
              text, fill=0, font=font)
    return img


def seed_all(seed):
    """Seed every RNG the transforms touch (global random + torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transform(name, n_steps, jitter, persona_seed):
    """Assemble the minimal (exp_params, subject_id, train_df) the class needs."""
    exp_params = {"synthetic": ALL_TRANSFORMS}
    subject_id = "subject_0"
    train_df = pd.DataFrame({
        "unique_id": [subject_id],
        "synth_label": [ALL_TRANSFORMS.index(name)],
    })
    return SyntheticTransform(
        exp_params, subject_id, train_df,
        persona_seed=persona_seed, n_steps=n_steps, jitter=jitter,
    )


def render_strip(img, name, n, jitter, persona_seed, base_seed):
    """One transform instance (one persona) called at steps 0..n-1."""
    st = build_transform(name, n_steps=n, jitter=jitter, persona_seed=persona_seed)
    frames = []
    for step in range(n):
        seed_all(base_seed + step)          # reproducible per-frame randomness
        frames.append(st(img.copy(), step))
    return frames


def save_strip(frames, name, n, outdir):
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.7))
    axes = np.atleast_1d(axes)
    denom = max(n - 1, 1)
    for i, (ax, fr) in enumerate(zip(axes, frames)):
        ax.imshow(fr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"t = {i / denom:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(name, fontsize=13)
    fig.tight_layout()
    path = os.path.join(outdir, f"strip_{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def save_grid(all_frames, n, outdir):
    names = list(all_frames)
    fig, axes = plt.subplots(len(names), n,
                             figsize=(2.0 * n, 1.9 * len(names)))
    axes = np.atleast_2d(axes)
    denom = max(n - 1, 1)
    for r, name in enumerate(names):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(all_frames[name][c], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"t = {c / denom:.2f}", fontsize=9)
        # row label (axis is off, so draw text in axes coords)
        axes[r, 0].text(-0.04, 0.5, name, transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, "all_transforms.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    outdir = "/home/a_morelli/vscode_projects/model_training/data/tests"
    image_path = os.path.join(outdir, "digits.png")  # or None to generate a sample image
    n_steps = 6
    jitter = 0
    seed = 1234
    transforms=ALL_TRANSFORMS
    persona_seed = 0

    os.makedirs(outdir, exist_ok=True)

    if image_path:
        img = Image.open(image_path).convert("L")
    else:
        img = make_sample_image()
        img.save(os.path.join(outdir, "input.png"))

    unknown = [t for t in transforms if t not in ALL_TRANSFORMS]
    if unknown:
        raise SystemExit(f"Unknown transform(s): {unknown}\nChoose from: {ALL_TRANSFORMS}")

    all_frames = {}
    for name in transforms:
        frames = render_strip(img, name, n_steps, jitter,
                              persona_seed, seed)
        path = save_strip(frames, name, n_steps, outdir)
        all_frames[name] = frames
        print(f"[ok] {name:26s} -> {path}")

    grid = save_grid(all_frames, n_steps, outdir)
    print(f"[ok] {'grid':26s} -> {grid}")
    print(f"\nDone. {len(transforms)} strips + 1 grid written to {outdir}")



if __name__ == "__main__":
    main()