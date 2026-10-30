"""Matplotlib figures for every stage. All functions save a PNG and return its path."""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .spatial import upsample

INK = '#1f2328'
POS, NEG = '#c0392b', '#2e6fbd'


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return path


def _show_img(ax, img):
    ax.imshow(img, cmap='gray' if img.ndim == 2 else None, vmin=0, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])


def plot_sequence(frames, item, display: Callable, cams: dict, token_df: pd.DataFrame,
                  crop_names: Sequence[str], title: str, path: str, max_tokens: int = 12,
                  importance_col: str = 'occl_remove', alpha: float = 0.5) -> str:
    """Rows = tokens in time order (top `max_tokens` by |importance| if longer), columns =
    crops. Heatmaps share ONE colour scale across the whole sequence, so unimportant
    crops look cold. Left panel: per-token importance."""
    td = token_df.copy()
    if len(td) > max_tokens:
        keep = td[importance_col].abs().nlargest(max_tokens).index
        td = td.loc[sorted(keep)]
    T, k = len(td), item.k
    vmax = max([np.max(v) for v in cams.values()] + [1e-12])
    fig = plt.figure(figsize=(1.2 + 1.6 * k, 0.4 + 1.5 * T))
    gs = fig.add_gridspec(T, k + 1, width_ratios=[1.0] + [1.6] * k, wspace=0.05, hspace=0.12)
    imp = td[importance_col].values
    lim = np.nanmax(np.abs(imp)) if np.isfinite(imp).any() else 1
    mask = item.base_mask().cpu().numpy()
    for r, row in enumerate(td.itertuples()):
        t = int(row.token)
        ax = fig.add_subplot(gs[r, 0])
        v = getattr(row, importance_col)
        ax.barh([0], [v if np.isfinite(v) else 0], color=POS if v >= 0 else NEG)
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_yticks([]), ax.tick_params(labelsize=6)
        ax.set_ylabel(f'slot {row.slot}', fontsize=8, rotation=0, ha='right', va='center')
        if r < T - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel(importance_col, fontsize=7)
        for j in range(k):
            ax = fig.add_subplot(gs[r, j + 1])
            if not mask[t, j]:
                ax.set_facecolor('#e6e6e6')
                ax.set_xticks([]), ax.set_yticks([])
                ax.text(0.5, 0.5, 'missing', ha='center', va='center', fontsize=7, color='#777',
                        transform=ax.transAxes)
            else:
                img = display(frames[t, j])
                _show_img(ax, img)
                if (t, j) in cams:
                    ax.imshow(upsample(cams[(t, j)], img.shape[:2]), cmap='inferno',
                              alpha=alpha, vmin=0, vmax=vmax)
            if r == 0:
                ax.set_title(crop_names[j], fontsize=8)
    fig.suptitle(title, fontsize=9, y=1.0)
    return _save(fig, path)


def plot_crop_methods(img: np.ndarray, maps: dict, title: str, path: str) -> str:
    """One crop: input + each attribution map. Signed maps use a diverging scale centred
    on 0 (red supports the target, blue opposes)."""
    n = len(maps) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.6))
    _show_img(axes[0], img)
    axes[0].set_title('input', fontsize=8)
    for ax, (name, m) in zip(axes[1:], maps.items()):
        m = upsample(m, img.shape[:2]) if m.shape != img.shape[:2] else m
        _show_img(ax, img)
        if m.min() < 0:
            lim = np.percentile(np.abs(m), 99.5) + 1e-12
            ax.imshow(m, cmap='RdBu_r', vmin=-lim, vmax=lim, alpha=0.75)
        else:
            ax.imshow(m, cmap='inferno', vmin=0, vmax=m.max() + 1e-12, alpha=0.55)
        ax.set_title(name, fontsize=8)
    fig.suptitle(title, fontsize=9)
    return _save(fig, path)


def plot_token_methods(td: pd.DataFrame, title: str, path: str) -> str:
    cols = [c for c in ['occl_remove', 'occl_mask', 'ig_signed', 'attention'] if c in td and td[c].notna().any()]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.8 * len(cols), 2.4), squeeze=False)
    for ax, c in zip(axes[0], cols):
        v = td[c].values
        ax.bar(range(len(v)), v, color=[POS if x >= 0 else NEG for x in np.nan_to_num(v)])
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(td['slot'].values, fontsize=6)
        ax.set_title(c, fontsize=8), ax.axhline(0, color=INK, lw=0.5)
        ax.set_xlabel('slot', fontsize=7), ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_truncation(m: pd.DataFrame, path: str, metric: str = 'auc') -> str:
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for d, g in m.groupby('direction'):
        ax.plot(g['frac_upto'], g[metric], marker='o', label=f'{d} tokens')
    ax.set_xlabel('fraction of sequence seen'), ax.set_ylabel(metric)
    ax.set_title('Prediction from partial sequences', fontsize=9)
    ax.legend(fontsize=8), ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_order_tests(df: pd.DataFrame, path: str) -> str:
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for kind, g in df.groupby('kind'):
        ax.hist(g['d_prob'], bins=40, alpha=0.6, label=f'{kind} (flip {g.flipped.mean():.1%})')
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel('change in target probability'), ax.set_ylabel('count')
    ax.set_title('Order sensitivity', fontsize=9), ax.legend(fontsize=8)
    return _save(fig, path)


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: str, err: Optional[str] = None,
             ref: Optional[float] = None) -> str:
    fig, ax = plt.subplots(figsize=(max(3.5, 0.55 * len(df) + 1), 3))
    ax.bar(df[x].astype(str), df[y], yerr=df[err] if err else None, color='#5b6b7f')
    if ref is not None:
        ax.axhline(ref, color=INK, ls='--', lw=0.7)
    ax.set_title(title, fontsize=9), ax.set_ylabel(y)
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right', fontsize=7)
    return _save(fig, path)


def plot_shapley(df: pd.DataFrame, path: str) -> str:
    """df: rows seq_id, cell, modality, shapley."""
    mods = list(dict.fromkeys(df['modality']))
    cells = list(dict.fromkeys(df['cell']))
    fig, ax = plt.subplots(figsize=(min(11, 1.2 + 1.3 * len(mods) * max(1, len(cells)) / 2), 3))
    w = 0.8 / max(1, len(cells))
    for ci, c in enumerate(cells):
        for mi, m in enumerate(mods):
            v = df[(df.cell == c) & (df.modality == m)]['shapley'].values
            xs = mi + (ci - (len(cells) - 1) / 2) * w
            ax.boxplot([v], positions=[xs], widths=w * 0.9, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=f'C{ci}', alpha=0.5))
    ax.set_xticks(range(len(mods))), ax.set_xticklabels(mods)
    ax.axhline(0, color=INK, lw=0.6)
    for ci, c in enumerate(cells):
        ax.plot([], [], color=f'C{ci}', lw=6, alpha=0.5, label=c)
    ax.legend(fontsize=7, ncol=2), ax.set_ylabel('Shapley value (log-odds)')
    ax.set_title('Modality contributions by selection cell', fontsize=9)
    return _save(fig, path)


def plot_top_activating(patches: list, title: str, path: str, ncol: int = 8) -> str:
    """patches: list of (image array, caption)."""
    n = len(patches)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.3 * ncol, 1.45 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, (img, cap) in zip(axes.ravel(), patches):
        _show_img(ax, img)
        ax.axis('on'), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(cap, fontsize=5)
    fig.suptitle(title, fontsize=9)
    return _save(fig, path)


def plot_projection(tok: pd.DataFrame, highlight: Sequence[str], path: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = sorted(tok['label'].dropna().unique()) if 'label' in tok else [None]
    for c in labels:
        g = tok[tok.label == c] if c is not None else tok
        axes[0].scatter(g.x, g.y, s=3, alpha=0.35, label=f'label {c}')
    axes[0].legend(fontsize=7, markerscale=3), axes[0].set_title('tokens by label', fontsize=9)
    sc = axes[1].scatter(tok.x, tok.y, c=tok.slot, s=3, alpha=0.5, cmap='viridis')
    plt.colorbar(sc, ax=axes[1], label='slot'), axes[1].set_title('tokens by slot', fontsize=9)
    axes[2].scatter(tok.x, tok.y, s=2, color='#cccccc', alpha=0.3)
    for sid in highlight:
        g = tok[tok.seq_id == sid].sort_values('slot')
        if g.empty:
            continue
        col = f'C{int(g.label.iloc[0])}' if 'label' in g else 'C0'
        axes[2].plot(g.x, g.y, '-', color=col, lw=0.9, alpha=0.8)
        axes[2].scatter(g.x.iloc[-1], g.y.iloc[-1], color=col, s=14, marker='>')
        axes[2].scatter(g.x.iloc[0], g.y.iloc[0], color=col, s=10, marker='o')
    axes[2].set_title('trajectories of selected sequences (o start, > end)', fontsize=9)
    fig.suptitle(tok.attrs.get('projection', ''), fontsize=9)
    return _save(fig, path)


def plot_randomization(df: pd.DataFrame, path: str) -> str:
    fig, ax = plt.subplots(figsize=(max(5, 0.35 * df.stage.nunique() + 2), 3.2))
    for meth, g in df.groupby('method'):
        ax.errorbar(range(len(g)), g['spearman_mean'], yerr=g['spearman_std'], marker='o',
                    label=meth, capsize=2)
        ticks = g['stage'].tolist()
    ax.set_xticks(range(len(ticks))), ax.set_xticklabels(ticks, rotation=60, ha='right', fontsize=6)
    ax.set_ylabel('Spearman vs original |map|'), ax.axhline(0, color=INK, lw=0.5)
    ax.set_title('Cascading randomization (should drop fast)', fontsize=9), ax.legend(fontsize=8)
    return _save(fig, path)


def plot_deletion(df: pd.DataFrame, title: str, path: str) -> str:
    g = df.groupby('method')[['deletion_auc', 'insertion_auc']].agg(['mean', 'sem']).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, c, note in zip(axes, ['deletion_auc', 'insertion_auc'], ['lower is better', 'higher is better']):
        ax.bar(g['method'], g[(c, 'mean')], yerr=g[(c, 'sem')], color='#5b6b7f')
        ax.set_title(f'{c} ({note})', fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return _save(fig, path)
