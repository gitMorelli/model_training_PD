"""Static HTML summary of whatever stages have been run (open out_dir/report.html)."""
from __future__ import annotations

import glob
import html
import json
import os

import pandas as pd

CSS = """
body{font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2328}
h1{font-size:22px} h2{font-size:18px;margin-top:36px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:15px;margin-top:20px}
.note{background:#f6f8fa;border-left:3px solid #8a96a3;padding:8px 12px;margin:10px 0;font-size:13px}
table{border-collapse:collapse;font-size:12px;margin:8px 0} td,th{border:1px solid #ddd;padding:3px 7px;text-align:right}
th{background:#f6f8fa} img{max-width:100%;border:1px solid #eee;margin:6px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}
.warn{color:#b3261e;font-weight:600}
"""


def _tbl(path, n=30, round_=3):
    if not os.path.exists(path):
        return ''
    df = pd.read_csv(path)
    return df.head(n).round(round_).to_html(index=False, border=0)


def _img(out, rel):
    return f'<img src="{html.escape(rel)}">' if os.path.exists(os.path.join(out, rel)) else ''


def _imgs(out, pattern, n=12):
    files = sorted(glob.glob(os.path.join(out, pattern)))[:n]
    if not files:
        return ''
    return '<div class="grid">' + ''.join(
        f'<a href="{os.path.relpath(f, out)}"><img src="{os.path.relpath(f, out)}"></a>' for f in files) + '</div>'


def write_report(out, cfg=None) -> str:
    P = lambda *p: os.path.join(out, *p)
    s = [f'<!doctype html><meta charset="utf-8"><title>Interpretability report</title><style>{CSS}</style>',
         '<h1>Sequence model interpretability report</h1>']

    s.append('<h2>1. Round-trip check</h2>')
    rt = P('cache', 'roundtrip.json')
    if os.path.exists(rt):
        r = json.load(open(rt))
        bad = (r.get('csv_mismatches') or 0) > 0 or (r.get('full_vs_cache_maxdiff') or 0) > 1e-3
        s.append(f'<pre class="{"warn" if bad else ""}">{html.escape(json.dumps(r, indent=2))}</pre>')
        s.append('<div class="note">Every explanation is of the model as re-run here. If these numbers '
                 'are not ~0, the pipeline does not reproduce the CSV (transforms, frame sampling, '
                 'checkpoint or eval mode differ) and explanations would describe a different model.</div>')

    s.append('<h2>2. Selection</h2>')
    rule = P('selection', 'selection_rule.json')
    if os.path.exists(rule):
        s.append(f'<pre>{html.escape(open(rule).read())}</pre>')

    s.append('<h2>3. Time: which slots matter, and does order matter?</h2>')
    s.append(_tbl(P('temporal', 'order_tests_summary.csv')))
    s.append('<div class="note">Shuffle permutes slot positions among tokens; reverse flips time. '
             'If |Δp| and flip rates are tiny, the head behaves like bag-of-frames pooling: '
             'temporal explanations then reduce to "which frames", not "which order".</div>')
    s.append(_img(out, 'temporal/order_tests.png') + _img(out, 'temporal/truncation.png'))
    s.append(_tbl(P('temporal', 'truncation_metrics.csv')))
    s.append('<h3>Agreement between token-importance methods (Spearman vs removal-occlusion)</h3>')
    ma = P('temporal', 'method_agreement.csv')
    if os.path.exists(ma):
        d = pd.read_csv(ma).drop(columns=['seq_id', 'target_tag'], errors='ignore')
        s.append(d.describe().round(3).to_html(border=0))
    s.append('<div class="note">Low agreement between attention and occlusion means attention weights '
             'should not be reported as explanations for this model.</div>')
    s.append(_imgs(out, 'temporal/sequences/*.png', 6))

    s.append('<h2>4. Modalities and crops</h2>')
    s.append(_img(out, 'modality/subset_metrics.png') + _tbl(P('modality', 'subset_metrics.csv'), 60))
    s.append(_img(out, 'modality/modality_shapley.png'))
    s.append('<div class="note">Shapley values split score(all) − score(no images) among modalities, '
             'per sequence. "none" in the subset table is the properties-only model: if it is close '
             'to "all", the images add little beyond the tabular inputs.</div>')
    s.append(_img(out, 'modality/crop_position_importance.png') + _img(out, 'modality/property_ablation.png'))

    s.append('<h2>5. Where in the images</h2>')
    s.append('<div class="note">Grids: rows are time slots (bar = removal-occlusion effect), columns are '
             'crops; heatmaps share one colour scale per sequence. Crop panels compare Grad-CAM, '
             'HiResCAM, Integrated Gradients and patch occlusion on the most important crops.</div>')
    s.append(_imgs(out, 'spatial/grids/*.png', 8))
    s.append(_imgs(out, 'spatial/crops/*.png', 8))
    bm = P('spatial', 'border_mass.csv')
    if os.path.exists(bm):
        d = pd.read_csv(bm)
        if len(d):
            s.append('<h3>Border attribution (shortcut hint)</h3>')
            s.append(d.groupby('method')[['border_share', 'border_area', 'border_ratio']].median().round(3).to_html(border=0))

    s.append('<h2>6. What the features represent</h2>')
    st = P('features', 'sae_stats.json')
    if os.path.exists(st):
        s.append(f'<pre>{html.escape(open(st).read())}</pre>')
    s.append(_tbl(P('features', 'sae_latents.csv'), 15))
    s.append(_imgs(out, 'features/sae/*.png', 6))
    s.append(_imgs(out, 'features/top_activating/*.png', 6))
    s.append('<h3>Metadata decodability (shortcut risk)</h3>' + _tbl(P('features', 'metadata_probes.csv')))
    s.append(_tbl(P('features', 'concept_probes.csv')) + _tbl(P('features', 'tcav.csv')))

    s.append('<h2>7. Embedding geometry</h2>')
    s.append(_img(out, 'geometry/tokens.png') + _tbl(P('geometry', 'class_separation.csv')))
    s.append('<div class="note">Large effect sizes on cx/cy = classes differ by state; on '
             'mean_step/net_disp/dx/dy = they differ by dynamics.</div>')

    s.append('<h2>8. Sanity checks</h2>')
    s.append(_img(out, 'sanity/randomization.png') + _tbl(P('sanity', 'randomization.csv')))
    s.append('<div class="note">Maps that stay correlated with the original after weights are '
             'randomized are not explaining the model.</div>')
    s.append(_img(out, 'sanity/temporal_deletion.png') + _img(out, 'spatial/pixel_deletion.png'))
    s.append('<div class="note">Methods should beat "random": lower deletion AUC, higher insertion AUC.</div>')

    path = P('report.html')
    with open(path, 'w') as fh:
        fh.write('\n'.join(x for x in s if x))
    return path
