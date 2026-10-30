"""End-to-end test on the synthetic task (see synthetic.py) + checks that the planted
signal is recovered. Usage: python tests/run_synthetic.py [out_dir]"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import synthetic as syn  # noqa: E402

from seqxai import Config, Pipeline, SelectionRule  # noqa: E402
from seqxai import temporal as tem, modality as mod, spatial as spa  # noqa: E402
from seqxai.runner import ModelRunner  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/seqxai_test'
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

data = syn.make_data(420)
train, val = data[:300], data[300:]
ckpt = os.path.join(OUT, 'model.pt')
model = syn.build('attnpool')
if os.path.exists(ckpt):
    model.load_state_dict(torch.load(ckpt))
    model.eval()
else:
    syn.train(model, train, epochs=30)
    torch.save(model.state_dict(), ckpt)
csv = os.path.join(OUT, 'val_preds.csv')
df = syn.predict_csv(model, val, csv)
acc = ((df.prob_pd > 0.5).astype(int) == df.y_true).mean()
print(f'val accuracy {acc:.3f}')

# crop-level concept labels: 'square' present in a crop
rows = []
for s in val:
    for t, sl in enumerate(s['slots'].tolist()):
        for j in (0, 1, 3):
            if s['mask'][t, j]:
                rows.append(dict(seq_id=s['id'], slot=sl, crop=j,
                                 concept='square_pen' if j < 2 else 'square_scan',
                                 value=int((t, j) in s['boxes'])))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'concepts.csv'), index=False)

cfg = Config(out_dir=os.path.join(OUT, 'xai'), device='cpu', id_col='subject', label_col='y_true',
             prob_cols='prob_pd', selection=SelectionRule(n_per_cell=3, n_random=4, n_clusters=4,
                                                          n_outliers=2),
             sae_steps=800, sae_top_latents=4, n_channels=4, randomization_n_seqs=2,
             concepts_csv=os.path.join(OUT, 'concepts.csv'), pixel_ig_steps=16, ig_steps=16)
pipe = Pipeline(model, syn.SynthAdapter(val), csv, cfg)
t0 = time.time()
for st in ['cache', 'select', 'temporal', 'modality', 'spatial', 'features', 'geometry', 'sanity', 'report']:
    t1 = time.time()
    pipe.run([st])
    print(f'   stage {st}: {time.time() - t1:.1f}s')
print(f'total {time.time() - t0:.1f}s')

# ------------------------------------------------------------------ checks
X = cfg.out_dir
ok = True


def check(name, cond, info=''):
    global ok
    ok &= bool(cond)
    print(f'[{"PASS" if cond else "FAIL"}] {name} {info}')


import json  # noqa: E402
rt = json.load(open(os.path.join(X, 'cache', 'roundtrip.json')))
check('round trip vs CSV', rt['csv_vs_cache_maxdiff'] < 1e-4, rt)
check('round trip full forward', rt['full_vs_cache_maxdiff'] < 1e-4)

sh = pd.read_csv(os.path.join(X, 'modality', 'modality_shapley.csv'))
p = sh[(sh.target_tag == 'pred')].copy()
lab = pd.read_csv(os.path.join(X, 'cache', 'recomputed.csv'), dtype={'seq_id': str}).set_index('seq_id')['label']
p['label'] = p.seq_id.map(lab)
m = p[p.label == 1].groupby('modality')['shapley'].apply(lambda x: x.abs().mean())
check('pen modality dominates (label 1)', m.idxmax() == 'pen', m.round(3).to_dict())

tok = pd.read_csv(os.path.join(X, 'temporal', 'token_importance.csv'))
tok['label'] = tok.seq_id.map(lab)
t1 = tok[(tok.label == 1) & (tok.target == 1) & (tok.target_tag == 'pred')]
late, early = t1[t1.slot >= 5].occl_remove.mean(), t1[t1.slot < 5].occl_remove.mean()
check('late slots matter more for class 1', late > early + 0.1, f'late {late:.3f} early {early:.3f}')
check('token IG completeness', tok.ig_completeness_gap.abs().max() < 0.05,
      f'max gap {tok.ig_completeness_gap.abs().max():.4f}')

# CAM on the square: compare map mass inside the planted box vs. its area
runner = ModelRunner(model, 'cpu')
layer = spa.get_layer(model.vision_model)
ratios, ig_ratios, gaps = [], [], []
for s in val:
    if s['label'] != 1:
        continue
    item = runner.features(syn.SynthAdapter(val).load(s['id']))
    res = spa.gradcam_sequence(runner, item, s['frames'], 1, layer)
    for (t, j), (y, x) in s['boxes'].items():
        if j > 1:
            continue
        cam = spa.upsample(res['hirescam'][(t, j)], (32, 32))
        box = np.zeros((32, 32), bool)
        box[max(0, y - 2):y + 7, max(0, x - 2):x + 7] = True
        ratios.append((cam[box].sum() / (cam.sum() + 1e-9)) / box.mean())
        if len(ig_ratios) < 6:
            ig = spa.pixel_ig(runner, item, s['frames'], t, j, 1, 'blur', 16)
            a = np.clip(ig['map'], 0, None)
            ig_ratios.append((a[box].sum() / (a.sum() + 1e-9)) / box.mean())
            gaps.append(abs(ig['completeness_gap']))
    if len(ratios) > 40:
        break
check('HiResCAM concentrates on the square', np.median(ratios) > 2, f'median enrichment {np.median(ratios):.2f}x')
check('pixel IG concentrates on the square', np.median(ig_ratios) > 2, f'median enrichment {np.median(ig_ratios):.2f}x')
check('pixel IG completeness', max(gaps) < 0.1 * 5, f'max gap {max(gaps):.3f}')

# chained gradient == full-model gradient (exactness of the spatial trick)
s = val[0]
item = runner.features(syn.SynthAdapter(val).load(s['id']))
fr = s['frames'].clone().requires_grad_(True)
b = syn.collate([s])
out = model(fr, b['seq_ids'], b['slot_ids'], b['lengths'], global_properties=b['global_properties'],
            view_mask=b['view_mask'])
(out[0, 1] - out[0, 0]).backward()   # explained score = log-odds
g_full = fr.grad
g_feat, _ = runner.feature_grad(item, 1)
t_, j_ = [tuple(x) for x in item.base_mask().nonzero().tolist()][0]
x = s['frames'][t_, j_][None].clone().requires_grad_(True)
f = runner.vision(x)
(f * g_feat[t_, j_]).sum().backward()
err = (x.grad[0] - g_full[t_, j_]).abs().max().item()
check('chained pixel gradient matches full model', err < 1e-5, f'max abs err {err:.2e}')

rd = pd.read_csv(os.path.join(X, 'sanity', 'randomization.csv'))
last = rd.groupby('method').spearman_mean.last()
check('randomization destroys maps', (last < 0.5).all(), last.round(3).to_dict())

tc = os.path.join(X, 'features', 'tcav.csv')
tcd = pd.read_csv(tc)
print(tcd.round(3).to_string())
r1 = tcd[(tcd.concept == 'square_pen') & (tcd.target == 0)].iloc[0]
check('TCAV: on class-0 subjects, pen square pushes away from class 0',
      r1.tcav_mean < r1.null_mean and r1.p_value < 0.05,
      f'{r1.tcav_mean:.2f} vs null {r1.null_mean:.2f}, p={r1.p_value:.3g}')
r2 = tcd[(tcd.concept == 'square_scan') & (tcd.target == 1)].iloc[0]
print(f'       scan square (distractor): {r2.tcav_mean:.2f} vs null {r2.null_mean:.2f}, p={r2.p_value:.3g}')

# ------------------------------------------------------------------ GRU per-step mixer + 1-logit head
print('\n--- GRU (per-step) mixer, 1-output head: smoke test ---')
g = syn.build('gru', n_out=1).eval()
r = ModelRunner(g, 'cpu')
ad = syn.SynthAdapter(val[:6])
items = [r.features(ad.load(i)) for i in ad.ids()]
for sm, it in zip(val[:6], items):
    b = syn.collate([sm])
    with torch.no_grad():
        o = g(b['frames'], b['seq_ids'], b['slot_ids'], b['lengths'],
              global_properties=b['global_properties'], view_mask=b['view_mask'])[1]
    assert torch.allclose(o, r.logits([it]), atol=1e-5)
pref = r.prefix_logits(items)
assert all(p.shape == (it.T, 1) for p, it in zip(pref, items))
for it in items[:2]:
    td = tem.explain_tokens(r, it, 1, None, 8)
    assert abs(td.ig_completeness_gap.iloc[0]) < 0.05, td.ig_completeness_gap.iloc[0]
    mod.modality_shapley(r, it, 0)
    mod.crop_occlusion(r, it, 1)
    spa.gradcam_sequence(r, it, ad.load(it.seq_id).frames, 1, spa.get_layer(g.vision_model))
tem.order_tests(r, items, [1] * len(items), 3)
tem.truncation_curves(r, items)
print(r.pooled(items).shape)
check('GRU / 1-logit path runs and matches model.forward', True)

print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')
