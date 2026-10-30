"""Orchestrates every stage. Each stage reads earlier stages' outputs from disk, so you
can re-run one stage (e.g. 'spatial' with another target layer) without the others.

    pipe = Pipeline(model, adapter, 'val_predictions.csv', Config(out_dir='xai_out'))
    pipe.run()                         # all stages
    pipe.run(['temporal', 'report'])   # some stages

Output layout (out_dir/):
    cache/      features, recomputed predictions, embeddings, round-trip check
    selection/  selection.csv + the rule used
    temporal/   order tests, truncation, token importances, per-sequence plots
    modality/   subset metrics, Shapley values, crop occlusion/IG, property ablation
    spatial/    Grad-CAM grids per sequence, per-crop method comparisons, shortcut stats
    features/   top-activating patches, SAE latents, probes, TCAV
    geometry/   token-embedding projections and trajectory statistics
    sanity/     randomization test, deletion/insertion curves
    report.html
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from . import features as feat
from . import geometry as geo
from . import modality as mod
from . import sanity as san
from . import spatial as spa
from . import temporal as tem
from . import viz
from .adapter import DataAdapter
from .cache import FeatureCache, build_cache, load_embeddings
from .predictions import load_predictions
from .runner import ModelRunner, resolve_target
from .selection import SelectionRule, select_sequences

STAGES = ['cache', 'select', 'temporal', 'modality', 'spatial', 'features', 'geometry',
          'sanity', 'report']


@dataclass
class Config:
    out_dir: str = 'xai_out'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 0
    # predictions CSV
    id_col: str = 'id'
    label_col: str = 'label'
    prob_cols: Union[str, list] = 'prob'
    threshold: float = 0.5
    # caching
    num_workers: int = 0
    cache_half: bool = False
    max_tokens_per_call: int = 20_000
    # which sequences
    selection: SelectionRule = field(default_factory=SelectionRule)
    explain_true_class_for_errors: bool = True   # errors: also explain the true class
    # temporal
    n_perm: int = 10
    ig_steps: int = 32
    token_ig_baseline: str = 'mean'              # 'mean' token embedding or 'zero'
    temporal_all_sequences: bool = True          # token importances for every sequence
    # modality
    crop_ig_baseline: str = 'mean'               # 'mean' per-crop-position feature or 'zero'
    # spatial
    target_layer: Optional[str] = None           # dotted name in vision_model; None = last Conv2d
    cam_for_grid: str = 'hirescam'               # 'gradcam' or 'hirescam'
    pixel_baseline: Union[str, float] = 'blur'
    top_tokens: int = 2                          # tokens (by occlusion) that get pixel IG/occlusion
    top_crops_per_token: int = 2
    pixel_ig_steps: int = 32
    pixel_ig_noise_samples: int = 0
    occlusion_patch: Optional[int] = None
    max_tokens_in_grid: int = 12
    pixel_deletion: bool = True
    # features
    topk_K: int = 16
    n_channels: int = 12
    patch_size: Optional[int] = None
    sae_expansion: int = 4
    sae_k: int = 16
    sae_steps: int = 3000
    sae_top_latents: int = 12
    concepts_csv: Optional[str] = None
    tcav_method: str = 'mean_diff'               # or 'lr' (needs clean negatives)
    metadata_csv: Optional[str] = None           # extra per-sequence metadata (index = id)
    # sanity
    randomization_n_seqs: int = 4
    randomization_max_steps: int = 10


class Pipeline:
    def __init__(self, model, adapter: DataAdapter, predictions, cfg: Config = Config()):
        self.cfg = cfg
        self.adapter = adapter
        self.runner = ModelRunner(model, cfg.device, getattr(adapter, 'crop_names', None),
                                  cfg.max_tokens_per_call)
        self.preds = load_predictions(predictions, cfg.id_col, cfg.label_col, cfg.prob_cols,
                                      threshold=cfg.threshold) if predictions is not None else None
        self.out = cfg.out_dir
        os.makedirs(self.out, exist_ok=True)
        self.cache = FeatureCache(os.path.join(self.out, 'cache'))
        self._items = None
        torch.manual_seed(cfg.seed)

    # ---------------------------------------------------------------- helpers
    def d(self, *p):
        path = os.path.join(self.out, *p)
        os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
        return path

    def log(self, *a):
        print('[seqxai]', *a, flush=True)

    @property
    def ids(self):
        return self.cache.ids()

    def items(self):
        if self._items is None:
            self._items = {i: it.to(self.runner.device) for i, it in
                           zip(self.ids, self.cache.load_many(self.ids))}
        return self._items

    def selection(self):
        return pd.read_csv(self.d('selection', 'selection.csv'), dtype={'seq_id': str}).set_index('seq_id')

    def recomputed(self):
        return pd.read_csv(self.d('cache', 'recomputed.csv'), dtype={'seq_id': str}).set_index('seq_id')

    def targets_for(self, sid):
        """[(tag, target class)]: predicted class, plus the true class for errors."""
        it = self.items()[sid]
        pred = resolve_target(self.runner, it, 'pred')
        out = [('pred', pred)]
        if self.cfg.explain_true_class_for_errors and it.label is not None and int(it.label) != pred:
            out.append(('true', int(it.label)))
        return out

    def meta(self):
        rc = self.recomputed()
        m = rc[[c for c in rc.columns if c.startswith('meta_')]].rename(columns=lambda c: c[5:])
        if self.cfg.metadata_csv:
            extra = pd.read_csv(self.cfg.metadata_csv)
            extra[extra.columns[0]] = extra[extra.columns[0]].astype(str)
            m = m.join(extra.set_index(extra.columns[0]), how='left')
        return m

    def run(self, stages: Optional[Sequence[str]] = None):
        for s in stages or STAGES:
            if s not in STAGES:
                raise ValueError(f'unknown stage {s}; choose from {STAGES}')
            self.log(f'=== {s} ===')
            if s in ('temporal', 'modality', 'spatial', 'features', 'geometry', 'sanity'):
                shutil.rmtree(os.path.join(self.out, s), ignore_errors=True)   # no stale figures
            getattr(self, f'stage_{s}')()

    # ---------------------------------------------------------------- stages
    def stage_cache(self):
        ids = list(self.preds.index) if self.preds is not None else self.adapter.ids()
        avail = set(self.adapter.ids())
        miss = [i for i in ids if i not in avail]
        if miss:
            self.log(f'{len(miss)} CSV ids not found in the adapter, e.g. {miss[:3]}')
        ids = [i for i in ids if i in avail]
        build_cache(self.runner, self.adapter, self.preds, self.out, ids,
                    num_workers=self.cfg.num_workers, half=self.cfg.cache_half)
        self._items = None

    def _preds_from_cache(self):
        """Use the recomputed predictions (they are what we actually explain)."""
        rc = self.recomputed()
        pc = [f'rprob_{c}' for c in range(self.runner.n_classes)]
        df = pd.DataFrame({'id': rc.index, 'label': rc['label'].astype(int),
                           **({'prob': rc['rprob_1']} if self.runner.n_classes == 2
                              else {f'p{c}': rc[f'rprob_{c}'] for c in range(self.runner.n_classes)})})
        cols = 'prob' if self.runner.n_classes == 2 else [f'p{c}' for c in range(self.runner.n_classes)]
        return load_predictions(df, 'id', 'label', cols, threshold=self.cfg.threshold)

    def stage_select(self):
        preds = self._preds_from_cache()
        emb = load_embeddings(self.out)
        man = select_sequences(preds, self.cfg.selection, emb, self.meta(), self.d('selection'))
        self.log(f'selected {len(man)} sequences:', man['cells'].str.split('|').explode().value_counts().to_dict())

    def stage_temporal(self):
        items = self.items()
        ids = self.ids
        sel = self.selection()
        L = [items[i] for i in ids]
        tg = [resolve_target(self.runner, it, 'pred') for it in L]
        # dataset level
        od = tem.order_tests(self.runner, L, tg, self.cfg.n_perm, self.cfg.seed)
        od.to_csv(self.d('temporal', 'order_tests.csv'), index=False)
        if len(od):
            viz.plot_order_tests(od, self.d('temporal', 'order_tests.png'))
            summ = od.groupby('kind').agg(mean_abs_dprob=('d_prob', lambda x: x.abs().mean()),
                                          flip_rate=('flipped', 'mean'), n=('d_prob', 'size'))
            summ.to_csv(self.d('temporal', 'order_tests_summary.csv'))
        tr = tem.truncation_curves(self.runner, L)
        tr.to_csv(self.d('temporal', 'truncation.csv'), index=False)
        tm = tem.truncation_metrics(tr, self.runner.n_classes)
        tm.to_csv(self.d('temporal', 'truncation_metrics.csv'), index=False)
        viz.plot_truncation(tm, self.d('temporal', 'truncation.png'))

        emb = load_embeddings(self.out)
        base = emb['q'].mean(0) if self.cfg.token_ig_baseline == 'mean' else None
        todo = ids if self.cfg.temporal_all_sequences else list(sel.index)
        allrows = []
        for sid in todo:
            tags = self.targets_for(sid) if sid in sel.index else [('pred', resolve_target(self.runner, items[sid]))]
            for tag, t in tags:
                td = tem.explain_tokens(self.runner, items[sid], t, base, self.cfg.ig_steps)
                td['target_tag'] = tag
                allrows.append(td)
                if sid in sel.index:
                    viz.plot_token_methods(td, f'{sid} | target={t} ({tag}) | {sel.loc[sid, "cells"]}',
                                           self.d('temporal', 'sequences', f'{sid}_{tag}.png'))
        tok = pd.concat(allrows, ignore_index=True)
        tok.to_csv(self.d('temporal', 'token_importance.csv'), index=False)
        # agreement between methods (is attention a usable explanation here?)
        agr = []
        for (sid, tag), g in tok.groupby(['seq_id', 'target_tag']):
            if len(g) < 3:
                continue
            r = dict(seq_id=sid, target_tag=tag)
            for c in ['ig_signed', 'occl_mask', 'attention']:
                if g[c].notna().all() and g[c].std() > 0 and g['occl_remove'].std() > 0:
                    r[f'spearman_occl_vs_{c}'] = spearmanr(g['occl_remove'], g[c])[0]
            agr.append(r)
        agr = pd.DataFrame(agr)
        agr.to_csv(self.d('temporal', 'method_agreement.csv'), index=False)
        # where in time does evidence sit, per class
        p = tok[tok.target_tag == 'pred'].merge(self.recomputed()[['label']], left_on='seq_id', right_index=True)
        p.groupby(['label', 'slot'])['occl_remove'].agg(['mean', 'sem', 'size']).to_csv(
            self.d('temporal', 'importance_by_slot.csv'))
        self.log('temporal done; method agreement (median):',
                 agr.drop(columns=['seq_id', 'target_tag']).median().round(3).to_dict() if len(agr) else {})

    def stage_modality(self):
        items, ids, sel = self.items(), self.ids, self.selection()
        L = [items[i] for i in ids]
        n_cls = self.runner.n_classes
        if self.runner.modality_aware:
            sm = mod.subset_metrics(self.runner, L)
            sm.to_csv(self.d('modality', 'subset_predictions.csv'), index=False)
            ss = mod.summarize_subsets(sm, n_cls)
            ss.to_csv(self.d('modality', 'subset_metrics.csv'), index=False)
            a = ss[ss.subjects == 'all']
            viz.plot_bar(a, 'subset', 'auc' if a['auc'].notna().any() else 'accuracy',
                         'Performance per modality subset (none = properties only)',
                         self.d('modality', 'subset_metrics.png'), ref=0.5 if n_cls == 2 else None)
            rows, srows = [], []
            for sid in ids:
                for tag, t in (self.targets_for(sid) if sid in sel.index else [('pred', resolve_target(self.runner, items[sid]))]):
                    r = mod.modality_shapley(self.runner, items[sid], t)
                    cell = sel.loc[sid, 'cell'] if sid in sel.index else 'not_selected'
                    for m, v in r['shapley'].items():
                        rows.append(dict(seq_id=sid, target_tag=tag, target=t, cell=cell,
                                         modality=m, shapley=v, v_all=r['v_all'], v_none=r['v_none']))
            sh = pd.DataFrame(rows)
            sh.to_csv(self.d('modality', 'modality_shapley.csv'), index=False)
            p = sh[sh.target_tag == 'pred'].copy()
            p2 = p.copy()
            p2['cell'] = 'all sequences'
            viz.plot_shapley(pd.concat([p2, p[p.cell != 'not_selected']]), self.d('modality', 'modality_shapley.png'))
        # crop level, selected sequences
        k = L[0].k
        names = self.runner.crop_names(k)
        if self.cfg.crop_ig_baseline == 'mean':
            F_ = torch.stack([(it.feats * it.base_mask()[..., None]).sum(0) for it in L]).sum(0)
            cnt = torch.stack([it.base_mask().sum(0) for it in L]).sum(0).clamp(min=1)
            fbase = F_ / cnt[:, None]
        else:
            fbase = None
        crows = []
        for sid in sel.index:
            it = items[sid]
            for tag, t in self.targets_for(sid):
                occ = mod.crop_occlusion(self.runner, it, t)
                ig, gap = mod.crop_ig(self.runner, it, t, fbase, self.cfg.ig_steps)
                va = mod.view_attention(self.runner, it)
                np.savez_compressed(self.d('modality', 'sequences', f'{sid}_{tag}.npz'), occlusion=occ,
                                    ig=ig, view_attention=va if va is not None else np.array([]),
                                    slot_ids=it.slot_ids.cpu().numpy(), crop_names=np.array(names))
                for tt in range(it.T):
                    for j in range(k):
                        if np.isfinite(occ[tt, j]):
                            crows.append(dict(seq_id=sid, target_tag=tag, token=tt,
                                              slot=int(it.slot_ids[tt]), crop=j, crop_name=names[j],
                                              modality=self.runner.crop_modality(k)[j],
                                              occlusion=occ[tt, j], ig=ig[tt, j],
                                              view_attention=va[tt, j] if va is not None else np.nan,
                                              ig_gap=gap))
        cr = pd.DataFrame(crows)
        cr.to_csv(self.d('modality', 'crop_importance.csv'), index=False)
        if len(cr):
            g = cr[cr.target_tag == 'pred'].groupby('crop_name')['occlusion'].agg(
                mean_abs=lambda x: x.abs().mean(), sem=lambda x: x.abs().sem()).reset_index()
            viz.plot_bar(g, 'crop_name', 'mean_abs', 'Mean |crop occlusion effect| per crop position',
                         self.d('modality', 'crop_position_importance.png'), err='sem')
        # tabular properties
        if L[0].local_properties is not None or L[0].global_properties is not None:
            lm = torch.cat([it.local_properties for it in L]).mean(0) if L[0].local_properties is not None else None
            gm = torch.stack([it.global_properties for it in L]).mean(0) if L[0].global_properties is not None else None
            prow = []
            for sid in ids:
                t = resolve_target(self.runner, items[sid])
                r = mod.property_ablation(self.runner, items[sid], t, lm, gm)
                prow.append(dict(seq_id=sid, **r))
            pa = pd.DataFrame(prow)
            pa.to_csv(self.d('modality', 'property_ablation.csv'), index=False)
            s = pa.drop(columns='seq_id').abs().mean().rename('mean_abs_delta').reset_index()
            viz.plot_bar(s, 'index', 'mean_abs_delta', 'Score change when properties are set to the mean',
                         self.d('modality', 'property_ablation.png'))

    def stage_spatial(self):
        items, sel = self.items(), self.selection()
        layer = spa.get_layer(self.runner.model.vision_model, self.cfg.target_layer)
        tok = pd.read_csv(self.d('temporal', 'token_importance.csv'), dtype={'seq_id': str})
        cri = self.d('modality', 'crop_importance.csv')
        crop_imp = pd.read_csv(cri, dtype={'seq_id': str}) if os.path.exists(cri) else None
        k = next(iter(items.values())).k
        names = self.runner.crop_names(k)
        brows, chrows, delrows = [], [], []
        for sid in sel.index:
            it = items[sid]
            frames = self.adapter.load(sid).frames
            for tag, t in self.targets_for(sid):
                res = spa.gradcam_sequence(self.runner, it, frames, t, layer)
                td = tok[(tok.seq_id == sid) & (tok.target_tag == tag)].sort_values('slot')
                p = self.runner.probs(self.runner.logits([it]))[0, t].item()
                title = (f'{sid} | label={it.label} | target={t} ({tag}) p={p:.2f} | '
                         f'{sel.loc[sid, "cells"]} | {self.cfg.cam_for_grid}, shared scale')
                viz.plot_sequence(frames, it, self.adapter.to_display, res[self.cfg.cam_for_grid], td, names,
                                  title, self.d('spatial', 'grids', f'{sid}_{tag}.png'),
                                  self.cfg.max_tokens_in_grid)
                for (tt, j), c in res['channel_contrib'].items():
                    chrows.append(np.r_[t, c])
                # detailed methods on the most important tokens/crops
                top_tok = td.sort_values('occl_remove', ascending=False)['token'].head(self.cfg.top_tokens)
                maps_store = {}
                for tt in top_tok.astype(int):
                    if crop_imp is not None and len(crop_imp):
                        ci = crop_imp[(crop_imp.seq_id == sid) & (crop_imp.target_tag == tag) & (crop_imp.token == tt)]
                        crops = ci.sort_values('occlusion', ascending=False)['crop'].head(self.cfg.top_crops_per_token).astype(int).tolist()
                    else:
                        crops = it.base_mask()[tt].nonzero().flatten().tolist()[:self.cfg.top_crops_per_token]
                    for j in crops:
                        ig = spa.pixel_ig(self.runner, it, frames, tt, j, t, self.cfg.pixel_baseline,
                                          self.cfg.pixel_ig_steps, noise_samples=self.cfg.pixel_ig_noise_samples)
                        oc = spa.patch_occlusion(self.runner, it, frames, tt, j, t, self.cfg.occlusion_patch,
                                                 fill=self.cfg.pixel_baseline)
                        H, W = frames.shape[-2:]
                        maps = {'gradcam': spa.upsample(res['gradcam'][(tt, j)], (H, W)),
                                'hirescam': spa.upsample(res['hirescam'][(tt, j)], (H, W)),
                                'integrated_gradients': ig['map'], 'occlusion': oc}
                        maps_store[f'{tt}_{j}'] = maps
                        viz.plot_crop_methods(self.adapter.to_display(frames[tt, j]), maps,
                                              f'{sid} slot {int(it.slot_ids[tt])} {names[j]} | target={t} ({tag}) | '
                                              f'IG gap {ig["completeness_gap"]:.3f}',
                                              self.d('spatial', 'crops', f'{sid}_{tag}_t{tt}_c{j}.png'))
                        for mname, m in maps.items():
                            brows.append(dict(seq_id=sid, target_tag=tag, token=tt, crop=j, method=mname,
                                              ig_gap=ig['completeness_gap'], **san.border_mass(m)))
                        if self.cfg.pixel_deletion:
                            dd = san.pixel_deletion(self.runner, it, frames, tt, j, t, maps,
                                                    self.cfg.pixel_baseline, seed=self.cfg.seed)
                            dd['target_tag'] = tag
                            delrows.append(dd)
                np.savez_compressed(self.d('spatial', 'maps', f'{sid}_{tag}.npz'),
                                    gradcam=np.array([res['gradcam'][c] for c in sorted(res['gradcam'])]),
                                    hirescam=np.array([res['hirescam'][c] for c in sorted(res['hirescam'])]),
                                    crops=np.array(sorted(res['gradcam'])),
                                    **{f'{m}__{key}': v.astype(np.float16) for key, mm in maps_store.items()
                                       for m, v in mm.items()})
        pd.DataFrame(brows).to_csv(self.d('spatial', 'border_mass.csv'), index=False)
        if delrows:
            dl = pd.concat(delrows, ignore_index=True)
            dl.to_csv(self.d('spatial', 'pixel_deletion.csv'), index=False)
            viz.plot_deletion(dl, 'Pixel deletion / insertion (target probability)',
                              self.d('spatial', 'pixel_deletion.png'))
        if chrows:
            C = np.stack(chrows)
            rows = []
            for t in np.unique(C[:, 0]):
                X = C[C[:, 0] == t, 1:]
                rows.append(pd.DataFrame(dict(target=int(t), channel=np.arange(X.shape[1]),
                                              mean_contrib=X.mean(0), mean_abs_contrib=np.abs(X).mean(0))))
            pd.concat(rows).to_csv(self.d('spatial', 'channel_importance.csv'), index=False)

    def stage_features(self):
        cfg, items, ids = self.cfg, self.items(), self.ids
        layer = spa.get_layer(self.runner.model.vision_model, cfg.target_layer)
        k = next(iter(items.values())).k
        names = self.runner.crop_names(k)
        cmod = self.runner.crop_modality(k)
        # --- top activating patches
        ta = feat.top_activating(self.runner, self.adapter, ids, layer, cfg.topk_K)
        np.savez_compressed(self.d('features', 'top_activating.npz'), values=ta['values'], refs=ta['refs'],
                            mean_act=ta['mean_act'], ids=np.array(ids, dtype=object))
        chp = self.d('spatial', 'channel_importance.csv')
        if os.path.exists(chp):
            ch = pd.read_csv(chp).groupby('channel')['mean_abs_contrib'].mean().sort_values(ascending=False)
            chosen = ch.index[:cfg.n_channels].tolist()
            why = 'largest mean |contribution| to explained predictions'
        else:
            chosen = np.argsort(-ta['mean_act'])[:cfg.n_channels].tolist()
            why = 'largest mean activation'
        need = {}
        for c in chosen:
            for r in ta['refs'][c]:
                need.setdefault(int(r[0]), []).append((c, r))
        crops = {c: [] for c in chosen}
        for si, lst in need.items():
            fr = self.adapter.load(ids[si]).frames
            H, W = fr.shape[-2:]
            for c, (s_, t, j, y, x) in lst:
                y0, x0, p = feat.patch_box(y, x, ta['act_hw'], (H, W), cfg.patch_size)
                img = self.adapter.to_display(fr[t, j][:, y0:y0 + p, x0:x0 + p])
                crops[c].append((img, f'{ids[si][:10]} s{int(items[ids[si]].slot_ids[t])} {names[j]}'))
        for c in chosen:
            viz.plot_top_activating(crops[c], f'channel {c} ({why})',
                                    self.d('features', 'top_activating', f'channel_{c:04d}.png'))
        # --- SAE on CNN crop features
        X, G, lab, mods, refs = [], [], [], [], []
        tgt_mode = 1 if self.runner.n_classes == 2 else 'pred'
        for si, sid in enumerate(ids):
            it = items[sid]
            t = resolve_target(self.runner, it, tgt_mode)
            g, _ = self.runner.feature_grad(it, t)
            m = it.base_mask()
            X.append(it.feats[m].cpu()), G.append(g[m].cpu())
            nz = m.nonzero().cpu().numpy()
            lab += [it.label] * len(nz)
            mods += [cmod[j] for j in nz[:, 1]]
            refs += [(si, tt, j) for tt, j in nz]
        X, G = torch.cat(X), torch.cat(G)
        nl = cfg.sae_expansion * X.shape[1]
        sae = feat.SparseAutoencoder(X.shape[1], nl, min(cfg.sae_k, nl)).fit(
            X, cfg.sae_steps, seed=cfg.seed, verbose=False)
        st = sae.stats(X)
        Z = st['Z']
        A = feat.sae_attribution(sae, Z, G).cpu()
        lab = np.asarray(lab, float)
        rows = []
        for l in range(nl):
            z = Z[:, l].numpy()
            fire = z > 0
            if fire.sum() == 0:
                continue
            r = dict(latent=l, freq=fire.mean(), mean_act=z[fire].mean(),
                     mean_attr=float(A[:, l].mean()), mean_abs_attr=float(A[:, l].abs().mean()),
                     label_corr=np.corrcoef(z, lab)[0, 1] if np.std(lab) > 0 and np.std(z) > 0 else np.nan)
            for mname in sorted(set(mods)):
                r[f'frac_{mname}'] = float((np.asarray(mods)[fire] == mname).mean())
            rows.append(r)
        lt = pd.DataFrame(rows).sort_values('mean_abs_attr', ascending=False)
        lt.to_csv(self.d('features', 'sae_latents.csv'), index=False)
        with open(self.d('features', 'sae_stats.json'), 'w') as fh:
            json.dump(dict(explained_variance=st['explained_variance'], dead_frac=st['dead_frac'],
                           n_latents=nl, k=sae.k, n_crops=len(X),
                           attribution_target='class 1 log-odds' if tgt_mode == 1 else 'predicted class'), fh, indent=2)
        torch.save(sae.state_dict(), self.d('features', 'sae.pt'))
        for l in lt['latent'].head(cfg.sae_top_latents):
            top = Z[:, l].topk(min(12, len(Z))).indices.tolist()
            pics = []
            for n in top:
                si, tt, j = refs[n]
                fr = self.adapter.load(ids[si]).frames
                pics.append((self.adapter.to_display(fr[tt, j]), f'{ids[si][:10]} {names[j]} y={int(lab[n])}'))
            rr = lt[lt.latent == l].iloc[0]
            viz.plot_top_activating(pics, f'SAE latent {l}: attr {rr.mean_attr:+.3f}, freq {rr.freq:.2%}, '
                                    f'label corr {rr.label_corr:+.2f}', self.d('features', 'sae', f'latent_{l:05d}.png'), ncol=6)
        # --- probes on metadata (shortcut risk) and concepts
        emb = load_embeddings(self.out)
        meta = self.meta()
        if meta.shape[1]:
            pm = feat.probe_metadata(emb, meta)
            pm.to_csv(self.d('features', 'metadata_probes.csv'), index=False)
        if cfg.concepts_csv:
            cc = pd.read_csv(cfg.concepts_csv, dtype={'seq_id': str})
            pc = feat.probe_concepts(cc, self.cache, emb, names)
            pc.to_csv(self.d('features', 'concept_probes.csv'), index=False)
            crop_level = cc[cc['crop'].notna()] if 'crop' in cc else cc.iloc[:0]
            if len(crop_level):
                # Each concept is tested on the crop positions where it was annotated:
                # negatives/random CAVs come from crops at those positions, and the
                # sensitivity is measured on those positions only. Mixing positions
                # (e.g. a pen crop concept evaluated on scan crops) gives meaningless signs.
                crop_pos = np.array([r[2] for r in refs])
                res = []
                def crop_feats(rows_):
                    out, jj = [], set()
                    for r in rows_.itertuples():
                        it = items.get(str(r.seq_id))
                        if it is None:
                            continue
                        j = names.index(r.crop) if isinstance(r.crop, str) else int(r.crop)
                        tk = (it.slot_ids == int(r.slot)).nonzero()
                        if len(tk) and bool(it.base_mask()[int(tk[0]), j]):
                            out.append(it.feats[int(tk[0]), j].cpu().numpy())
                            jj.add(j)
                    return out, jj

                for concept, g in crop_level.groupby('concept'):
                    v, pos = crop_feats(g[g.value > 0])
                    nv, _ = crop_feats(g[g.value <= 0])
                    if len(v) < 10:
                        continue
                    cval = (np.stack(v), np.stack(nv)) if len(nv) >= 10 else np.stack(v)
                    filt = np.isin(np.arange(k), sorted(pos))
                    cand = np.where(np.isin(crop_pos, sorted(pos)))[0]
                    pool = X[torch.as_tensor(np.random.default_rng(cfg.seed).permutation(cand)[:5000])].numpy()
                    for t in range(self.runner.n_classes):
                        ev = [it for it in items.values() if it.label == t]
                        if ev:
                            r_ = feat.tcav(self.runner, self.cache, {concept: cval}, pool, ev, t,
                                           crop_filter=filt, seed=cfg.seed,
                                           method=cfg.tcav_method)
                            r_['crops'] = '+'.join(names[j] for j in sorted(pos))
                            res.append(r_)
                if res:
                    pd.concat(res).to_csv(self.d('features', 'tcav.csv'), index=False)

    def stage_geometry(self):
        emb = load_embeddings(self.out)
        preds = self._preds_from_cache()
        sel = self.selection()
        tok = geo.token_table(emb, preds)
        tok.to_csv(self.d('geometry', 'token_projection.csv'), index=False)
        hl = sel.index[sel.cells.str.contains('confident')].tolist()[:30]
        viz.plot_projection(tok, hl, self.d('geometry', 'tokens.png'))
        ts = geo.trajectory_stats(tok)
        ts.to_csv(self.d('geometry', 'trajectory_stats.csv'), index=False)
        geo.class_separation(ts).to_csv(self.d('geometry', 'class_separation.csv'), index=False)

    def stage_sanity(self):
        items, sel = self.items(), self.selection()
        tok = pd.read_csv(self.d('temporal', 'token_importance.csv'), dtype={'seq_id': str})
        rows = []
        for sid in sel.index:
            for tag, t in self.targets_for(sid):
                td = tok[(tok.seq_id == sid) & (tok.target_tag == tag)].sort_values('token')
                if len(td) < 2:
                    continue
                rk = {'occlusion': td['occl_remove'].values, 'ig': td['ig_signed'].values}
                if td['attention'].notna().all():
                    rk['attention'] = td['attention'].values
                d = san.temporal_deletion(self.runner, items[sid], t, rk, seed=self.cfg.seed)
                d['target_tag'] = tag
                rows.append(d)
        if rows:
            td = pd.concat(rows, ignore_index=True)
            td.to_csv(self.d('sanity', 'temporal_deletion.csv'), index=False)
            viz.plot_deletion(td, 'Token deletion / insertion (target probability)',
                              self.d('sanity', 'temporal_deletion.png'))
        # cascading randomization on a few confidently-correct sequences
        cand = sel[sel.cells.str.contains('confident_correct')].index.tolist() or sel.index.tolist()
        cand = cand[:self.cfg.randomization_n_seqs]
        its = [items[s] for s in cand]
        frs = [self.adapter.load(s).frames for s in cand]
        tg = [resolve_target(self.runner, it) for it in its]
        rd = san.randomization_test(self.runner, its, frs, tg, self.cfg.target_layer,
                                    max_steps=self.cfg.randomization_max_steps, seed=self.cfg.seed)
        rd.to_csv(self.d('sanity', 'randomization.csv'), index=False)
        viz.plot_randomization(rd, self.d('sanity', 'randomization.png'))

    def stage_report(self):
        from .report import write_report
        p = write_report(self.out, self.cfg)
        self.log('report:', p)
