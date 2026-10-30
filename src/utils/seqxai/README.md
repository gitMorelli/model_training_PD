# seqxai — interpretability for the multi-modal CNN + sequence-head model

Built around `ModularModel` / `SequenceModularHead`. It only calls public pieces of
your model (`_features`, `encode_tokens`, `mix`, `forward_prefixes`, `build_view_mask`,
`view_pool.modality_view_mask`), so every analysis is exact with respect to the real
forward pass, and it works with any `seq_model` / `view_agg` combination, per-step
mixers, 1-logit or multi-logit heads, local/global properties and missing crops.

## Install / run

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib   # umap-learn optional
cp examples/setup_template.py my_setup.py                    # fill in 3 TODOs
python -m seqxai --setup my_setup.py --predictions val_preds.csv --out xai_out
# re-run selected stages later, e.g. with another target layer:
python -m seqxai --setup my_setup.py --predictions val_preds.csv --out xai_out --stages spatial,report
```
Then open `xai_out/report.html`.

**Predictions CSV:** one row per sequence (subject): id, true label, and P(class 1)
(binary) or one probability column per class. Column names are set in `CONFIG`.
If an id appears in several rows (one per image), probabilities are averaged per id.

**Adapter:** `examples/setup_template.py` wraps your existing validation `Dataset`.
`load(id)` must return the subject's tensors exactly as the model consumes them:
`frames (T, k, C, H, W)` with eval transforms, `slot_ids (T,)`, `view_mask (T, k)`,
properties. Images are never copied to disk; the adapter stays the single source of
pixels.

## Stages

| stage | on | what it does |
|---|---|---|
| `cache` | all | CNN once per sequence → features, token embeddings `q`, pooled embedding `z`, logits. **Round-trip check** vs the CSV and vs `model.forward` on raw frames. |
| `select` | all | Fixed rule, written before any explanation: per class, confident-correct / confident-wrong / borderline / low-confidence-correct, k-means medoids + outliers of `z`, random controls. Optional stratification by metadata. |
| `temporal` | all + selected | Slot-order tests (shuffle/reverse slot positions), truncation curves (first-j via `forward_prefixes`, last-j), per-token importance: removal occlusion, mask occlusion, IG on `q`, mixer attention, and attention-vs-occlusion agreement. |
| `modality` | all + selected | AUC/accuracy for every modality subset (incl. "none" = properties only), exact per-sequence **Shapley values of modalities**, per-crop occlusion and IG, view attention, property ablation. |
| `spatial` | selected | Grad-CAM + HiResCAM for every crop (sequence grid, one shared colour scale), and on the top tokens/crops: pixel IG and patch occlusion. Border-mass shortcut statistic, channel contributions. |
| `features` | all | Top-activating patches for the most-contributing channels, TopK sparse autoencoder on crop features with latents ranked by attribution to the prediction, metadata decodability probes, concept probes + TCAV (if `concepts_csv`). |
| `geometry` | all | PCA/UMAP of token embeddings, trajectories of selected subjects, state-vs-dynamics effect sizes. |
| `sanity` | selected | Cascading model randomization (Adebayo et al.), token deletion/insertion curves vs random, pixel deletion/insertion. |
| `report` | — | `report.html` with the key tables and figures. |

For misclassified sequences, explanations are computed for the predicted class **and**
the true class (`explain_true_class_for_errors`).

## Design notes

* **Explained quantity** = log-odds of the target class vs the rest,
  `logit_c − logsumexp(others)` (= `logit₁ − logit₀` for 2 logits, ±logit for a 1-output
  head). Not the raw `logit_c`: softmax ignores shifts common to all logits, so its sign
  can disagree with the probability. Not the probability: it saturates. Probabilities
  are used only for curves.
* **Cheap and exact pixel gradients.** The CNN maps crops independently in eval mode,
  so `d score/d pixels = (d score/d feats) · (d feats/d pixels)`: one head backward
  gives the first factor, then the CNN is re-run on crop chunks only. The test suite
  checks this equals the full-model gradient.
* **Occlusion uses the model's own masking** (`view_mask`, and token removal), i.e.
  the same path as `forward(..., modalities=subset)`. "Removal" is like a missing visit;
  masking relies on your modality dropout / imputed-crop training.
* **Baselines matter.** Token IG uses the dataset-mean token, crop IG the mean feature
  per crop position, pixels a heavy blur by default. For handwriting on white paper a
  constant "blank page" value (`pixel_baseline=<normalized background value>`) is often
  more meaningful — try both; disagreement is itself informative.
* **GRU on GPU:** cuDNN refuses RNN backward in eval mode; gradients are taken with
  cuDNN disabled, so dropout stays off.
* Attention weights are reported but always compared with occlusion; don't present
  them alone.

## Optional inputs

* `metadata_csv`: per-sequence metadata (first column = id). Scalars put in
  `SequenceSample.meta` are picked up automatically. Used for stratified selection and
  for decodability probes (site/device decodable from `z` → shortcut risk).
* `concepts_csv`: `seq_id, slot, concept, value[, crop]`. Without `crop`: token-level
  concept, probed on `q`. With `crop` (index or name): crop-level, probed on CNN
  features, and used for TCAV. Rows with `value = 0` are used as the CAV negatives
  (strongly recommended). TCAV p-values become overconfident when CAVs are very stable
  (std ≈ 0): judge the effect size (`tcav_mean` vs `null_mean`) first.

## Using pieces directly

```python
from seqxai import ModelRunner
from seqxai import temporal, modality, spatial
r = ModelRunner(model, 'cuda')
item = r.features(adapter.load('subj001'))          # CNN once
modality.modality_shapley(r, item, target=1)
temporal.explain_tokens(r, item, target=1)
spatial.pixel_ig(r, item, adapter.load('subj001').frames, t=3, j=0, target=1)
```

## Tests

`python tests/run_synthetic.py` trains a small model on a synthetic task with a planted
signal (a square in the 'pen' crops at late slots for class 1; a non-predictive square
in 'scan' crops) using stand-ins for the unshared head components plus your
`SequenceModularHead`/`ModularModel` code verbatim, runs every stage, and checks that
the planted signal is recovered.
