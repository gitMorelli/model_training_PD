"""Template: connect seqxai to YOUR model and dataset.

Run:   python -m seqxai --setup examples/setup_template.py --predictions val_preds.csv --out xai_out
or from Python:   see the bottom of this file.

Fill in the three TODO blocks. Everything else is optional.
"""
import numpy as np
import torch

from seqxai import Config, Pipeline, SelectionRule, SequenceSample, TorchDatasetAdapter

# ---------------------------------------------------------------- settings
CONFIG = dict(
    # columns of your predictions CSV
    id_col='subject_id',          # TODO
    label_col='label',            # TODO
    prob_cols='prob',             # binary: P(class 1); multiclass: ['p0', 'p1', 'p2']
    threshold=0.5,
    target_layer=None,            # e.g. 'layer4' for a torchvision ResNet; None = last Conv2d
    pixel_baseline='blur',        # or a float: the normalized value of a blank background
    selection=dict(n_per_cell=8, n_random=20, n_clusters=8, stratify_by=[]),
    num_workers=4,
    # metadata_csv='subjects_meta.csv',   # site / device / age... decodability = shortcut check
    # concepts_csv='concepts.csv',        # seq_id, slot, concept, value [, crop]
)


# ---------------------------------------------------------------- model
def build_model():
    # TODO: build exactly as in training and load the checkpoint
    from my_project.models import ModularModel, build_vision_model   # noqa
    model = ModularModel(build_vision_model(), feat_dim=512, n_slots=10, n_classes=2,
                         d_model=256, seq_model='gru', view_agg='hierarchical',
                         view_groups={'modA': 2, 'modB': 1, 'modC': 1}, n_global=2)
    state = torch.load('checkpoints/best.pt', map_location='cpu')
    model.load_state_dict(state['model'] if 'model' in state else state)
    return model.eval()


# ---------------------------------------------------------------- data
def build_adapter():
    # TODO: your validation Dataset with EVAL transforms (no random crops/flips/jitter,
    # deterministic frame sampling)
    from my_project.data import SubjectDataset   # noqa
    ds = SubjectDataset(split='val', train=False)

    def to_sample(item, seq_id):
        # TODO: map whatever ds[i] returns to the per-subject tensors the model consumes
        return SequenceSample(
            seq_id=seq_id,
            frames=item['frames'],                 # (T, k, C, H, W), crops of each modality
                                                   # concatenated along k in view_groups order
            slot_ids=item['slot_ids'],             # (T,)
            label=int(item['label']),
            view_mask=item.get('view_mask'),       # (T, k) bool, False = imputed/missing crop
            local_properties=item.get('local_properties'),   # (T, n_local) or None
            global_properties=item.get('global_properties'), # (n_global,) or None
            meta={'site': item.get('site')},       # scalars here become metadata columns
        )

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    def to_display(img):                           # undo YOUR normalization for plots
        x = img.detach().cpu().numpy().transpose(1, 2, 0) * std + mean
        return np.clip(x, 0, 1) if x.shape[-1] == 3 else np.clip(x[..., 0], 0, 1)

    return TorchDatasetAdapter(ds, id_fn=lambda i: ds.subject_ids[i], to_sample=to_sample,
                               to_display=to_display)


if __name__ == '__main__':
    cfg = dict(CONFIG)
    cfg['selection'] = SelectionRule(**cfg['selection'])
    pipe = Pipeline(build_model(), build_adapter(), 'val_preds.csv', Config(out_dir='xai_out', **cfg))
    pipe.run()                        # or pipe.run(['cache', 'select', 'temporal'])
