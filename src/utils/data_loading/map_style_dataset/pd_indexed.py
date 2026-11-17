"""PD validation set as a map-style dataset (random access by subject id), built from
the offset index and the SAME compose / collate functions as prepare_PD_dataset.

    ds = build_pd_dataset(INDEX_CSV, SHARD_ROOT, exp_params, transform, grid_dict,
                          exclusion_set=val_exclusion_set, train_df=train_df,
                          shard_prefix='val/', ids=subject_ids)
    item  = ds.get('A5I5L1N9_0021')     # = what flatten_samples yields for that subject
    batch = pd_collate([item])          # = what your val loader yields (batch of one)
    loader = DataLoader(ds, batch_size=4, collate_fn=pd_collate, num_workers=4)

    adapter = make_seqxai_adapter(ds, pd_collate)     # for the interpretability pipeline

"""
from __future__ import annotations

from functools import partial
from typing import Optional, Sequence

import torch

from .indexed_shards import IndexedShardDataset, compare_with_webdataset, load_offset_index

from src.utils.data_loading.pd_dataloaders import _get_collate, _get_compose, modalities_map
from src.utils.image_processing import get_augmentation_transform

# position of subject_id in the tuple flatten_samples yields:
# (sequence, label, subject_id, questionnaires, modalities, resized_list,
#  global_properties, local_properties, view_mask)
SUBJECT_FIELD = 2


def pd_subject_of(item):
    return item[SUBJECT_FIELD]


# --------------------------------------------------------------------------- processing
def make_pd_compose(exp_params, transform, grid_dict, exclusion_set=frozenset(), train_df=None,
                    augmentation_transform=None, is_val=True, debug=False): 
    """The flatten_samples function prepare_PD_dataset builds, with the validation
    settings. Pass `augmentation_transform` exactly as your validation loader does if
    you want to reproduce its predictions bit for bit; explanations need it to have no
    effect when is_val=True (check with the determinism test in the __main__ block)."""
    modality = exp_params['data_modality']
    if not isinstance(modality, list):
        raise ValueError('prepare_PD_dataset only supports a list of modalities in this branch')
    augmentation_transform = get_augmentation_transform(exp_params)
    base = [m.split('_')[0] for m in modality]
    modality_string = [modalities_map[m] for m in base if m in modalities_map]
    filter_modality = modalities_map.get(exp_params['filter_modality'])
    compose_fn = _get_compose(grouped=exp_params.get('grouped', False),
                              unsupervised_pretrain=exp_params.get('unsupervised_pretrain', False))
    return compose_fn(transform, augmentation_transform,
                      n_views=exp_params['num_tiles'], modality_string_list=modality_string,
                      exclusion_set=exclusion_set,
                      huggingface_transform=exp_params['huggingface_transform'],
                      invert_color=exp_params['invert_color'], grid_dict=grid_dict,
                      censor_time=exp_params['censor_time'], original_modality_names=modality,
                      filter_modality=filter_modality, debug=debug, train_df=train_df,
                      exp_params=exp_params, is_val=is_val)


def make_pd_collate(exp_params, debug=False):
    """Your collate with the same kwargs as prepare_PD_dataset, followed by the numpy ->
    tensor conversion your DataLoader(batch_size=None) applies to every batch."""
    collate_fn = _get_collate(grouped=exp_params.get('grouped', False),
                              unsupervised_pretrain=exp_params.get('unsupervised_pretrain', False))
    kwargs = dict(mod_to_id=exp_params.get('mod_to_id', None),
                  input_range=exp_params.get('unsupervised_kwargs', {}).get('input_range', 'ink_pm1'),
                  check=True)
    return _ConvertedCollate(partial(collate_fn, debug=debug, **kwargs))


class _ConvertedCollate:
    """Picklable wrapper: collate, then torch's default_convert (as DataLoader does)."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, items):
        return torch.utils.data.default_convert(self.fn(items))


# --------------------------------------------------------------------------- dataset
def build_pd_dataset(index_csv, shard_root, exp_params, transform, grid_dict,
                     exclusion_set=frozenset(), train_df=None, shard_prefix='val/',
                     ids: Optional[Sequence[str]] = None, on_split='error', verify_headers=True) -> IndexedShardDataset:
    """ids: the subjects to serve (e.g. the ids in your predictions CSV). Default: every
    id in the selected shards; some of them may be dropped by compose (see ds.scan())."""
    index = load_offset_index(index_csv, shard_root, ids=ids, shard_prefix=shard_prefix)
    print('[pd_indexed]', index.summary())
    process = make_pd_compose(exp_params, transform, grid_dict, exclusion_set, train_df)
    return IndexedShardDataset(index, process, ids=ids, id_of=pd_subject_of,
                               on_split=on_split, verify_headers=verify_headers)


# --------------------------------------------------------------------------- seqxai
def make_seqxai_adapter(ds: IndexedShardDataset, collate, to_display=None, crop_names=None):
    """TorchDatasetAdapter for the interpretability package: each item is collated as a
    batch of one (exactly what the model sees in validation) and unpacked."""
    from src.utils.seqxai.seqxai import SequenceSample, TorchDatasetAdapter

    def to_sample(item, seq_id):
        b = collate([item])                               # SubjectBatch holding one subject
        T = b.frames.shape[0]
        # one subject: seq_ids all 0 and lengths == [T]; seqxai rebuilds both itself
        assert str(b.subject_ids[0]) == seq_id, (b.subject_ids[0], seq_id)
        assert int(torch.as_tensor(b.lengths).reshape(-1)[0]) == T and not torch.as_tensor(b.seq_ids).any()
        gp, lp, vm = b.global_properties, b.local_properties, b.view_mask
        return SequenceSample(
            seq_id=seq_id,
            frames=b.frames,                              # (T, k, C, H, W)
            slot_ids=b.slot_ids.long(),                   # (T,)
            label=int(torch.as_tensor(b.labels).reshape(-1)[0]),
            view_mask=None if vm is None else vm.bool(),  # (T, k)
            local_properties=lp,                          # (T, n_local) or None
            global_properties=None if gp is None else gp[0],   # (B=1, n_global) -> (n_global,)
        )                                                 # resized, modalities: not model inputs

    return TorchDatasetAdapter(ds, id_fn=lambda i: ds.ids[i], to_sample=to_sample,
                               to_display=to_display, crop_names=crop_names)


# --------------------------------------------------------------------------- checks
if __name__ == '__main__':
    import glob
    import os

    from seqxai import check_determinism   # noqa: F401  (if seqxai is installed)

    # TODO: same objects as your validation run
    exp_params, transform, grid_dict, val_exclusion_set, train_df = ..., ..., ..., set(), None
    INDEX_CSV = '/home/a_morelli/datasets/id_lists/ids_to_shards/final_png_whitebg_21_07_26_id_to_shard_offsets.csv'
    SHARD_ROOT = '/mnt/beegfs02/scratch/a_morelli/model_training/shards/PD/final_png_whitebg_21_07_26'

    ds = build_pd_dataset(INDEX_CSV, SHARD_ROOT, exp_params, transform, grid_dict,
                          exclusion_set=val_exclusion_set, train_df=train_df, shard_prefix='val/')

    # 1. which subjects compose keeps (should be the ids of your predictions CSV)
    status = ds.scan(num_workers=8)
    print(status['status'].value_counts())
    status.to_csv('pd_val_scan.csv')

    # 2. identical to the streaming pipeline, raw bytes and processed items
    val_shards = sorted(glob.glob(os.path.join(SHARD_ROOT, 'val', '*.tar')))[:2]
    for raw in (True, False):
        cmp = compare_with_webdataset(ds, val_shards, pd_subject_of, max_items=50, raw=raw)
        print('raw' if raw else 'processed', cmp['status'].value_counts().to_dict())
        print(cmp[cmp.status != 'same'].head())

    # 3. reading a subject twice gives the same tensors (no randomness left)
    ok_ids = status.index[status.status == 'ok'][:10]
    for uid in ok_ids:
        from indexed_shards import same
        d = same(ds.get(uid), ds.get(uid))
        if d:
            print('NOT DETERMINISTIC', uid, d)