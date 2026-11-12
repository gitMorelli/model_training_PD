"""Dataloaders for precomputed representations stored in a table (CSV / parquet / pickle).

One row = one 1-D representation of one subject, at one questionnaire `q` (0..12), from one
modality. Rows are grouped  subject -> tokens (one per q, sorted) -> views (modalities, in
`view_groups` order).

Each subject is emitted as the same 9-tuple the webdataset pipeline yields:
    (sequence, label, subject_id, questionnaires, modalities, resized_list,
     global_properties, local_properties, view_mask)
so `collate_variable_sequences_PD` can be reused as is. The only difference is that
`sequence` is (T, k, D) feature vectors instead of (T, k, C, H, W) images.

On the model side, nothing changes except the vision model, which becomes the identity:
    ModularModel(nn.Identity(), feat_dim=D, n_slots=13, n_classes=...,
                 view_agg='hierarchical', view_groups=view_groups, ...)
`ModularModel._features` flattens (N, k, D) -> (N*k, D), the identity passes it through,
masked views become zeros, and `forward_subsets` / `forward_prefixes` work unchanged.
"""
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .pd_dataloaders import collate_variable_sequences_PD   
from .pd_shared_helpers import questionnaires_to_keep, SubjectBatch


# --------------------------------------------------------------------------- helpers

def parse_rep(x):
    """Table cell -> float32 vector.
    Accepts np.ndarray / list (parquet, pickle) or the strings pandas writes to CSV:
    '[0.1 0.2\\n 0.3]' (numpy repr) or '[0.1, 0.2, 0.3]'."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False).ravel()
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=np.float32)
    s = str(x)
    if '...' in s:
        raise ValueError(
            'rep was truncated by numpy when the CSV was written ("..." in the cell). '
            'Re-save as parquet/pickle, or with np.set_printoptions(threshold=sys.maxsize).')
    return np.array(s.strip().strip('[]').replace(',', ' ').split(), dtype=np.float32)


def read_table(path_or_df):
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df
    p = str(path_or_df)
    if p.endswith('.parquet'):
        return pd.read_parquet(p)
    if p.endswith(('.pkl', '.pickle')):
        return pd.read_pickle(p)
    return pd.read_csv(p)

# --------------------------------------------------------------------------- questionnaire filtering
 
META_KEYS = ('last_avail_q', 'questionnaire_info', 'case_grid_pattern', 'rempli_seulq12')
META_DEFAULTS = dict(last_avail_q=None, questionnaire_info={}, case_grid_pattern=None, rempli_seulq12=1)
 
def meta_getter_from_df(meta_df, id_col='unique_id'):
    """{subject -> metadata dict} lookup from a table with one row per subject and columns
    last_avail_q, questionnaire_info, case_grid_pattern, rempli_seulq12 (the fields the webdataset
    json carried). questionnaire_info stored as a JSON string (CSV) is decoded."""
    import json
    cols = [c for c in META_KEYS if c in meta_df.columns]
    table = meta_df.drop_duplicates(id_col).set_index(id_col)[cols]
    table.index = table.index.astype(str)
    lookup = table.to_dict('index')
 
    def get_meta(sid):
        m = dict(META_DEFAULTS)
        m.update({k: v for k, v in lookup.get(str(sid), {}).items()
                  if not (np.isscalar(v) and pd.isna(v))})
        m['questionnaire_info'] = [] #placeholder, to change if you want to filter by case_dt (eg train only on 5 years before diagnosis)
        return m
    return get_meta

 
def filter_by_questionnaires(df, keep_fn, get_meta, censor_time, train_df=None,
                             subject_col='subject_id', q_col='q', drop_on_error=False,
                             verbose=True):
    """Keep only the rows whose q is among the steps `keep_fn` returns for their subject.
 
    keep_fn          : your questionnaires_to_keep(last_avail_q, censor_time, questionnaire_info,
                       train_df, subject_id, subject_images, case_grid_pattern, rempli_seulq12)
                       -> iterable of q (same indexing as the q column). None or empty drops
                       the subject, like build_sequence returning None in the image pipeline.
    get_meta         : subject_id -> dict with the META_KEYS (e.g. meta_getter_from_df(...)).
    subject_images_fn: builds the `subject_images` argument from the subject's rows; default
                       {q: [modalities]}. Swap it if your function expects another structure.
    """
    keep_pairs, dropped, errors = [], [], []
    subject_images = [q for q in range(14)]
    for sid, g in df.groupby(subject_col, sort=False):
        m = get_meta(sid)
        try:
            qs = keep_fn(m['last_avail_q'], censor_time, m['questionnaire_info'], train_df, sid,
                         subject_images,
                         m['case_grid_pattern'], m['rempli_seulq12'])
        except Exception as e:
            if not drop_on_error:
                raise RuntimeError(f'questionnaires_to_keep failed for subject {sid}') from e
            errors.append((sid, repr(e)))
            continue
        qs = [] if qs is None else [int(q) for q in qs]
        if not qs:
            dropped.append(sid)
            continue
        keep_pairs.extend((sid, q) for q in qs)
 
    keep = pd.MultiIndex.from_tuples(keep_pairs, names=[subject_col, q_col])
    idx = pd.MultiIndex.from_arrays([df[subject_col], df[q_col].astype(int)])
    out = df[idx.isin(keep)].reset_index(drop=True)
 
    if verbose:
        n_subj = df[subject_col].nunique()
        print(f'[filter_by_questionnaires] censor_time={censor_time!r}: '
              f'rows {len(df)} -> {len(out)}, subjects {n_subj} -> {out[subject_col].nunique()} '
              f'({len(dropped)} with no step kept, {len(errors)} errors)')
    return out


# --------------------------------------------------------------------------- dataset

class RepSequenceDataset(Dataset): 
    """
    view_groups : {modality: n_views}, e.g. {'modA': 1, 'modB': 2, 'modC': 1}. Must be the
                  SAME dict (same order) passed to the model. k = sum(n_views).
                  If a (subject, q, modality) cell has fewer rows than n_views the remaining
                  views are zero + view_mask=False; if it has more, `random_pick=True` samples
                  n_views of them each epoch (augmentation), otherwise the first n are taken.
    labels      : optional {subject_id: label}; otherwise read from `label_col` (one value
                  per subject). Subjects with label -1 / missing are dropped.
    """
    def __init__(self, df, view_groups, subject_col='subject_id', label_col='label', labels=None,
                 q_col='q', modality_col='modality', rep_col='rep',
                 exclusion_set=(), random_pick=False, verbose=True):
        self.names = list(view_groups) # ['X', 'digit', 'text']
        self.n_per = [int(view_groups[m]) for m in self.names] # [3,2,1]
        self.offsets = [int(o) for o in np.cumsum([0] + self.n_per)[:-1]] #[0, 3, 4] #starting indexes
        self.k = sum(self.n_per)
        self.random_pick = random_pick
        # modality name of each of the k view positions, e.g. ['modA', 'modB', 'modB', 'modC']
        self.view_names = [m for m, n in zip(self.names, self.n_per) for _ in range(n)]

        unknown = set(df[modality_col].unique()) - set(self.names)
        if unknown and verbose:
            print(f'[RepSequenceDataset] ignoring modalities not in view_groups: {sorted(unknown)}')
        df = df[df[modality_col].isin(self.names)].reset_index(drop=True)

        # parse every representation once and keep them in one contiguous tensor
        reps = [parse_rep(r) for r in df[rep_col]]
        dims = {r.shape[0] for r in reps}
        if len(dims) != 1:
            raise ValueError(f'representations have different lengths: {sorted(dims)}')
        self.reps = torch.from_numpy(np.stack(reps))              # (n_rows, D)
        self.feat_dim = self.reps.shape[1]

        df = df.assign(_row=np.arange(len(df)), #_row is an index that i will use to match back to the representations (which remains in the same order in self.reps)
                       _mod=df[modality_col].map({m: i for i, m in enumerate(self.names)}),
                       _q=df[q_col].astype(int))

        self.items, n_excl, n_unlab = [], 0, 0
        for sid, g in df.groupby(subject_col, sort=True):
            sid = str(sid)
            if sid in exclusion_set:
                n_excl += 1
                continue
            if labels is not None:
                lab = labels.get(sid, -1)
            else:
                vals = g[label_col].dropna().unique()
                if len(vals) > 1:
                    raise ValueError(f'subject {sid} has several labels: {vals}')
                lab = vals[0] if len(vals) else -1
            if pd.isna(lab) or int(lab) == -1:
                n_unlab += 1
                continue
            qs = sorted(g['_q'].unique().tolist())
            cells = {key: torch.tensor(rows.to_numpy(), dtype=torch.long)
                     for key, rows in g.groupby(['_q', '_mod'])['_row']}
            self.items.append((sid, int(lab), qs, cells))

        if verbose:
            print(f'[RepSequenceDataset] {len(self.items)} subjects, k={self.k}, D={self.feat_dim} '
                  f'(excluded {n_excl}, unlabelled {n_unlab})')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, label, qs, cells = self.items[i]
        T = len(qs)
        seq = self.reps.new_zeros(T, self.k, self.feat_dim)
        mask = torch.zeros(T, self.k, dtype=torch.bool)
        for t, q in enumerate(qs):
            for m, (off, n) in enumerate(zip(self.offsets, self.n_per)):
                rows = cells.get((q, m))
                if rows is None:
                    continue
                if len(rows) > n:
                    rows = rows[torch.randperm(len(rows))[:n]] if self.random_pick else rows[:n]
                seq[t, off:off + len(rows)] = self.reps[rows]
                mask[t, off:off + len(rows)] = True

        return (seq,                                   # (T, k, D)
                torch.tensor(label, dtype=torch.long),
                sid,
                qs,                                    # questionnaire index per token -> slot_ids
                [self.view_names] * T,                 # modality name of each view, per token
                [False] * T,                           # resized_list: nothing is resized here
                None,                                  # global_properties
                None,                                  # local_properties
                mask)                                  # view_mask (T, k)


# --------------------------------------------------------------------------- builder

def prepare_rep_dataloaders(table_path, train_df, exp_params, view_groups, batch_size=4, splits=('train', 'val', 'test'),
                            split_col='split', subject_col='subject_id', label_col='label',
                            labels=None, q_col='q', modality_col='modality', rep_col='rep',
                            exclusion_set=(), num_workers=0, partial_batch=True, debug=False,
                            collate_fn=None, seed=None):
    """Returns {split: DataLoader}. `table` is a path (.csv/.parquet/.pkl) or a DataFrame.
    Train is shuffled and samples among extra reps per cell; val/test are deterministic
    and never drop the last partial batch."""
    repr_df = read_table(table_path)

    # filter using the same logic as the image pipeline, so the model sees the same subjects
    get_meta = meta_getter_from_df(train_df, id_col='unique_id')
    df = filter_by_questionnaires(repr_df, questionnaires_to_keep, get_meta, 
                                                exp_params['censor_time'], train_df=train_df)

    
    if collate_fn is None:
        collate_fn = partial(collate_variable_sequences_PD, debug=debug)

    loaders = {}
    for split in splits:
        is_train = split == 'train'
        ds = RepSequenceDataset(
            df[df[split_col] == split], view_groups,
            subject_col=subject_col, label_col=label_col, labels=labels, q_col=q_col,
            modality_col=modality_col, rep_col=rep_col, exclusion_set=set(exclusion_set),
            random_pick=is_train)
        gen = torch.Generator().manual_seed(seed) if (seed is not None and is_train) else None
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=is_train, generator=gen,
            drop_last=is_train and not partial_batch,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0)
    return loaders

def read_rep_dim(table, rep_col='rep', n_check=100):
    """Representation length D, reading only the first `n_check` rows when possible
    (CSV / parquet); a pickle or DataFrame is loaded as is. Raises if those rows disagree."""
    if isinstance(table, pd.DataFrame):
        cells = table[rep_col].head(n_check)
    else:
        p = str(table)
        if p.endswith('.parquet'):
            import pyarrow.parquet as pq
            batch = next(pq.ParquetFile(p).iter_batches(batch_size=n_check, columns=[rep_col]))
            cells = batch.to_pandas()[rep_col]
        elif p.endswith(('.pkl', '.pickle')):
            cells = pd.read_pickle(p)[rep_col].head(n_check)
        else:
            cells = pd.read_csv(p, usecols=[rep_col], nrows=n_check)[rep_col]
    if len(cells) == 0:
        raise ValueError(f'no rows found in {table!r}')
    dims = {parse_rep(c).shape[0] for c in cells}
    if len(dims) != 1:
        raise ValueError(f'representations have different lengths in the first rows: {sorted(dims)}')
    return dims.pop()
# --------------------------------------------------------------------------- usage
#
# view_groups = {'modA': 1, 'modB': 2, 'modC': 1}
# loaders = prepare_rep_dataloaders('reps.csv', view_groups, batch_size=16, num_workers=4)
# D = loaders['train'].dataset.feat_dim
#
# model = ModularModel(nn.Identity(), feat_dim=D, n_slots=13, n_classes=2,
#                      d_model=256, seq_model='gru',
#                      view_agg='hierarchical', view_groups=view_groups,
#                      view_kwargs=dict(intra='attention', inter='concat', modality_dropout=0.3))
#
# The training loop is the same as for images: whatever _collate_subject_level returns
# (frames, seq_ids, slot_ids, lengths, labels, view_mask, ...) is fed to the model as before.