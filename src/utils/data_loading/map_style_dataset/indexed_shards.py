"""Map-style (random access) dataset over WebDataset tar shards, via a byte-offset index.

The index is the CSV written by get_shard_index():
    unique_id, ext, shard, offset, size        (shard relative to `root`)

For a requested id, the dataset reads that id's files with open -> seek -> read and
rebuilds the raw sample exactly as WebDataset's tar reader would:

    {'__key__': id, '__url__': shard path, '<ext>': bytes, ...}   (ext lowercased)

then passes it through your compose function (the callable you give to
`WebDataset(...).compose(...)`) as a one-sample stream. So `ds[i]` returns exactly what
your streaming pipeline yields for that subject before batching, and

    DataLoader(ds, batch_size=B, collate_fn=your_collate, num_workers=W)

reproduces your batched loader, with random access by id on top.

Safety checks
  * stale index: before each file is read, the 512-byte tar header right before its
    data is parsed and its size (and name, when not truncated) compared with the index.
    A rewritten shard raises StaleIndexError instead of returning wrong bytes.
  * ids whose files sit in more than one shard, or with duplicated extensions: WebDataset
    would have produced several separate samples for them, so by default they raise.
  * subjects your compose drops (exclusion set, label -1, build_sequence -> None) raise
    SampleDropped; use `scan()` to list them in advance.
Only for UNCOMPRESSED .tar shards.
"""
from __future__ import annotations

import os
import tarfile
from collections import OrderedDict
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch


class StaleIndexError(RuntimeError):
    """The shard on disk no longer matches the offset index (rebuild the index)."""


class SampleDropped(KeyError):
    """The compose function yielded nothing for this id."""


# --------------------------------------------------------------------------- index
class ShardIndex:
    """Compact in-memory form of the offset CSV: one row per file, sorted by id, with
    numpy columns (cheap to copy into DataLoader workers)."""

    def __init__(self, df: pd.DataFrame, root: str):
        df = df.sort_values(['unique_id', 'shard', 'offset'], kind='stable').reset_index(drop=True)
        self.root = os.path.abspath(root)
        self.shards = sorted(df['shard'].unique().tolist())
        code = {s: n for n, s in enumerate(self.shards)}
        self.ext = df['ext'].to_numpy(object)
        self.shard_code = df['shard'].map(code).to_numpy(np.int32)
        self.offset = df['offset'].to_numpy(np.int64)
        self.size = df['size'].to_numpy(np.int64)
        uid = df['unique_id'].to_numpy(object)
        if len(uid):
            starts = np.r_[0, np.flatnonzero(uid[1:] != uid[:-1]) + 1]
            stops = np.r_[starts[1:], len(uid)]
        else:
            starts = stops = np.array([], int)
        self.ids = uid[starts].tolist()
        self._span = {u: (int(a), int(b)) for u, a, b in zip(self.ids, starts, stops)}
        n_shards = df.groupby('unique_id')['shard'].nunique()
        self.split_ids = sorted(n_shards.index[n_shards > 1])
        dup = df[df.duplicated(['unique_id', 'ext'], keep=False)]
        self.dup_ids = sorted(dup['unique_id'].unique())

    def __contains__(self, uid):
        return uid in self._span

    def __len__(self):
        return len(self.ids)

    def shard_path(self, code: int) -> str:
        return os.path.join(self.root, self.shards[code])

    def files(self, uid: str):
        """[(ext, absolute shard path, offset, size)] in tar order."""
        a, b = self._span[uid]
        return [(self.ext[r], self.shard_path(self.shard_code[r]), int(self.offset[r]), int(self.size[r]))
                for r in range(a, b)]

    def summary(self) -> str:
        return (f'{len(self.ids)} ids, {len(self.ext)} files, {len(self.shards)} shards; '
                f'{len(self.split_ids)} ids split over >1 shard, {len(self.dup_ids)} ids with duplicated files')


def load_offset_index(csv_path: str, root: str, ids: Optional[Iterable] = None,
                      shard_prefix: Union[None, str, Sequence[str]] = None,
                      lowercase_ext: bool = True) -> ShardIndex:
    """Read the offset CSV.

    root:          the directory the `shard` column is relative to (get_shard_index's root)
    ids:           keep only these ids (e.g. the subjects in your predictions CSV)
    shard_prefix:  keep only shards whose relative path starts with this (e.g. 'val/'),
                   or with any of a list of prefixes
    lowercase_ext: WebDataset lowercases extensions when building samples; keep True so
                   the raw samples match the streaming ones
    """
    df = pd.read_csv(csv_path, dtype={'unique_id': str, 'ext': str, 'shard': str},
                     keep_default_na=False)
    df['offset'] = df['offset'].astype(np.int64)
    df['size'] = df['size'].astype(np.int64)
    if shard_prefix:
        prefixes = (shard_prefix,) if isinstance(shard_prefix, str) else tuple(shard_prefix)
        df = df[df['shard'].str.startswith(prefixes)]
    if ids is not None:
        df = df[df['unique_id'].isin(set(map(str, ids)))]
    if lowercase_ext:
        df = df.assign(ext=df['ext'].str.lower())
    return ShardIndex(df, root)


# --------------------------------------------------------------------------- dataset
class IndexedShardDataset(torch.utils.data.Dataset):
    """
    index:     ShardIndex (load_offset_index)
    process:   your compose function: takes an iterator of raw samples, yields processed
               samples (exactly what you pass to WebDataset(...).compose(...)).
               None -> items are the raw sample dicts.
    ids:       which ids this dataset serves, in this order (default: all in the index)
    id_of:     processed item -> its id. If given, every item is checked against the
               requested id (catches key/subject mismatches)
    on_split:  ids whose files are in several shards: 'error' (default) or 'merge'
    verify_headers: check each file's tar header against the index before reading
    max_open_files: per process; least recently used shard files are closed beyond this
    """

    def __init__(self, index: ShardIndex, process: Optional[Callable] = None,
                 ids: Optional[Sequence] = None, id_of: Optional[Callable] = None,
                 on_split: str = 'error', verify_headers: bool = True, max_open_files: int = 32):
        if on_split not in ('error', 'merge'):
            raise ValueError("on_split must be 'error' or 'merge'")
        self.index = index
        self.process = process
        self.id_of = id_of
        self.verify_headers = verify_headers
        self.max_open_files = max_open_files
        self.ids = list(index.ids) if ids is None else [str(i) for i in ids]
        missing = [i for i in self.ids if i not in index]
        if missing:
            raise KeyError(f'{len(missing)} requested ids are not in the index (wrong shard_prefix, '
                           f'or not in these shards), e.g. {missing[:5]}')
        bad = sorted(set(self.ids) & set(index.dup_ids))
        if bad:
            raise ValueError(f'{len(bad)} ids have the same file (id, ext) more than once, e.g. {bad[:5]}; '
                             f'WebDataset would not have produced one sample for them')
        split = sorted(set(self.ids) & set(index.split_ids))
        if split and on_split == 'error':
            raise ValueError(f"{len(split)} ids have files in more than one shard, e.g. {split[:5]}. "
                             f"WebDataset yields one partial sample per shard for them; pass "
                             f"on_split='merge' to treat them as one subject, or exclude them")
        self._pos = {u: n for n, u in enumerate(self.ids)}
        if len(self._pos) != len(self.ids):
            raise ValueError('duplicated ids in `ids`')
        self._fh: OrderedDict = OrderedDict()
        self._pid = None

    # ------------------------------------------------------------------ files
    def __getstate__(self):                       # never send open files to workers
        d = self.__dict__.copy()
        d['_fh'], d['_pid'] = OrderedDict(), None
        return d

    def _handle(self, path):
        if self._pid != os.getpid():              # new (forked) worker: start with no handles
            self._fh, self._pid = OrderedDict(), os.getpid()
        fh = self._fh.get(path)
        if fh is None:
            fh = self._fh[path] = open(path, 'rb')
            while len(self._fh) > self.max_open_files:
                self._fh.popitem(last=False)[1].close()
        else:
            self._fh.move_to_end(path)
        return fh

    def close(self):
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()

    def _read(self, uid, ext, path, offset, size) -> bytes:
        fh = self._handle(path)
        if self.verify_headers:
            if offset < 512:
                raise StaleIndexError(f'{path}: offset {offset} cannot be a tar member')
            fh.seek(offset - 512)
            hdr = fh.read(512)
            try:
                ti = tarfile.TarInfo.frombuf(hdr, tarfile.ENCODING, 'surrogateescape')
            except tarfile.HeaderError as e:
                raise StaleIndexError(f'{path} @ {offset}: no tar header before the data ({e}); '
                                      f'shard rewritten since indexing? rebuild the index') from None
            name = os.path.basename(ti.name)
            if ti.size != size or (len(ti.name) < 99 and name.lower() != f'{uid}.{ext}'.lower()):
                raise StaleIndexError(f'{path} @ {offset}: header says {ti.name!r} ({ti.size} B), index says '
                                      f'{uid}.{ext} ({size} B); shard rewritten since indexing? rebuild the index')
        fh.seek(offset)
        data = fh.read(size)
        if len(data) != size:
            raise StaleIndexError(f'{path}: short read at {offset} ({len(data)}/{size} B)')
        return data

    # ------------------------------------------------------------------ samples
    def raw_sample(self, uid: str) -> dict:
        """The dict WebDataset's tar reader would yield for this id (before decoding)."""
        files = self.index.files(uid)
        raw = {'__key__': uid, '__url__': files[0][1]}
        for ext, path, offset, size in files:
            raw[ext] = self._read(uid, ext, path, offset, size)
        return raw

    def get(self, uid: str):
        uid = str(uid)
        raw = self.raw_sample(uid)
        if self.process is None:
            return raw
        out = list(self.process(iter([raw])))
        if not out:
            raise SampleDropped(f'{uid}: the compose function yielded nothing (exclusion set, missing '
                                f'label, censoring, modality filter, or build_sequence returned None)')
        if len(out) > 1:
            raise ValueError(f'{uid}: the compose function yielded {len(out)} items for one raw sample')
        item = out[0]
        if self.id_of is not None and str(self.id_of(item)) != uid:
            raise ValueError(f'requested {uid!r} but the processed item says {self.id_of(item)!r}')
        return item

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.get(self.ids[i])

    def position(self, uid: str) -> int:
        return self._pos[str(uid)]

    # ------------------------------------------------------------------ checks
    def scan(self, ids: Optional[Sequence] = None, num_workers: int = 0) -> pd.DataFrame:
        """Process every id once; return a table id -> status ('ok' / 'dropped' / 'error')
        with the error message. Use it to get the ids your compose actually produces."""
        probe = _Probe(self, [str(i) for i in (ids if ids is not None else self.ids)])
        loader = torch.utils.data.DataLoader(probe, batch_size=None, num_workers=num_workers,
                                             collate_fn=_identity)
        return pd.DataFrame(list(loader), columns=['id', 'status', 'message']).set_index('id')


def _identity(x):
    return x


class _Probe(torch.utils.data.Dataset):
    def __init__(self, ds, ids):
        self.ds, self.ids = ds, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        uid = self.ids[i]
        try:
            self.ds.get(uid)
            return uid, 'ok', ''
        except SampleDropped as e:
            return uid, 'dropped', str(e)
        except Exception as e:                      # noqa: BLE001 - report, don't stop the scan
            return uid, 'error', f'{type(e).__name__}: {e}'


# --------------------------------------------------------------------------- comparison
def same(a, b, path='item') -> Optional[str]:
    """None if a and b are identical (recursively: tensors, arrays, dicts, sequences,
    PIL images, bytes, scalars), else a description of the first difference."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return f'{path}: tensor vs {type(b if torch.is_tensor(a) else a).__name__}'
        if a.shape != b.shape or a.dtype != b.dtype:
            return f'{path}: {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}'
        return None if torch.equal(a, b) else f'{path}: values differ (max |d| {(a.double() - b.double()).abs().max():.3g})'
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return None if np.array_equal(np.asarray(a), np.asarray(b)) else f'{path}: arrays differ'
    if hasattr(a, 'tobytes') and hasattr(a, 'size') and hasattr(a, 'mode'):      # PIL image
        return None if (a.mode, a.size, a.tobytes()) == (b.mode, b.size, b.tobytes()) else f'{path}: images differ'
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            return f'{path}: keys {sorted(map(str, a))} vs {sorted(map(str, b)) if isinstance(b, dict) else type(b)}'
        for k in a:
            d = same(a[k], b[k], f'{path}[{k!r}]')
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            return f'{path}: length {len(a)} vs {len(b) if isinstance(b, (list, tuple)) else type(b)}'
        for n, (x, y) in enumerate(zip(a, b)):
            d = same(x, y, f'{path}[{n}]')
            if d:
                return d
        return None
    try:
        return None if a == b else f'{path}: {a!r} vs {b!r}'
    except Exception:                               # noqa: BLE001
        return f'{path}: cannot compare {type(a).__name__}'


def compare_with_webdataset(ds: IndexedShardDataset, shards: Sequence[str], id_of: Callable,
                            max_items: Optional[int] = None, raw: bool = False) -> pd.DataFrame:
    """Stream `shards` with WebDataset (no shuffling) through the same compose function
    and compare every item with ds.get(id). raw=True compares the raw sample dicts
    instead (id_of is then ignored; the key is used). Returns one row per item:
    id, status ('same' / 'DIFFERENT' / 'not in dataset' / 'dataset error'), detail."""
    import webdataset as wds
    pipe = wds.WebDataset(list(shards), shardshuffle=False, empty_check=False)
    if not raw:
        pipe = pipe.compose(ds.process)
    rows = []
    for n, item in enumerate(pipe):
        if max_items is not None and n >= max_items:
            break
        uid = str(item['__key__'] if raw else id_of(item))
        if uid not in ds._pos and uid not in ds.index:
            rows.append((uid, 'not in dataset', ''))
            continue
        try:
            mine = ds.raw_sample(uid) if raw else ds.get(uid)
        except Exception as e:                      # noqa: BLE001
            rows.append((uid, 'dataset error', f'{type(e).__name__}: {e}'))
            continue
        if raw:                                     # where a sample comes from is not content
            item = {k: v for k, v in item.items() if k not in ('__url__', '__local_path__')}
            mine = {k: v for k, v in mine.items() if k != '__url__'}
        d = same(item, mine)
        rows.append((uid, 'same' if d is None else 'DIFFERENT', d or ''))
    return pd.DataFrame(rows, columns=['id', 'status', 'detail'])