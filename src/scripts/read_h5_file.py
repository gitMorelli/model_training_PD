#!/usr/bin/env python3
"""
Parallel pre-loading of grid data from an HDF5 file.

Reads a list of target IDs from a single .h5 file using multiple worker
processes (each opens its own read-only handle), then aggregates the
per-worker sub-dictionaries into a single dict in the parent process.

Returned structure:
    {id: {q_name: {class_key: (scalar, array)}}}

Notes / gotchas baked in:
  * h5py handles cannot be shared across processes -> each worker opens
    the file independently.
  * Standard HDF5 serializes calls behind a global lock, so threads do not
    parallelize reads -> we use processes.
  * Worker results are pickled and piped back to the parent. For very large
    arrays this IPC cost can dominate; see the module docstring of
    `get_ids_data_from_h5_file_list` for the file-handoff alternative.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pickle

try:
    import pandas as pd
except ImportError:  # pandas only needed if reading IDs from a CSV
    pd = None

EXPERIMENT = 'pre_training' #'PD' #'handedness'
# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _chunked(seq, n_chunks):
    """Split `seq` into at most `n_chunks` roughly equal, non-empty chunks."""
    seq = list(seq)
    k, m = divmod(len(seq), n_chunks)
    out, start = [], 0
    for i in range(n_chunks):
        size = k + (1 if i < m else 0)
        if size:
            out.append(seq[start:start + size])
            start += size
    return out


def _dict_nbytes(d):
    """Sum of the .nbytes of every array stored in the nested result dict.

    This reflects the real array payload, unlike sys.getsizeof(d) which only
    measures the top-level container.
    """
    total = 0
    for q_dict in d.values():
        for class_dict in q_dict.values():
            for _scalar, arr in class_dict.values():
                if isinstance(arr, np.ndarray):
                    total += arr.nbytes
    return total


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _read_id_chunk(file_path, id_chunk, chunk_idx, progress_every=100):
    """Worker process: open own handle, read a chunk of IDs, return sub-dict.

    Returns (results, missing) where:
        results : {id: {q_name: {class_key: (scalar, array)}}}
        missing : list of IDs not present in the file
    """
    results = {}
    missing = []
    with h5py.File(file_path, "r") as f:          # each process opens independently
        for j, target_id in enumerate(id_chunk):
            if target_id not in f:
                missing.append(target_id)
                continue
            id_grp = f[target_id]
            id_data = {}
            for q_name in id_grp.keys():
                q_grp = id_grp[q_name]
                class_data = {}
                for class_key in q_grp.keys():
                    ds = q_grp[class_key]                  # bind once, avoid double lookup
                    class_data[class_key] = (
                        ds.attrs.get("scalar_value"),
                        ds[()],                            # [()] reads the full array
                    )
                id_data[q_name] = class_data
            results[target_id] = id_data

            if progress_every and j % progress_every == 0:
                print(f"[worker {chunk_idx}] {j + 1}/{len(id_chunk)} IDs", flush=True)
    return results, missing


# --------------------------------------------------------------------------- #
# Parallel aggregation
# --------------------------------------------------------------------------- #
def get_ids_data_from_h5_file_list(file_path, target_ids, n_workers=4):
    """Read data for many IDs from one .h5 file in parallel.

    Returns: {id: {q_name: {class_key: (scalar, array)}}}
    IDs not found in the file are omitted and reported.

    If your arrays are large and the pickle-and-pipe of results back to the
    parent dominates runtime, switch the worker to write its sub-dict to its
    own file (pickle protocol 5 or np.savez) and return the path instead;
    that keeps array bytes off the IPC channel.
    """
    target_ids = [str(i) for i in target_ids]
    if not target_ids:
        return {}

    # Never submit one task per ID: chunk so each worker opens the file once.
    n_workers = max(1, min(n_workers, len(target_ids)))
    chunks = _chunked(target_ids, n_workers)

    full_dict = {}
    all_missing = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(_read_id_chunk, file_path, chunk, i)
            for i, chunk in enumerate(chunks)
        ]
        for fut in as_completed(futures):
            results, missing = fut.result()
            full_dict.update(results)          # disjoint keys -> plain update is safe
            all_missing.extend(missing)

    if all_missing:
        preview = all_missing[:5]
        print(f"{len(all_missing)} IDs not found in {file_path}, e.g. {preview}")
    return full_dict


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def pre_load_grid_data(h5_filepath, csv_data, n_workers=4):
    """Pre-load grid data for every unique subject in `csv_data`.

    `csv_data` is a pandas DataFrame containing an 'ident_projet' column.
    Returns the aggregated nested dict.
    """
    start = time.time()
    if EXPERIMENT == 'handedness':
        unique_subjects = csv_data["ident_projet"].unique()
    else:
        unique_subjects = csv_data['unique_id'].str.split('_').str[0].unique()

    full_dict = get_ids_data_from_h5_file_list(
        h5_filepath, unique_subjects, n_workers=n_workers
    )

    end = time.time()
    print(
        f"Preloaded grid data for {len(unique_subjects)} subjects "
        f"({len(full_dict)} found) in {end - start:.2f} seconds."
    )

    # Real array payload, not sys.getsizeof(full_dict) which only sees the
    # top-level container.
    total_mb = _dict_nbytes(full_dict) / (1024 * 1024)
    print(f"Total array memory in preloaded grid data: {total_mb:.2f} MB")
    return full_dict


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args():
    p = argparse.ArgumentParser(
        description="Parallel pre-load of grid data from an HDF5 file."
    )
    p.add_argument(
        "--n_workers", type=int, default=4, help="Number of worker processes."
    )
    return p.parse_args()


def main():
    args = _parse_args()

    if EXPERIMENT == 'handedness':
        csv_data = pd.read_csv('/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv')
        save_path = f"/mnt/beegfs02/scratch/a_morelli/datasets/rr_data_h5.pkl"
    elif EXPERIMENT == 'PD':
        csv_data = pd.read_parquet('/home/a_morelli/datasets/id_lists/PD_training_set_13_7_26.parquet')
        date = time.strftime("%d_%m_%y")
        save_path = f"/home/a_morelli/datasets/id_lists/h5/PD_data_h5_{date}.pkl"
    elif EXPERIMENT == 'pre_training':
        csv_data = pd.read_parquet('/home/a_morelli/datasets/id_lists/final_table_for_matching_splitted_13_7_26_pre_training.parquet')
        date = time.strftime("%d_%m_%y")
        save_path = f"/home/a_morelli/datasets/id_lists/h5/pre_training_data_h5_{date}.pkl"
    
    data = pre_load_grid_data(
        "/mnt/beegfs01/scratch/a_morelli/extraction/final/results_aggregated/final_aggregated_data.h5", 
        csv_data, n_workers=args.n_workers
    )

    print(f"Loaded {len(data)} IDs.")

    with open(save_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    start = time.time()
    with open(save_path, "rb") as f:
        data = pickle.load(f)
    end = time.time()
    print(f"Data loaded from pickle in {end - start:.2f} seconds.")


if __name__ == "__main__":   # required: ProcessPoolExecutor re-imports this module
    main()