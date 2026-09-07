#!/usr/bin/env python3
"""
Parallel handwriting-feature extraction over WebDataset-style tar shards.

One process per shard. Each process streams its tar sequentially, groups
members into WebDataset samples (key = filename up to the FIRST dot), reads
the per-sample JSON for grid coordinates, runs extract_image_properties on
each requested variant, and writes one parquet file per shard.

Idempotent: shards whose output already exists are skipped.
"""
import argparse
import hashlib
import io
import json
from linecache import cache
import os
import shutil
import sys
import tarfile
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Single-threaded BLAS: parallelism comes from the process pool. Skipping this
# is a classic 10x slowdown on high-core-count nodes.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# Your module holding extract_image_properties / _binarize / _cell_features / ...
from utils.handwriting_features import extract_image_properties  
from utils.data_loading_utils import return_file_paths, load_grid_dict, grid_lookup

VARIANTS = ("X", "hand", "hand_sentences_full", "number", "number_random")

# whitebg dataset -> ink is DARK on a light background. Verify on one image.
EXTRACT_KWARGS = dict(
    threshold=128,
    ink_is_bright=False,
    use_otsu=False,
    min_ink_pixels=12,
    min_component_area=8,
    compute_slant=True,
    trend_keys=("bbox_height", "ink_area","stroke_width_mean", "fill_ratio"), 
    reductions=("mean", "std", "median", "iqr")
)
exp_params = {
    'use_grid':True,
}
exp_params['list_of_ids_paths'], exp_params['data_folder'], exp_params['grid_dict_path'] = return_file_paths('PD',False,False)

# --------------------------------------------------------------------------
# Grid lookup. Adapt to your actual JSON layout.
# --------------------------------------------------------------------------
def get_grid(key,meta: dict, question: str, variant: str, grid_dict):
    """Return (x_array, y_array) for one question/variant, or None."""
    questionnaire_info = meta.get("questionnaire_info", {})
    rescale_factor = questionnaire_info[str(question)]['rescale_factor']
    num, grid = grid_lookup(grid_dict, key, 'q' + str(question), variant)
    if num>0:
        grid[0, :] = grid[0, :] * rescale_factor[0]
        grid[1, :] = grid[1, :] * rescale_factor[1]

    return num,grid


def flatten(props: dict) -> dict:
    """Make the returned dict parquet-safe: expand tuples, coerce numpy scalars."""
    out = {}
    for k, v in props.items():
        if isinstance(v, (tuple, list)):
            for i, e in enumerate(v):
                out[f"{k}_{i}"] = (float(e) if isinstance(e, (int, float, np.number))
                                   else str(e))
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def process_shard(args) -> dict:
    shard_path, out_dir, scratch_dir, grid_dict = args
    shard_path = Path(shard_path)
    out_path = Path(out_dir) / (shard_path.stem + ".parquet")
    if out_path.exists():
        return {"shard": shard_path.name, "status": "skipped"}

    t0 = time.time()
    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    local = scratch / shard_path.name
    if not local.exists():
        shutil.copyfile(shard_path, local)      # one big sequential read

    rows, cache = [], {}
    stats = dict(computed=0, cached=0, uniform=0, no_grid=0, errors=0)

    def handle(key, bucket):
        """bucket: {'json': bytes, 'q1.X.png': bytes, ...} for one sample."""
        meta = json.loads(bucket["json"]) if "json" in bucket else {}
        for ext, blob in bucket.items():
            if not ext.endswith(".png"):
                continue
            parts = ext[:-4].split(".")           # 'q1.hand' -> ['q1', 'hand']
            if len(parts) != 2:
                continue
            question, variant = parts
            question = question[1:] if question.startswith("q") else question # 'q1' -> '1'
            if VARIANTS and variant not in VARIANTS:
                continue

            num,grid = get_grid(key,meta, question, variant, grid_dict)
            if grid is None:
                stats["no_grid"] += 1
                continue

            # Cache on image bytes + grid. Identical blank PNGs recur constantly
            # here -- every 342-byte file in the listing is the same image.
            ck = hashlib.sha1(blob).digest()
            props = cache.get(ck)
            if props is None:
                try:
                    with Image.open(io.BytesIO(blob)) as im:
                        im.load()
                        arr = np.asarray(im.convert("L"))
                        if arr.max() == arr.min():
                            stats["uniform"] += 1   # blank page: skip the cell loop
                            props = {"is_uniform": True}
                        else:
                            if num>0:
                                x_coords = [0] + sorted(list(grid[0, :])) + [im.width]
                                y_coords = [0] + sorted(list(grid[1, :] )) + [im.height]
                            else:
                                x_coords = [0, im.width]
                                y_coords = [0, im.height]
                            props = extract_image_properties(
                                im, x_coords, y_coords, **EXTRACT_KWARGS)
                    props = flatten(props)
                except Exception:
                    stats["errors"] += 1
                    traceback.print_exc(file=sys.stderr)
                    props = {"error": True}
                cache[ck] = props
                stats["computed"] += 1
            else:
                stats["cached"] += 1
            rows.append({"sample": key, "question": question,
                         "variant": variant, "shard": shard_path.stem, **props})

    try:
        with tarfile.open(local, "r|") as tf:      # "r|" = true streaming, no seeks
            cur, bucket = None, {}
            for m in tf:
                if not m.isfile():
                    continue
                base = os.path.basename(m.name)
                key, _, ext = base.partition(".")  # WebDataset: split on FIRST dot
                if key != cur:
                    if cur is not None:
                        handle(cur, bucket)
                    cur, bucket = key, {}
                bucket[ext] = tf.extractfile(m).read()
            if cur is not None:
                handle(cur, bucket)
    finally:
        local.unlink(missing_ok=True)

    tmp = out_path.with_suffix(".parquet.tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, out_path)                      # atomic; no partial reads

    return {"shard": shard_path.name, "status": "ok", "rows": len(rows),
            **stats, "sec": round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    p.add_argument("--task-id", type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--num-tasks", type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    p.add_argument("--limit", type=int, default=0, help="debug: only N shards")
    a = p.parse_args()

    shards = sorted(Path(a.root, a.split).glob("*.tar"))
    if not shards:
        sys.exit(f"no shards under {Path(a.root, a.split)}")
    mine = shards[a.task_id::a.num_tasks]          # stride balances uneven shards
    if a.limit:
        mine = mine[:a.limit]

    out_dir = Path(a.out) / a.split
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(a.scratch) / f"shards_{a.task_id}"

    print(f"[task {a.task_id}/{a.num_tasks}] {len(mine)}/{len(shards)} shards, "
          f"{a.workers} procs", flush=True)
    
    grid_dict =load_grid_dict(exp_params)  # Load the grid dictionary before processing shards

    jobs = [(str(s), str(out_dir), str(scratch), grid_dict) for s in mine]
    with Pool(processes=min(a.workers, len(jobs))) as pool:
        for i, r in enumerate(pool.imap_unordered(process_shard, jobs, chunksize=1), 1):
            print(f"[{i}/{len(jobs)}] {r}", flush=True)

    shutil.rmtree(scratch, ignore_errors=True)
    print(f"[task {a.task_id}] done", flush=True)


if __name__ == "__main__":
    main()