"""
Build a byte-offset index for WebDataset-style tar shards.

One CSV row per file inside the shards:
    unique_id, ext, shard, offset, size

    unique_id = file name up to the first dot   (A5I5L1N9_0021)
    ext       = everything after the first dot   (q5.hand.png)
    shard     = shard path relative to root      (train/worker0_shard-000000.tar)
    offset    = byte offset of the file's data inside the tar
    size      = size of the file's data in bytes

With this, a file can be read with: open(shard) -> seek(offset) -> read(size).
Only works for UNCOMPRESSED .tar shards, and only as long as the shards are
not rewritten (rebuild the index if they are).
"""
import csv
import os
import sys
import tarfile
from collections import defaultdict
from multiprocessing import Pool


# Must be at module level: Pool pickles the function to send it to the
# workers, and functions defined inside another function can't be pickled.
def files_in_shard(tar_path):
    """Return (tar_path, [(id, ext, offset, size), ...]) reading only tar headers."""
    rows = []
    # "r:" (not "r|") lets tarfile seek past file contents -> fast, headers only
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            sample_id, _, ext = base.partition(".")
            rows.append((sample_id, ext, m.offset_data, m.size))
    return tar_path, rows


def get_shard_index(root, out_path, jobs=8):
    root = os.path.abspath(root)
    tars = sorted(
        os.path.join(d, f)
        for d, _, files in os.walk(root)
        for f in files
        if f.endswith(".tar")
    )
    print(f"found {len(tars)} shards under {root}", file=sys.stderr)

    id_to_shards = defaultdict(set)   # id -> shards it appears in
    seen = set()                      # (id, ext) pairs, to catch duplicates
    dup_files = []

    with Pool(jobs) as pool, open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unique_id", "ext", "shard", "offset", "size"])
        for i, (tar_path, rows) in enumerate(pool.imap_unordered(files_in_shard, tars), 1):
            rel = os.path.relpath(tar_path, root)  # e.g. train/worker0_shard-000000.tar
            for sample_id, ext, offset, size in rows:
                w.writerow([sample_id, ext, rel, offset, size])
                id_to_shards[sample_id].add(rel)
                if (sample_id, ext) in seen:
                    dup_files.append((sample_id, ext, rel))
                seen.add((sample_id, ext))
            n_ids = len({r[0] for r in rows})
            print(f"[{i}/{len(tars)}] {rel}: {n_ids} ids, {len(rows)} files", file=sys.stderr)

    split = {k: sorted(v) for k, v in id_to_shards.items() if len(v) > 1}
    print(f"total ids: {len(id_to_shards)}  |  files: {len(seen) + len(dup_files)}", file=sys.stderr)
    print(f"ids spread over >1 shard: {len(split)}", file=sys.stderr)
    for k, v in list(split.items())[:10]:
        print(f"  split {k}: {v}", file=sys.stderr)
    print(f"duplicate (id, ext) files: {len(dup_files)}", file=sys.stderr)
    for d in dup_files[:10]:
        print(f"  dup {d}", file=sys.stderr)
    print(f"written {out_path}", file=sys.stderr)


def load_index(csv_path, root):
    """
    Read the CSV back into  {id: {ext: (abs_shard_path, offset, size)}}
    -- the structure the map-style Dataset uses.
    """
    root = os.path.abspath(root)
    index = defaultdict(dict)
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            index[r["unique_id"]][r["ext"]] = (
                os.path.join(root, r["shard"]),
                int(r["offset"]),
                int(r["size"]),
            )
    return dict(index)


if __name__ == "__main__":
    root = "/mnt/beegfs02/scratch/a_morelli/model_training/shards/PD/final_png_whitebg_21_07_26"
    shard_name = os.path.basename(root)
    out_path = os.path.join(
        "/home/a_morelli/datasets/id_lists/ids_to_shards",
        f"{shard_name}_id_to_shard_offsets.csv",
    )
    get_shard_index(root, out_path, jobs=8)