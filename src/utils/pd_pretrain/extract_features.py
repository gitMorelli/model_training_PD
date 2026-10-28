"""Extract embeddings from a trained checkpoint for the downstream PD classifier.

    python extract_features.py \
        --model_class pdhw.DenoisingAE --ckpt runs/dae/.../last.ckpt \
        --data.class_path my_project.MyDataModule --data.init_args.image_size 32 \
        --split predict --out features_dae.pt

    # diffusion + residual statistics from partial denoising
    python extract_features.py --model_class pdhw.DiffusionPretrainer --ckpt ... --anomaly true ...

The saved dict holds "features" (N, D) plus every non-image batch key
(subject_id, label, ...) in the same order, and "subject_features" aggregated
per subject if the batches carry "subject_id".
"""
import importlib

import torch
from jsonargparse import ArgumentParser
from lightning import LightningDataModule

from pdhw.features import aggregate_by_subject, extract_dataset


def main():
    p = ArgumentParser()
    p.add_argument("--model_class", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_subclass_arguments(LightningDataModule, "data")
    p.add_argument("--split", type=str, default="predict", choices=["train", "val", "test", "predict"])
    p.add_argument("--anomaly", type=bool, default=False)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="features.pt")
    args = p.parse_args()
    init = p.instantiate_classes(args)

    mod_name, cls_name = args.model_class.rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    model = cls.load_from_checkpoint(args.ckpt, map_location=args.device)

    dm = init.data
    dm.setup("predict" if args.split == "predict" else "fit")
    loader = getattr(dm, f"{args.split}_dataloader")()

    out = extract_dataset(model, loader, args.device, with_anomaly=args.anomaly)
    if "subject_id" in out:
        feats = torch.cat([out["features"], out["anomaly"]], 1) if "anomaly" in out else out["features"]
        out["subjects"], out["subject_features"] = aggregate_by_subject(feats, out["subject_id"])
    torch.save(out, args.out)
    print(f"saved {tuple(out['features'].shape)} features to {args.out}")


if __name__ == "__main__":
    main()
