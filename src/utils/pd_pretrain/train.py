"""Training entry point.

    python train.py fit --config configs/dae.yaml
    python train.py fit --config configs/diffusion.yaml --data.init_args.image_size 64

Your own LightningDataModule plugs in through the `data:` section of the config
(class_path + init_args); it must yield batches following the contract in
pdhw/__init__.py.
"""
import torch
from lightning import LightningDataModule
from lightning.pytorch.cli import LightningCLI

from pdhw.base import PretrainModule


def main():
    torch.set_float32_matmul_precision("high")
    LightningCLI(
        PretrainModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
