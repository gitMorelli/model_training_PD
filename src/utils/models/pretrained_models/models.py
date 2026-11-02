from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import pickle
import yaml

from src.utils.pd_pretrain.pdhw.cnn import ConvEncoder
from src.utils.pd_pretrain.pdhw.blocks import ink_mask, masked_spatial_mean
from src.utils.image_processing import get_transforms,get_mu_std

SOURCE_PATH = "/home/a_morelli/models/model_training_logs/pre_trained_models/unsupervised"

import os
import pickle

import yaml


def _load_preprocessing_params(path, keys=("norm_mu", "norm_std", "custom_transform"), section="experiment"):
    """Return the values of `keys` (default: norm_mu, norm_std, custom_transform) saved with a run.

    path: the run's config.yaml, or the run folder that contains it.
    Looks in exp_params.pkl next to the config first (the exact exp_params the run used),
    then in config.yaml under `section`, then at the top level of config.yaml.
    Raises KeyError if a key is found nowhere.
    """
    config_path = os.path.join(path, "config.yaml") if os.path.isdir(path) else path
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    sources = []
    '''pkl_path = os.path.join(os.path.dirname(config_path), "exp_params.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            sources.append(("exp_params.pkl", pickle.load(f)))'''
    sources += [(f"config.yaml['{section}']", cfg.get(section) or {}), ("config.yaml (top level)", cfg)]

    values = []
    for key in keys:
        for _, src in sources:
            if key in src:
                values.append(src[key])
                break
        else:
            raise KeyError(f"'{key}' not found in: {', '.join(name for name, _ in sources)}")
    return tuple(values)
def _process_name(name):
    """Return the pretraining mode and input size from a model name like ConvEncoder_dae_128."""
    parts = name.split("_")
    if len(parts) != 3 or parts[0] != "ConvEncoder":
        raise ValueError(f"Invalid ConvEncoder name: {name}")
    return parts[1], int(parts[2])
#ConvEncoder
mapping_conv_encoder = {
        ("dae",128): f"DenoisingAE_128px/version_7/checkpoints/epoch=14-step=60352.ckpt",
        ("cvae",128): None, #TODO: add path to cvae checkpoint
    }
class ConvEncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, unsup_modality: str = "dae", extraction_module: nn.Module = None,
                 out_dim=None, input_size=None):
        super().__init__()
        self.encoder = encoder
        self.unsup_modality = unsup_modality
        self.extraction_module = extraction_module
        self.out_dim = out_dim
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size is not None and x.shape[-1] != self.input_size:
            raise ValueError(f"encoder was pre-trained on {self.input_size}px, got {x.shape[-1]}px")
        z = self.encoder(x)   
        if self.unsup_modality == "dae":
            # x: (B,1,H,W) in [-1,1], ink = +1
            feats = torch.cat([z.mean(dim=(2, 3)), masked_spatial_mean(z, ink_mask(x))], dim=1)
        elif self.unsup_modality == "cvae":
            feats = self.extraction_module(z.flatten(1)).chunk(2, dim=1)[0]
        else:
            raise ValueError(f"Unsupported unsupervised modality: {self.unsup_modality}")
        return feats
def _load_conv_encoder_from_ckpt(pretraining_mode, ckpt_path, pretrained=True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp, sd = ckpt["hyper_parameters"], ckpt["state_dict"]

    def sub_state(prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    if pretraining_mode == "dae":
        encoder = ConvEncoder(hp["channels"], hp["blocks_per_stage"], hp["latent_channels"], hp["dropout"])
        extraction, out_dim = None, 2 * hp["latent_channels"]
    elif pretraining_mode == "cvae":
        encoder = ConvEncoder(hp["channels"], hp["blocks_per_stage"], 32, hp["dropout"])
        w = sd["ema.module.to_stats.weight"]                     # (2*z_dim, flat)
        extraction, out_dim = nn.Linear(w.shape[1], w.shape[0]), hp["z_dim"]
    else:
        raise ValueError(f"Pretraining mode {pretraining_mode} not supported.")

    if pretrained:                                               # else: same architecture, random init
        encoder.load_state_dict(sub_state("ema.module.encoder."))
        if extraction is not None:
            extraction.load_state_dict(sub_state("ema.module.to_stats."))
    return encoder, extraction, out_dim, hp.get("image_size")

def get_ConvEncoder(
    name: str = "ConvEncoder_dae_128",
    pretrained: bool = False,
) -> nn.Module:
    #name should be in the format ConvEncoder_{pretraining_mode}_{input_size}, e.g. ConvEncoder_dae_128
    pretraining_mode, input_size = _process_name(name)

    specific_path = mapping_conv_encoder.get((pretraining_mode, input_size), None)
    ckpt_path = os.path.join(SOURCE_PATH, specific_path) if specific_path else None

    encoder, extraction_module, out_dim, ckpt_size = _load_conv_encoder_from_ckpt(pretraining_mode, ckpt_path, pretrained=pretrained)


    return ConvEncoderWrapper(encoder, pretraining_mode, extraction_module, out_dim,
                              input_size=ckpt_size or input_size)
def get_ConvEncoder_transforms(name) -> transforms.Compose:
    """It is the custom transform applied during training -> depends only on the resolution"""
    pretraining_mode, input_size = _process_name(name)
    specific_path = mapping_conv_encoder.get((pretraining_mode, input_size), None)
    config_path = os.path.join(SOURCE_PATH, os.path.dirname(specific_path)) 
    exp_params = {
        'input_size': input_size
    }
    exp_params["norm_mu"], exp_params["norm_std"], exp_params["custom_transform"] = _load_preprocessing_params(config_path)
    exp_params["norm_mu"], exp_params["norm_std"] = get_mu_std(exp_params)

    return get_transforms(exp_params)