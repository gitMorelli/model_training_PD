"""Assign trainable parameters to AdamW param groups.

Every group carries an `lr_name` ("lr_backbone", "lr_head", or a name from an
explicit OptimGroup). The two-stage scheduler and the LR logger key on it.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Tuple

import torch.nn as nn

from ..config import OptimConfig

log = logging.getLogger(__name__)

NamedParams = List[Tuple[str, nn.Parameter]]

# Tensors that should never be weight-decayed, by name fragment.
NO_DECAY_KEYS = (
    "layer_scale",                   # convnext (torchvision), [C,1,1]
    "gamma",                         # convnext under timm / original naming
    "relative_position_bias_table",  # swin v1
    "cpb_mlp",                       # swin v2 continuous position bias MLP
    "logit_scale",                   # swin v2
    "pos_embed", "pos_embedding", "position_embedding",
    "cls_token", "class_token", "dist_token",
)


def split_decay(named_params: Iterable[Tuple[str, nn.Parameter]], extra_no_decay=()):
    """(decay, no_decay): biases, 1-D (norm) params and the NO_DECAY_KEYS
    tensors get no weight decay."""
    decay, no_decay = [], []
    keys = NO_DECAY_KEYS + tuple(extra_no_decay)
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or any(k in name for k in keys):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def make_groups(named: NamedParams, lr: float, lr_name: str, weight_decay: float) -> list:
    """Up to two groups (decayed / not decayed) sharing one learning rate."""
    decay, no_decay = split_decay(named)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay, "lr_name": lr_name})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "lr_name": lr_name})
    return groups


def _trainable(model: nn.Module) -> NamedParams:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def explicit_groups(model: nn.Module, cfg: OptimConfig) -> list:
    """One lr per OptimGroup; the first group whose name fragment matches wins.
    Raises if any trainable parameter matches no group."""
    groups, assigned = [], {}   # param name -> lr_name that claimed it
    trainable = _trainable(model)
    for group in cfg.groups:
        matched = [(n, p) for n, p in trainable
                   if any(key in n.lower() for key in group.names)]
        for n, _ in matched:
            if n in assigned:
                log.warning("parameter '%s' matches group '%s' but was already assigned "
                            "to group '%s' (first group in order wins)",
                            n, group.lr_name, assigned[n])
        named = [(n, p) for n, p in matched if n not in assigned]
        assigned.update({n: group.lr_name for n, _ in named})
        groups += make_groups(named, group.lr, group.lr_name, cfg.weight_decay)

    leftover = [n for n, _ in trainable if n not in assigned]
    if leftover:
        raise ValueError(f"{len(leftover)} trainable parameters matched no optimizer "
                         f"group and would silently not be trained: {leftover}")
    return groups


def lora_groups(model: nn.Module, cfg: OptimConfig) -> list:
    """timm layer-wise lr decay on the PEFT-wrapped encoder; everything else
    trainable is the head."""
    from peft import PeftModel
    from timm.optim import param_groups_layer_decay

    peft_backbone = next((m for m in model.modules() if isinstance(m, PeftModel)), None)
    if peft_backbone is None:
        raise ValueError("lora_tuning=True but the model contains no peft.PeftModel")
    encoder = peft_backbone.get_base_model()   # timm model with LoRA layers injected

    nwd = encoder.no_weight_decay() if hasattr(encoder, "no_weight_decay") else set()
    enc_groups = param_groups_layer_decay(
        encoder, weight_decay=cfg.weight_decay,
        no_weight_decay_list=nwd, layer_decay=cfg.lora_lr_decay)
    for g in enc_groups:
        g["lr"] = cfg.lr_backbone * g.pop("lr_scale", 1.0)
        g["lr_name"] = "lr_backbone"
    n_lrs = len({round(g["lr"], 12) for g in enc_groups})
    log.info("LoRA: %d encoder groups, %d distinct lrs", len(enc_groups), n_lrs)

    enc_ids = {id(p) for p in encoder.parameters()}
    rest = [(n, p) for n, p in _trainable(model) if id(p) not in enc_ids]
    return enc_groups + make_groups(rest, cfg.lr_head, "lr_head", cfg.weight_decay)


def backbone_head_groups(model: nn.Module, cfg: OptimConfig) -> list:
    """Parameters whose name contains one of `cfg.head_keys` -> head, rest -> backbone."""
    backbone, head = [], []
    for n, p in _trainable(model):
        (head if any(k in n.lower() for k in cfg.head_keys) else backbone).append((n, p))
    return (make_groups(backbone, cfg.lr_backbone, "lr_backbone", cfg.weight_decay)
            + make_groups(head, cfg.lr_head, "lr_head", cfg.weight_decay))


def build_param_groups(model: nn.Module, cfg: OptimConfig) -> list:
    if cfg.groups:
        return explicit_groups(model, cfg)
    if cfg.lora_tuning:
        return lora_groups(model, cfg)
    return backbone_head_groups(model, cfg)
