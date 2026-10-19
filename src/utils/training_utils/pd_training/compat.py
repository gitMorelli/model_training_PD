"""Loading weights from checkpoints of the old ModelPDClassification.

The constructor changed, so old checkpoints can't go through
`load_from_checkpoint`. Load the weights into either the bare network or the
new LightningModule:

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    load_legacy_state_dict(model, ckpt["state_dict"])      # bare nn.Module
    load_legacy_state_dict(lit_module, ckpt["state_dict"]) # PDClassificationModule

In the checkpoint the network's weights are stored under "model.*" (the old
LightningModule held it as `self.model`). The old module also saved derived
tensors (class_weights, pos_weight, the loss modules' buffers); they are
recomputed from class_counts now, so they are dropped. Loading is strict.
"""
from __future__ import annotations

import logging

import torch.nn as nn

log = logging.getLogger(__name__)

PREFIX = "model."


def load_legacy_state_dict(module: nn.Module, state_dict: dict):
    weights = {k: v for k, v in state_dict.items() if k.startswith(PREFIX)}
    dropped = sorted(set(state_dict) - set(weights))
    if dropped:
        log.info("load_legacy_state_dict: ignoring derived keys %s", dropped)
    if not weights:
        raise ValueError(f"no '{PREFIX}*' keys in state_dict; is this a LightningModule checkpoint?")

    # A LightningModule wrapping the network as `.model` uses the checkpoint's
    # key names; a bare network needs the "model." prefix stripped.
    wraps_network = isinstance(getattr(module, "model", None), nn.Module) and any(
        k.startswith(PREFIX) for k in module.state_dict())
    if not wraps_network:
        weights = {k[len(PREFIX):]: v for k, v in weights.items()}
    return module.load_state_dict(weights, strict=True)