"""Loading weights from checkpoints of the old ModelPDClassification.

The constructor changed, so old checkpoints can't go through
`load_from_checkpoint`. Build the new module, then load the weights:

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    load_legacy_state_dict(module, ckpt["state_dict"])

The old module also saved derived tensors (class_weights, pos_weight and the
loss modules' buffers). They are recomputed from class_counts now, so they are
dropped; only the `model.*` weights are loaded, strictly.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def load_legacy_state_dict(module, state_dict: dict):
    weights = {k: v for k, v in state_dict.items() if k.startswith("model.")}
    dropped = sorted(set(state_dict) - set(weights))
    if dropped:
        log.info("load_legacy_state_dict: ignoring derived keys %s", dropped)
    return module.load_state_dict(weights, strict=True)
