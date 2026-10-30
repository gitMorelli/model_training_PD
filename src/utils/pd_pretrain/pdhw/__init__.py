"""Self-supervised pre-training of handwriting backbones for PD classification.

Batch contract (what every model expects from your DataLoader):
    batch["image"]    : FloatTensor (B, 1, H, W) in [-1, 1], ink = +1, background = -1
    batch["modality"] : LongTensor  (B,) with 0 = digit, 1 = checkmark, 2 = text
    any other keys (e.g. "subject_id") are ignored by training and passed through
    by feature extraction.
"""

MODALITIES = ("digit", "checkmark", "text") #default value

from .dae import DenoisingAE  # noqa: E402
from .mae import MaskedAE  # noqa: E402
from .cvae import ConditionalVAE  # noqa: E402
from .diffusion import DiffusionPretrainer  # noqa: E402

__all__ = [
    "MODALITIES",
    "DenoisingAE",
    "MaskedAE",
    "ConditionalVAE",
    "DiffusionPretrainer",
]
