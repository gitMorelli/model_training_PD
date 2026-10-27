import numpy as np

def build_modality_names(modalities = None):
    if modalities is None:
        modalities = ['X_window','digit_window','text_window']
    #write all the combinations as modality[i] + modality[j] + modality[k] for i,j,k in combinations of 3 modalities
    from itertools import combinations
    modality_combinations = []
    for r in range(1, len(modalities) + 1):
        modality_combinations.extend(combinations(modalities, r))
    modality_names = ["+".join(combo) for combo in modality_combinations]
    return modality_names

# ======================================================================
# score helpers
# ======================================================================
def _as_pos_scores(all_scores, pos_label=1):
    """Binary only: accept 1D P(pos) OR a 2D (N,2) array, return 1D P(pos)."""
    s = np.asarray(all_scores)
    if s.ndim == 2 and s.shape[1] == 2:
        s = s[:, pos_label]
    return s.ravel()

def _as_prob_matrix(all_scores, num_classes=None):
    """Return an (N, C) probability matrix from various score shapes.

    Accepts:
      - 1D array of P(pos)        -> treated as binary, expanded to (N, 2)
      - 2D (N, C) probabilities   -> used as-is
      - 2D (N, C) logits          -> softmaxed (detected when rows don't sum to 1)

    argmax is invariant to softmax, but the AUC metrics need proper
    probabilities, so we normalize logits here.
    """
    s = np.asarray(all_scores, dtype=float)
    if s.ndim == 1:
        s = np.column_stack([1.0 - s, s])          # binary P(pos) -> (N, 2)

    row_sums = s.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):  # looks like logits -> softmax
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        s = e / e.sum(axis=1, keepdims=True)

    if num_classes is not None and s.shape[1] != num_classes:
        raise ValueError(f"expected {num_classes} columns, got {s.shape[1]}")
    return s
