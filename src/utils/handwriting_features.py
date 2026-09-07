from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import theilslopes
from skimage.measure import euler_number, find_contours, label
from skimage.measure import perimeter as sk_perimeter
from skimage.morphology import skeletonize
"""
Per-cell handwriting feature extraction for writer identification, PD diagnosis
and gender detection from white-on-black handwriting grids.

Design contract
---------------
* The image is a grid of cells defined by `x_array` (vertical grid lines,
  including 0 and width) and `y_array` (horizontal grid lines, including 0 and
  height).  Each cell holds ~1 character (up to ~3 in some modalities).
* Every scalar is computed **per cell**, then reduced across the non-empty
  cells with mean / std / median / iqr.  The number of characters in the image
  therefore does not affect the descriptor length or (for intensive features)
  its expected value.
* Sizes are kept in **absolute pixels** (constant acquisition resolution is
  assumed).  Nothing is normalised by cell size: the grid is an assembly
  artefact, not part of the original page.
* Cells are traversed in row-major order, which is assumed to match writing
  order; this is what makes the size-trend (micrographia) features meaningful.

Ink polarity: ink is assumed to be **bright** (white on black).  Set
`ink_is_bright=False` for dark-on-light scans.
"""

__all__ = ["extract_image_properties"]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
_NEIGHBOUR_KERNEL = np.array([[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]], dtype=np.uint8)


def _f(x):
    """Cast to a plain python float, mapping non-finite values to nan."""
    x = float(x)
    return x if np.isfinite(x) else float("nan")


def _binarize(gray_u8, threshold=128, ink_is_bright=True, use_otsu=False):
    """Return a boolean ink mask from a uint8 grayscale array."""
    if use_otsu:
        from skimage.filters import threshold_otsu
        lo, hi = int(gray_u8.min()), int(gray_u8.max())
        threshold = threshold_otsu(gray_u8) if hi > lo else (lo + 1 if ink_is_bright else lo - 1)
    return gray_u8 >= threshold if ink_is_bright else gray_u8 <= threshold


def _runs(bool_1d):
    """Lengths of the True runs in a 1-D boolean array."""
    if bool_1d.size == 0 or not bool_1d.any():
        return np.empty(0, dtype=np.int64)
    padded = np.concatenate(([False], bool_1d, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[1::2] - edges[0::2]


def _interior_gap_runs(mask_2d, axis=0):
    """
    Lengths of the background runs strictly *between* the first and last ink
    pixel of each line, i.e. real intra-writing gaps rather than margin.
    `axis=0` scans rows (horizontal gaps), `axis=1` scans columns.
    """
    m = mask_2d if axis == 0 else mask_2d.T
    out = []
    for line in m:
        idx = np.flatnonzero(line)
        if idx.size < 2:
            continue
        out.append(_runs(~line[idx[0]:idx[-1] + 1]))
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def _ink_runs(mask_2d, axis=0):
    """Lengths of the ink runs along rows (axis=0) or columns (axis=1)."""
    m = mask_2d if axis == 0 else mask_2d.T
    out = [_runs(line) for line in m if line.any()]
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def _mean_std(values):
    """(mean, std) of an array, nan-safe on empty input."""
    if values is None or len(values) == 0:
        return float("nan"), float("nan")
    v = np.asarray(values, dtype=np.float64)
    return _f(v.mean()), _f(v.std())


# --------------------------------------------------------------------------- #
# contour-based features
# --------------------------------------------------------------------------- #
def _contour_features(mask, smooth_sigma=2.0, min_contour_points=12):
    """
    Mean absolute curvature (rad/px), direction reversals per px of contour and
    the RMS residual between the raw contour and a low-pass version of it
    (a tremor / jitter proxy).  Aggregated over all contours of the cell,
    weighted by contour length.
    """
    nan3 = {"curvature_mean_abs": float("nan"),
            "contour_reversals_per_px": float("nan"),
            "contour_jitter_rms": float("nan"),
            "contour_length": float("nan")}

    padded = np.pad(mask.astype(np.float64), 1)
    try:
        contours = find_contours(padded, 0.5)
    except Exception:
        return nan3

    tot_len = 0.0
    tot_abs_dtheta = 0.0
    tot_reversals = 0
    jitter_sq_sum = 0.0
    jitter_n = 0

    for c in contours:
        if c.shape[0] < min_contour_points:
            continue
        d = np.diff(c, axis=0)
        seg = np.hypot(d[:, 0], d[:, 1])
        keep = seg > 1e-9
        if keep.sum() < min_contour_points - 1:
            continue
        d, seg = d[keep], seg[keep]

        theta = np.arctan2(d[:, 0], d[:, 1])
        dtheta = np.diff(theta)
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

        length = float(seg.sum())
        tot_len += length
        tot_abs_dtheta += float(np.abs(dtheta).sum())

        sign = np.sign(dtheta)
        sign = sign[sign != 0]
        if sign.size > 1:
            tot_reversals += int((np.diff(sign) != 0).sum())

        # jitter: distance between the raw contour and a smoothed copy
        sm = np.stack([ndi.gaussian_filter1d(c[:, k], smooth_sigma, mode="wrap")
                       for k in range(2)], axis=1)
        resid = np.hypot(*(c - sm).T)
        jitter_sq_sum += float((resid ** 2).sum())
        jitter_n += resid.size

    if tot_len <= 0:
        return nan3

    return {
        "curvature_mean_abs": _f(tot_abs_dtheta / tot_len),
        "contour_reversals_per_px": _f(tot_reversals / tot_len),
        "contour_jitter_rms": _f(np.sqrt(jitter_sq_sum / jitter_n)) if jitter_n else float("nan"),
        "contour_length": _f(tot_len),
    }


def _slant_deg(mask, angles=np.arange(-45, 46, 3.0)):
    """
    Shear-search slant estimate: the shear angle whose vertical projection
    profile is most 'peaky' (max sum of squares) is taken as the slant.
    Positive = leaning right.
    """
    h, w = mask.shape
    if h < 3 or w < 2 or not mask.any():
        return float("nan")

    rows, cols = np.nonzero(mask)
    yc = rows - (h - 1) / 2.0
    best_angle, best_score = float("nan"), -np.inf
    pad = int(np.ceil(np.tan(np.deg2rad(np.abs(angles).max())) * h / 2.0)) + 1

    for a in angles:
        shifted = cols + pad - np.round(np.tan(np.deg2rad(a)) * yc).astype(np.int64)
        np.clip(shifted, 0, w + 2 * pad - 1, out=shifted)
        prof = np.bincount(shifted, minlength=w + 2 * pad).astype(np.float64)
        s = prof.sum()
        if s <= 0:
            continue
        score = float(((prof / s) ** 2).sum())
        if score > best_score:
            best_score, best_angle = score, float(a)
    return best_angle


# --------------------------------------------------------------------------- #
# per-cell feature block
# --------------------------------------------------------------------------- #
def _cell_features(mask, min_component_area=8, compute_slant=True):
    """All per-cell scalars.  `mask` is the boolean ink mask of one cell."""
    ink_area = int(mask.sum())
    rows, cols = np.nonzero(mask)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()

    bbox_h = float(r1 - r0 + 1)
    bbox_w = float(c1 - c0 + 1)
    crop = mask[r0:r1 + 1, c0:c1 + 1]

    feats = {}

    # ---- connected components -------------------------------------------- #
    lab, n_raw = label(crop, connectivity=2, return_num=True)
    if n_raw:
        sizes = np.bincount(lab.ravel())[1:]
        keep_ids = np.flatnonzero(sizes >= min_component_area) + 1
    else:
        sizes, keep_ids = np.empty(0), np.empty(0, dtype=int)
    n_comp = int(keep_ids.size)
    n_comp_safe = max(n_comp, 1)

    feats["n_components"] = float(n_comp)
    feats["n_components_raw"] = float(n_raw)
    feats["frag_ratio"] = _f((n_raw - n_comp) / max(n_raw, 1))  # tiny-fragment share
    feats["components_per_1k_ink_px"] = _f(1000.0 * n_comp / max(ink_area, 1))

    # ---- size ------------------------------------------------------------- #
    feats["bbox_height"] = bbox_h
    feats["bbox_width"] = bbox_w
    feats["bbox_width_per_comp"] = _f(bbox_w / n_comp_safe)
    feats["bbox_aspect"] = _f(bbox_h / bbox_w)
    feats["ink_area"] = float(ink_area)
    feats["ink_area_per_comp"] = _f(ink_area / n_comp_safe)
    feats["fill_ratio"] = _f(ink_area / (bbox_h * bbox_w))

    # ---- second central moments (absolute px) ----------------------------- #
    rf, cf = rows.astype(np.float64), cols.astype(np.float64)
    sx = float(cf.std())
    sy = float(rf.std())
    sxy = float(((cf - cf.mean()) * (rf - rf.mean())).mean())
    feats["sigma_x"] = _f(sx)
    feats["sigma_y"] = _f(sy)
    feats["sigma_xy"] = _f(sxy)
    feats["moment_orientation_deg"] = _f(
        np.rad2deg(0.5 * np.arctan2(2 * sxy, (sx ** 2 - sy ** 2))))
    # per-component horizontal spread: count-independent version of sigma_x
    if n_comp:
        per_comp_sx = []
        for cid in keep_ids:
            cc = np.nonzero(lab == cid)[1].astype(np.float64)
            per_comp_sx.append(cc.std())
        feats["sigma_x_per_comp"] = _f(np.mean(per_comp_sx))
    else:
        feats["sigma_x_per_comp"] = float("nan")

    # ---- stroke morphology ------------------------------------------------ #
    dist = ndi.distance_transform_edt(crop)
    skel = skeletonize(crop)
    skel_len = float(skel.sum())
    feats["skeleton_length"] = skel_len
    feats["skeleton_len_per_ink_area"] = _f(skel_len / max(ink_area, 1))

    if skel_len > 0:
        widths = 2.0 * dist[skel]
        w_mean, w_std = _mean_std(widths)
        feats["stroke_width_mean"] = w_mean
        feats["stroke_width_std"] = w_std
        feats["stroke_width_cv"] = _f(w_std / w_mean) if w_mean else float("nan")
        feats["stroke_width_p10"] = _f(np.percentile(widths, 10))
        feats["stroke_width_p90"] = _f(np.percentile(widths, 90))

        nb = ndi.convolve(skel.astype(np.uint8), _NEIGHBOUR_KERNEL,
                          mode="constant", cval=0)
        nb = nb[skel]
        feats["endpoints_per_100_skel_px"] = _f(100.0 * (nb == 1).sum() / skel_len)
        feats["branchpoints_per_100_skel_px"] = _f(100.0 * (nb >= 3).sum() / skel_len)
    else:
        for k in ("stroke_width_mean", "stroke_width_std", "stroke_width_cv",
                  "stroke_width_p10", "stroke_width_p90",
                  "endpoints_per_100_skel_px", "branchpoints_per_100_skel_px"):
            feats[k] = float("nan")

    feats["ink_area_per_skel_px"] = _f(ink_area / skel_len) if skel_len else float("nan")

    # ---- topology & outline ----------------------------------------------- #
    try:
        per = float(sk_perimeter(crop, neighborhood=8))
    except TypeError:  # older scikit-image spells it 'neighbourhood'
        per = float(sk_perimeter(crop, neighbourhood=8))
    feats["perimeter"] = _f(per)
    feats["compactness"] = _f(per ** 2 / max(ink_area, 1))
    try:
        eul = int(euler_number(crop, connectivity=2))
    except Exception:
        eul = 0
    feats["n_holes"] = float(max(n_raw - eul, 0))
    feats["holes_per_comp"] = _f(max(n_raw - eul, 0) / n_comp_safe)

    # ---- contour curvature / jitter --------------------------------------- #
    feats.update(_contour_features(crop))

    # ---- slant ------------------------------------------------------------- #
    feats["slant_deg"] = _slant_deg(crop) if compute_slant else float("nan")

    # ---- run lengths inside the bbox --------------------------------------- #
    for axis, tag in ((0, "h"), (1, "v")):
        m, s = _mean_std(_ink_runs(crop, axis=axis))
        feats[f"ink_run_{tag}_mean"], feats[f"ink_run_{tag}_std"] = m, s
        m, s = _mean_std(_interior_gap_runs(crop, axis=axis))
        feats[f"gap_run_{tag}_mean"], feats[f"gap_run_{tag}_std"] = m, s

    return feats


# --------------------------------------------------------------------------- #
# reductions across cells
# --------------------------------------------------------------------------- #
_SCHEMA_CACHE = None


def _feature_schema():
    """
    Canonical ordered list of per-cell feature names.  Derived once from a tiny
    synthetic blob so that the returned dict always has the *same keys*, even
    for images where every cell is empty -- required for default_collate.
    """
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        probe = np.zeros((9, 9), dtype=bool)
        probe[2:7, 2:7] = True
        probe[3:6, 3:6] = False          # give it a hole and a real contour
        _SCHEMA_CACHE = list(_cell_features(probe, min_component_area=1,
                                            compute_slant=True).keys())
    return _SCHEMA_CACHE


def _reduce(per_cell, prefix="cell_",
            reductions=("mean", "std", "median", "iqr")):
    """Aggregate a list of per-cell dicts into flat scalars (nan-safe)."""
    out = {}
    keys = _feature_schema()
    for k in keys:
        v = np.array([d.get(k, np.nan) for d in per_cell], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            for r in reductions:
                out[f"{prefix}{k}_{r}"] = float("nan")
            continue
        for r in reductions:
            if r == "mean":
                out[f"{prefix}{k}_mean"] = _f(v.mean())
            elif r == "std":
                out[f"{prefix}{k}_std"] = _f(v.std())
            elif r == "median":
                out[f"{prefix}{k}_median"] = _f(np.median(v))
            elif r == "iqr":
                out[f"{prefix}{k}_iqr"] = _f(np.percentile(v, 75) - np.percentile(v, 25))
            elif r == "cv":
                m = v.mean()
                out[f"{prefix}{k}_cv"] = _f(v.std() / m) if m else float("nan")
    return out


def _trend(indices, values, min_points=4):
    """
    Robust (Theil-Sen) slope of `values` against `indices`.
    Returns slope in units/cell and the slope normalised by the mean value,
    which makes it comparable across writers of different baseline size.
    """
    idx = np.asarray(indices, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(idx) & np.isfinite(val)
    idx, val = idx[ok], val[ok]
    if idx.size < min_points or np.ptp(idx) == 0:
        return float("nan"), float("nan")
    slope = float(theilslopes(val, idx)[0])
    mean = float(val.mean())
    return _f(slope), (_f(slope / mean) if mean else float("nan"))


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def extract_image_properties(img_source,
                             x_array,
                             y_array,
                             threshold=128,
                             ink_is_bright=True,
                             use_otsu=False,
                             min_ink_pixels=12,
                             min_component_area=8,
                             compute_slant=True,
                             trend_keys=("bbox_height", "ink_area",
                                         "stroke_width_mean", "fill_ratio"), reductions=("mean", "std", "median", "iqr")):
    """
    Extract global + per-cell-aggregated handwriting features from one image.

    Parameters
    ----------
    img_source : PIL.Image
        White-on-black handwriting grid (any mode; converted to 'L').
    x_array, y_array : sequence of int
        Grid line coordinates, including the outer borders (0 .. width and
        0 .. height).  Cells are visited in row-major (writing) order.
    threshold : int
        Binarisation threshold on the 0-255 grayscale.  Ignored if `use_otsu`.
    ink_is_bright : bool
        True for white ink on black background.
    min_ink_pixels : int
        Cells with fewer ink pixels are treated as empty and skipped.
    min_component_area : int
        Connected components smaller than this are ignored when counting
        (kills diacritic fragments and speckle).
    trend_keys : tuple of str
        Per-cell features for which an across-cell robust slope is computed.

    Returns
    -------
    dict of plain python scalars (floats / ints / str / tuple), safe for
    torch's default_collate.
    """
    img = img_source.convert("L")
    arr = np.asarray(img)
    h, w = arr.shape

    xs = np.asarray(x_array, dtype=np.int64)
    ys = np.asarray(y_array, dtype=np.int64)
    xs = np.clip(xs, 0, w)
    ys = np.clip(ys, 0, h)

    mask = _binarize(arr, threshold=threshold,
                     ink_is_bright=ink_is_bright, use_otsu=use_otsu)

    arr_f = arr.astype(np.float64) / 255.0

    props = {
        # ---- provenance / global -------------------------------------- #
        "is_uniform": bool(arr.max() == arr.min()),
        # ---- grid metadata (NOT features) ------------------------------ #
        "grid_rows": int(max(len(ys) - 1, 0)),
        "grid_cols": int(max(len(xs) - 1, 0)),
    }

    # ---- per-cell pass ------------------------------------------------- #
    per_cell, cell_indices = [], []
    k = 0
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            y0, y1 = int(ys[j]), int(ys[j + 1])
            x0, x1 = int(xs[i]), int(xs[i + 1])
            k += 1
            if y1 - y0 < 2 or x1 - x0 < 2:
                continue
            cm = mask[y0:y1, x0:x1]
            if cm.sum() < min_ink_pixels:
                continue
            per_cell.append(_cell_features(cm,
                                           min_component_area=min_component_area,
                                           compute_slant=compute_slant))
            cell_indices.append(k - 1)

    props["n_cells_total"] = int(max(len(ys) - 1, 0) * max(len(xs) - 1, 0))
    props["n_cells_nonempty"] = int(len(per_cell))
    props["nonempty_cell_fraction"] = _f(len(per_cell) / props["n_cells_total"]) \
        if props["n_cells_total"] else float("nan")

    props.update(_reduce(per_cell, reductions=reductions))

    # ---- across-cell trends (micrographia) ------------------------------ #
    for key in trend_keys:
        vals = [d.get(key, np.nan) for d in per_cell]
        slope, slope_norm = _trend(cell_indices, vals)
        props[f"trend_{key}_slope_per_cell"] = slope
        props[f"trend_{key}_slope_norm"] = slope_norm

    return props