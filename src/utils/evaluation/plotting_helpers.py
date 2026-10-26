import numpy as np
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, recall_score, precision_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# MARK: ROC curve plotting helpers

def _plot_roc_curve_binary(y_true, y_scores, pos_label=1, threshold=None,
                           path=None, name="model", data_path=None):
    """Plot a ROC curve and optionally dump the curve data to `data_path` (.npz)."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
    auc = float(roc_auc_score(y_true, y_scores))

    # Youden-optimal point on the curve
    if len(thresholds) > 1:
        j = (tpr - fpr)[1:]
        k = int(np.argmax(j)) + 1
        youden_fpr, youden_tpr = float(fpr[k]), float(tpr[k])
        youden_j, youden_thr = float(j[k - 1]), float(thresholds[k])
    else:
        youden_fpr = youden_tpr = youden_j = youden_thr = np.nan

    # the operating point actually used
    if threshold is not None:
        y_pred = (y_scores >= threshold).astype(int)
        sens = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        spec = float(recall_score(y_true, y_pred, pos_label=1 - pos_label, zero_division=0))
        thr_fpr, thr_tpr = 1.0 - spec, sens
    else:
        thr_fpr = thr_tpr = np.nan

    curve = {
        "name": name,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "threshold": np.nan if threshold is None else float(threshold),
        "threshold_fpr": thr_fpr,
        "threshold_tpr": thr_tpr,
        "youden_fpr": youden_fpr,
        "youden_tpr": youden_tpr,
        "youden_j": youden_j,
        "youden_threshold": youden_thr,
    }

    if data_path is not None:
        np.savez_compressed(data_path, **curve)

    if plt is None:
        print("matplotlib not available; skipping ROC plot")
        return curve

    _draw_roc_curves([curve], path=path)
    return curve

def _load_roc_curve(data_path):
    """Load a curve saved by _plot_roc_curve_binary."""
    with np.load(data_path, allow_pickle=False) as d:
        out = {"name": str(d["name"]), "fpr": d["fpr"], "tpr": d["tpr"]}
        for key in ("auc", "threshold", "threshold_fpr", "threshold_tpr",
                    "youden_fpr", "youden_tpr", "youden_j", "youden_threshold"):
            out[key] = float(d[key])
        return out

def _draw_roc_curves(curves, path=None, title=None, show_youden=True):
    """Draw one or more ROC curves (dicts from _plot_roc_curve_binary / _load_roc_curve)."""
    if plt is None:
        print("matplotlib not available; skipping ROC plot")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="random (AUC=0.500)")

    for c in curves:
        line, = ax.plot(c["fpr"], c["tpr"], label=f"{c['name']} (AUC={c['auc']:.3f})")
        color = line.get_color()
        if show_youden and not np.isnan(c["youden_fpr"]):
            ax.scatter([c["youden_fpr"]], [c["youden_tpr"]], facecolors="none",
                       edgecolors=color, s=90, zorder=4,
                       label=f"{c['name']} Youden J={c['youden_j']:.3f} "
                             f"@ {c['youden_threshold']:.3f}")
        if not np.isnan(c["threshold"]):
            ax.scatter([c["threshold_fpr"]], [c["threshold_tpr"]], color=color,
                       edgecolor="black", zorder=5,
                       label=f"{c['name']} thr={c['threshold']:.3f}")

    ax.set_xlabel("false positive rate (1 - specificity)")
    ax.set_ylabel("true positive rate (sensitivity)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    else:
        plt.show()

def _plot_roc_curve_with_previous(y_true, y_scores, prev_data_paths, pos_label=1,
                                  threshold=None, path=None, compare_path=None,
                                  name="exp2", data_path=None, show_youden=False):
    """Plot the current experiment alone, then superimposed with previous runs."""
    curve = _plot_roc_curve_binary(y_true, y_scores, pos_label=pos_label,
                                   threshold=threshold, path=path,
                                   name=name, data_path=data_path)

    if isinstance(prev_data_paths, (str, bytes)):
        prev_data_paths = [prev_data_paths]
    previous = [_load_roc_curve(p) for p in prev_data_paths]

    _draw_roc_curves(previous + [curve], path=compare_path,
                     title="ROC curves comparison", show_youden=show_youden)
    return curve

def _plot_roc_curve_mc(y_true, y_prob, class_names, path=None):
    if plt is None:
        print("matplotlib not available; skipping ROC plot")
        return
    classes = np.arange(len(class_names))
    Y = label_binarize(y_true, classes=classes)
 
    plt.figure(figsize=(6, 6))
    for c in classes:
        if Y[:, c].sum() == 0 or Y[:, c].sum() == len(y_true):
            continue                                  # AUC undefined for that class
        fpr, tpr, _ = roc_curve(Y[:, c], y_prob[:, c])
        auc = roc_auc_score(Y[:, c], y_prob[:, c])
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], ls="--", color="gray", label="random (AUC=0.500)")
 
    plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
    plt.title("One-vs-rest ROC curves")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()

# MARK: PR curve plotting helpers
#--------------- pr curve ---------
def _plot_pr_curve_binary(y_true, y_scores, pos_label=1, threshold=None,
                          path=None, name="model", data_path=None):
    """Plot a PR curve and optionally dump the curve data to `data_path` (.npz)."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=pos_label)
    ap = float(average_precision_score(y_true, y_scores, pos_label=pos_label))
    prevalence = float(np.mean(y_true == pos_label))

    if threshold is not None:
        yp = (y_scores >= threshold).astype(int)
        thr_recall = float(recall_score(y_true, yp, pos_label=pos_label, zero_division=0))
        thr_precision = float(precision_score(y_true, yp, pos_label=pos_label, zero_division=0))
    else:
        thr_recall = thr_precision = np.nan

    curve = {
        "name": name,
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "prevalence": prevalence,
        "threshold": np.nan if threshold is None else float(threshold),
        "threshold_recall": thr_recall,
        "threshold_precision": thr_precision,
    }

    if data_path is not None:
        np.savez_compressed(data_path, **curve)

    if plt is None:
        print("matplotlib not available; skipping PR plot")
        return curve

    _draw_pr_curves([curve], path=path, prevalence=prevalence)
    return curve


def _load_pr_curve(data_path):
    """Load a curve saved by _plot_pr_curve_binary."""
    with np.load(data_path, allow_pickle=False) as d:
        return {
            "name": str(d["name"]),
            "precision": d["precision"],
            "recall": d["recall"],
            "ap": float(d["ap"]),
            "prevalence": float(d["prevalence"]),
            "threshold": float(d["threshold"]),
            "threshold_recall": float(d["threshold_recall"]),
            "threshold_precision": float(d["threshold_precision"]),
        }


def _draw_pr_curves(curves, path=None, prevalence=None, title=None):
    """Draw one or more curves (dicts from _plot_pr_curve_binary / _load_pr_curve)."""
    if plt is None:
        print("matplotlib not available; skipping PR plot")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    for c in curves:
        line, = ax.plot(c["recall"], c["precision"], label=f"{c['name']} (AP={c['ap']:.3f})")
        if not np.isnan(c["threshold"]):
            ax.scatter([c["threshold_recall"]], [c["threshold_precision"]],
                       color=line.get_color(), edgecolor="black", zorder=5,
                       label=f"{c['name']} thr={c['threshold']:.3f}")

    if prevalence is None:
        prevalence = curves[0]["prevalence"]
    ax.axhline(prevalence, ls="--", color="gray", label=f"random (AP={prevalence:.3f})")

    ax.set_xlabel("recall (PD)"); ax.set_ylabel("precision (PD)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path)
        plt.close(fig)
    else:
        plt.show()


def _plot_pr_curve_with_previous(y_true, y_scores, prev_data_paths, pos_label=1,
                                 threshold=None, path=None, compare_path=None,
                                 name="exp2", data_path=None):
    """Plot the current experiment alone, then superimposed with previous runs."""
    curve = _plot_pr_curve_binary(y_true, y_scores, pos_label=pos_label,
                                  threshold=threshold, path=path,
                                  name=name, data_path=data_path)

    if isinstance(prev_data_paths, (str, bytes)):
        prev_data_paths = [prev_data_paths]
    previous = [_load_pr_curve(p) for p in prev_data_paths]

    _draw_pr_curves(previous + [curve], path=compare_path,
                    prevalence=curve["prevalence"], title="PR curves comparison")
    return curve

def _plot_pr_curve_mc(y_true, y_prob, class_names, path=None):
    if plt is None:
        print("matplotlib not available; skipping PR plot")
        return
    classes = np.arange(len(class_names))
    Y = label_binarize(y_true, classes=classes)

    plt.figure(figsize=(6, 6))
    for c in classes:
        if Y[:, c].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(Y[:, c], y_prob[:, c])
        ap = average_precision_score(Y[:, c], y_prob[:, c])
        plt.plot(recall, precision, label=f"{class_names[c]} (AP={ap:.3f})")
    plt.xlabel("recall"); plt.ylabel("precision")
    plt.title("One-vs-rest PR curves")
    plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
    plt.savefig(path) if path else plt.show()

