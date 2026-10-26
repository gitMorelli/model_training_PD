from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, recall_score, precision_score,
                             f1_score, balanced_accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd
import os
from .helpers import _as_prob_matrix
from .plotting_helpers import (_plot_pr_curve_binary, _plot_roc_curve_binary,
                               _plot_pr_curve_mc, _plot_roc_curve_mc,
                               _plot_pr_curve_with_previous, _plot_roc_curve_with_previous)



def pick_threshold(y_true, y_scores, strategy="f1", target_recall=0.90, pos_label=1):
    """Choose a decision threshold for the positive class (binary only).
 
    strategy="f1"            -> threshold that maximizes F1 on the positive class
    strategy="youden"        -> threshold that maximizes Youden's J = TPR - FPR
                                (equivalently sensitivity + specificity - 1);
                                the point on the ROC curve furthest above the
                                chance diagonal. Weights both classes equally,
                                unlike F1, which ignores true negatives.
    strategy="target_recall" -> best-precision threshold that still hits
                                recall >= target_recall
    """
    if strategy == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
        # thresholds[0] is +inf (sklearn >= 1.3) or max(score) + 1: the degenerate
        # "predict everything negative" point, where J == 0. Drop it.
        if len(thresholds) > 1:
            fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]
        j = tpr - fpr
        return float(thresholds[np.argmax(j)])
 
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_scores, pos_label=pos_label
    )
    precision, recall = precision[:-1], recall[:-1]
 
    if strategy == "f1":
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall + 1e-12),
            0.0,
        )
        return float(thresholds[np.argmax(f1)])
 
    if strategy == "target_recall":
        ok = recall >= target_recall
        if not ok.any():
            return float(thresholds[np.argmax(recall)])
        idx = np.where(ok)[0]
        best = idx[np.argmax(precision[idx])]
        return float(thresholds[best])
 
    raise ValueError(f"unknown strategy: {strategy}")


def analyze_results(all_scores, all_labels, results_df, split="validation",
                    class_names=None, pos_label=1, threshold=None, strategy="f1",
                    target_recall=0.90, plot=False, out_dir_path='.'):
    """
    all_scores : per-sample class probabilities.
                 - binary: 1D P(pos) or 2D (N, 2)
                 - multiclass: 2D (N, C) probabilities or logits
                 (NOT hard predictions -- PR-AUC/thresholding need scores.)
    class_names: list of length C. Defaults to ["healthy","PD"] when C==2,
                 else ["class 0", ...].
    strategy   : "f1" | "youden" | "target_recall"  (binary only).
    Binary (C==2): threshold-based analysis, tuned via `strategy`/`threshold`.
                   With plot=True writes pr_curve_{split}.png and roc_curve_{split}.png.
    Multiclass (C>2): argmax-based analysis with macro / per-class metrics;
                      `pos_label`, `threshold`, `strategy` are ignored.
                      With plot=True writes one-vs-rest PR and ROC figures.
    Assumes labels are integers in [0, C-1].
    """
    y_true = all_labels.numpy() if hasattr(all_labels, "numpy") else np.asarray(all_labels)
    y_true = y_true.astype(int).ravel()
    scores = all_scores.numpy() if hasattr(all_scores, "numpy") else all_scores
    y_prob = _as_prob_matrix(scores)
    n_classes = y_prob.shape[1]
 
    if class_names is None:
        class_names = (["healthy", "PD"] if n_classes == 2
                       else [f"class {i}" for i in range(n_classes)])
    if len(class_names) != n_classes:
        raise ValueError(f"class_names has {len(class_names)} entries "
                         f"but scores imply {n_classes} classes")
 
    print(f"\n================ {split.upper()} STATISTICS ================")
 
    if n_classes == 2:
        result = _analyze_binary(y_true, y_prob, results_df, split, class_names,
                                 pos_label, threshold, strategy, target_recall,
                                 plot, out_dir_path)
    else:
        result = _analyze_multiclass(y_true, y_prob, results_df, split,
                                     class_names, plot, out_dir_path)
 
    print("=======================================================")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(results_df.head(10))
    return result

# MARK: Binary
def _threshold_free_report_binary(y_true, y_scores, pos_label=1):
    prevalence = float(np.mean(y_true == pos_label))
    pr_auc = average_precision_score(y_true, y_scores, pos_label=pos_label)
    roc_auc = roc_auc_score(y_true, y_scores)

    print("\n--- Threshold-free metrics ---")
    print(f"PR-AUC (avg precision) : {pr_auc:.3f}   "
          f"[random baseline = prevalence = {prevalence:.3f}]")
    print(f"ROC-AUC                : {roc_auc:.3f}   [random baseline = 0.500]")
    return pr_auc, roc_auc, prevalence


def _baseline_table_binary(y_true, pos_label=1, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    prevalence = float(np.mean(y_true == pos_label))

    def row(name, y_hat):
        return {
            "model": name,
            "PD_precision": precision_score(y_true, y_hat, pos_label=pos_label, zero_division=0),
            "PD_recall":    recall_score(y_true, y_hat, pos_label=pos_label, zero_division=0),
            "PD_f1":        f1_score(y_true, y_hat, pos_label=pos_label, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y_true, y_hat),
        }

    neg = 1 - pos_label
    all_healthy = np.full(n, neg, dtype=int)
    all_pd      = np.full(n, pos_label, dtype=int)
    unif        = rng.integers(0, 2, size=n)
    strat       = (rng.random(n) < prevalence).astype(int)

    return pd.DataFrame([
        row("always healthy (majority)", all_healthy),
        row("always PD (minority)",      all_pd),
        row("uniform random 50/50",      unif),
        row("stratified random",         strat),
    ])


def _analyze_binary(y_true, y_prob, results_df, split, class_names, pos_label,
                    threshold, strategy, target_recall, plot, out_dir_path):
    y_scores = y_prob[:, pos_label]
 
    # 1. threshold-free view
    pr_auc, roc_auc, prevalence = _threshold_free_report_binary(y_true, y_scores, pos_label)
 
    # 2. pick / apply a threshold
    if threshold is None:
        threshold = pick_threshold(y_true, y_scores, strategy=strategy,
                                   target_recall=target_recall, pos_label=pos_label)
        extra = (", target_recall=%.2f" % target_recall) if strategy == "target_recall" else ""
        print(f"\nChosen threshold ({strategy}{extra}): {threshold:.3f}")
    else:
        print(f"\nUsing fixed threshold: {threshold:.3f}")
    y_pred = (y_scores >= threshold).astype(int)
 
    # 2b. operating point in sensitivity / specificity terms
    sens = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=1 - pos_label, zero_division=0)
    print(f"  sensitivity = {sens:.3f}   specificity = {spec:.3f}   "
          f"Youden's J = {sens + spec - 1:.3f}")
 
    # 3. report at that threshold
    print("\n--- Classification Report @ threshold ---")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("--- Confusion Matrix @ threshold ---")
    print(confusion_matrix(y_true, y_pred))
 
    # 4. baselines
    print("\n--- Baseline comparison (positive = %s) ---" % class_names[pos_label])
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(_baseline_table_binary(y_true, pos_label).to_string(index=False))
 
    dirname = os.path.basename(os.path.normpath(out_dir_path))
    parent_path = os.path.dirname(os.path.normpath(out_dir_path))
    if plot:
        _plot_pr_curve_binary(y_true, y_scores, pos_label, threshold,
                              path=os.path.join(out_dir_path, f"pr_curve_{split}.png"), data_path=os.path.join(out_dir_path, f"pr_curve_{split}.npz"))
        _plot_roc_curve_binary(y_true, y_scores, pos_label, threshold,
                               path=os.path.join(out_dir_path, f"roc_curve_{split}.png"), data_path=os.path.join(out_dir_path, f"roc_curve_{split}.npz"))
        if dirname == "pre_diagnosis":
            _plot_pr_curve_with_previous(y_true, y_scores, prev_data_paths=[os.path.join(parent_path, f"pr_curve_{split}.npz")], pos_label=pos_label,
                                         threshold=threshold, path=os.path.join(out_dir_path, f"pr_curve_{split}.png"), compare_path=os.path.join(out_dir_path, f"pr_curve_{split}_comparison.png"),
                                         name=f"{split} (this run)", data_path=os.path.join(out_dir_path, f"pr_curve_{split}.npz"))
            _plot_roc_curve_with_previous(y_true, y_scores, prev_data_paths=[os.path.join(parent_path, f"roc_curve_{split}.npz")], pos_label=pos_label,
                                         threshold=threshold, path=os.path.join(out_dir_path, f"roc_curve_{split}.png"), compare_path=os.path.join(out_dir_path, f"roc_curve_{split}_comparison.png"),
                                         name=f"{split} (this run)", data_path=os.path.join(out_dir_path, f"roc_curve_{split}.npz"))
    #get the balanced accuracy score from the classification report
    balanced_acc = report["macro avg"]["recall"]
    f1_positive  = report["PD"]["f1-score"]
    precision_positive = report["PD"]["precision"]
    accuracy = report["accuracy"]
    return {"threshold": threshold, "y_pred": y_pred,
            "sensitivity": sens, "specificity": spec, "youden_j": sens + spec - 1,
            "pr_auc": pr_auc, "roc_auc": roc_auc, "prevalence": prevalence, "balanced_acc": balanced_acc, "f1_positive": f1_positive,
            "precision_positive": precision_positive, "accuracy": accuracy}


# MARK: Multiclass
def _analyze_multiclass(y_true, y_prob, results_df, split, class_names,
                        plot, out_dir_path):
    n_classes = y_prob.shape[1]
    labels = np.arange(n_classes)
 
    # 1. threshold-free view (macro OvR)
    macro_ap, roc_auc, prevalence = _threshold_free_report_mc(y_true, y_prob, class_names)
 
    # 2. predictions via argmax (no threshold in multiclass)
    y_pred = np.argmax(y_prob, axis=1)
 
    # 3. report
    print("\n--- Classification Report (argmax) ---")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=class_names, zero_division=0))
    print("--- Confusion Matrix (rows=true, cols=pred) ---")
    print(confusion_matrix(y_true, y_pred, labels=labels))
 
    # 4. baselines
    print("\n--- Baseline comparison (macro-averaged) ---")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(_baseline_table_mc(y_true, class_names).to_string(index=False))
 
    if plot:
        _plot_pr_curve_mc(y_true, y_prob, class_names,
                          path=os.path.join(out_dir_path, f"pr_curve_{split}.png"))
        _plot_roc_curve_mc(y_true, y_prob, class_names,
                           path=os.path.join(out_dir_path, f"roc_curve_{split}.png"))
    return {"threshold": None, "y_pred": y_pred,
            "pr_auc": macro_ap, "roc_auc": roc_auc}

def _threshold_free_report_mc(y_true, y_prob, class_names):
    C = y_prob.shape[1]
    classes = np.arange(C)
    prevalence = np.array([(y_true == c).mean() for c in classes])
    Y = label_binarize(y_true, classes=classes)   # (N, C) one-hot

    # per-class one-vs-rest average precision; macro = simple mean
    ap = np.full(C, np.nan)
    for c in classes:
        if Y[:, c].sum() > 0:                      # class present in y_true
            ap[c] = average_precision_score(Y[:, c], y_prob[:, c])
    macro_ap = np.nanmean(ap)

    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                average="macro", labels=classes)
    except ValueError:
        roc_auc = float("nan")                     # a class missing from y_true

    print("\n--- Threshold-free metrics (macro, one-vs-rest) ---")
    print(f"macro PR-AUC : {macro_ap:.3f}   "
          f"[random baseline = mean prevalence = {prevalence.mean():.3f}]")
    print(f"macro ROC-AUC: {roc_auc:.3f}   [random baseline = 0.500]")
    print("  per-class AP:")
    for c in classes:
        print(f"    {class_names[c]:>15s}: AP={ap[c]:.3f}   "
              f"[prevalence={prevalence[c]:.3f}]")
    return macro_ap, roc_auc, prevalence


def _baseline_table_mc(y_true, class_names, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    classes = np.arange(len(class_names))
    prevalence = np.array([(y_true == c).mean() for c in classes])
    majority = int(np.argmax(prevalence))

    def row(name, y_hat):
        return {
            "model": name,
            "macro_precision": precision_score(y_true, y_hat, average="macro", zero_division=0),
            "macro_recall":    recall_score(y_true, y_hat, average="macro", zero_division=0),
            "macro_f1":        f1_score(y_true, y_hat, average="macro", zero_division=0),
            "balanced_acc":    balanced_accuracy_score(y_true, y_hat),
        }

    always_majority = np.full(n, majority, dtype=int)
    unif            = rng.integers(0, len(classes), size=n)
    strat           = rng.choice(classes, size=n, p=prevalence)

    return pd.DataFrame([
        row(f"always majority ({class_names[majority]})", always_majority),
        row("uniform random",    unif),
        row("stratified random", strat),
    ])

