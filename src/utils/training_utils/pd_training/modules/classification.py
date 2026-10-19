"""Flat regime: per-subject classification with BCE (1 logit) or CE (>= 2 logits)."""
from __future__ import annotations

from typing import Optional, Sequence

import torch.nn as nn

from ..batch import PDBatch
from ..config import DataShape, EvalConfig, LossConfig, OptimConfig, SchedulerConfig
from ..losses import (ClassificationLoss,SilentPositiveLoss, binary_pos_weight, inverse_frequency_weights,
                      n_stat_classes, validate_class_counts)
from ..metrics import build_score_metrics, logits_to_preds, probs_and_preds, scores_for_metrics
from .base import PDModuleBase


def detect_per_step(model: nn.Module) -> bool:
    """Whether the model returns (token_logits, subject_logits).

    Prefer exposing `per_step` on the model itself; the classifier.mixer path
    is kept for models that don't yet."""
    if hasattr(model, "per_step"):
        return bool(model.per_step)
    mixer = getattr(getattr(model, "classifier", None), "mixer", None)
    return bool(getattr(mixer, "per_step", False))


class PDClassificationModule(PDModuleBase):
    """Per-subject classification.

        num_classes == 1 -> single-logit head, BCEWithLogitsLoss; class_counts = [n_neg, n_pos]
        num_classes >= 2 -> CrossEntropyLoss; one count per class
        num_classes None -> inferred from len(class_counts) (CE head)

    With `loss_cfg.use_balanced_weights` the training loss uses inverse-frequency
    class weights (BCE: pos_weight = n_neg / n_pos), interpolated toward
    uniform by `loss_cfg.balancing_factor`. Validation additionally logs the
    unweighted loss as a calibration diagnostic; it is not a selection metric
    (a constant predictor at the base rate minimises it).

    Expects the flat collate; see `PDBatch`.
    """

    def __init__(self, model: nn.Module, class_counts: Sequence[float], data_shape: DataShape,
                 num_classes: Optional[int] = None,
                 loss_cfg: LossConfig = LossConfig(),
                 optim_cfg: OptimConfig = OptimConfig(),
                 sched_cfg: SchedulerConfig = SchedulerConfig(),
                 eval_cfg: EvalConfig = EvalConfig(),
                 example_input_array=None):
        counts, num_classes = validate_class_counts(class_counts, num_classes)
        super().__init__(model, data_shape, num_classes=num_classes, optim_cfg=optim_cfg,
                         sched_cfg=sched_cfg, eval_cfg=eval_cfg,
                         example_input_array=example_input_array)
        self.predict_subsets = None #on which subsets to compute predictions
        self.save_hyperparameters(ignore=["model", "example_input_array"])

        self.class_counts = counts.tolist()
        self.loss_cfg = loss_cfg
        self.per_step = detect_per_step(model)

        if loss_cfg.use_balanced_weights and num_classes == 1:
            if loss_cfg.loss_name == "silent":
                self.criterion = SilentPositiveLoss(
                    pi=loss_cfg.pi,
                    learnable_pi=loss_cfg.learnable_pi,
                    pos_weight=binary_pos_weight(counts, loss_cfg.balancing_factor))
            else:
                self.criterion = ClassificationLoss(
                    num_classes, pos_weight=binary_pos_weight(counts, loss_cfg.balancing_factor))
        elif loss_cfg.use_balanced_weights:
            self.criterion = ClassificationLoss(
                num_classes, class_weights=inverse_frequency_weights(counts, loss_cfg.balancing_factor))
        else:
            self.criterion = ClassificationLoss(num_classes)
        self.criterion_unweighted = ClassificationLoss(num_classes)

        k = n_stat_classes(num_classes)
        self.train_metrics = build_score_metrics(k, "train/", include_pr_auc=not eval_cfg.drop_train_pr_auc)
        self.val_metrics = build_score_metrics(k, "val/")

    # ---- outputs -> loss inputs --------------------------------------------------------
    def _split_outputs(self, outputs, b: PDBatch):
        """-> (logits for the loss, targets for the loss, subject-level logits)."""
        if self.per_step:
            logits, subject_logits = outputs
            labels = b.labels[b.seq_ids]          # one label per token
        else:
            logits = subject_logits = outputs
            labels = b.labels
        targets = labels.float().unsqueeze(1) if self.num_classes == 1 else labels.long()
        return logits, targets, subject_logits

    def _loss(self, criterion, logits, targets, b: PDBatch):
        if self.per_step:
            return criterion(logits, targets, b.seq_ids, b.size)
        return criterion(logits, targets)

    # ---- steps --------------------------------------------------------------------------
    def compute_loss_and_metrics(self, batch, stage):
        b = PDBatch.from_collate(batch)
        bsz = b.size
        outputs = self(*b.model_args(), **b.covariates())
        logits, targets, subject_logits = self._split_outputs(outputs, b)

        if stage == "train" and not self.guard_finite(logits, "outputs"):
            return None
        loss = self._loss(self.criterion, logits, targets, b)
        if stage == "train" and not self.guard_finite(loss, "loss"):
            return None

        labels = b.labels.long().view(-1)
        preds = logits_to_preds(subject_logits, self.num_classes)
        self.update_class_stats(labels, preds, stage)

        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, on_step=on_step, on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log(f"{stage}/acc", (preds == labels).float().mean(),
                 on_step=on_step, on_epoch=True, prog_bar=True, batch_size=bsz)

        scores = self.train_metrics if stage == "train" else self.val_metrics
        scores.update(scores_for_metrics(subject_logits, self.num_classes), labels)
        # In aligned mode the base class computes/resets train_metrics at each
        # window; if Lightning also owned it, it would reset it at epoch end
        # underneath us. Val is always Lightning's (reset per val run).
        if stage == "val" or self.train_window_stats is None:
            self.log_dict(scores, on_step=False, on_epoch=True, batch_size=bsz)

        if stage == "train":
            self.train_samples_seen += bsz
            return loss

        self.log("val/loss_unweighted", self._loss(self.criterion_unweighted, logits, targets, b),
                 on_epoch=True, batch_size=bsz)
        return None
    
    def _supports_subsets(self):
        return (callable(getattr(self.model, 'forward_subsets', None))
                and getattr(self.model.classifier, 'modality_aware', False))

    def _pack(self, outputs, b, present=None, tok_present=None):
        token_logits, subject_logits = outputs if self.per_step else (None, outputs)
        probs, preds = probs_and_preds(subject_logits, self.num_classes)
        out = {"probs": probs.detach().cpu(),
               "preds": preds.detach().cpu(),
               "labels": b.labels.detach().cpu().flatten(),
               "subject_ids": b.subject_ids}
        if present is not None:
            out["present"] = present.detach().cpu()     # (B,) subject has real crops in subset
        if self.per_step:
            tok_probs, tok_preds = probs_and_preds(token_logits, self.num_classes)
            out.update(tok_probs=tok_probs.detach().cpu(),
                    tok_preds=tok_preds.detach().cpu(),
                    tok_seq_ids=b.seq_ids.detach().cpu(),
                    tok_slot_ids=b.slot_ids.detach().cpu())
            if tok_present is not None:
                out["tok_present"] = tok_present.detach().cpu()
        return out

    def predict_step(self, batch, batch_idx):
        b = PDBatch.from_collate(batch)
        if not self.predict_subsets:
            return self._pack(self(*b.model_args(), **b.covariates()), b)
        if self.predict_subsets and not self._supports_subsets():
            raise ValueError('predict_subsets needs a model with forward_subsets and view_groups')
        outs, present, tok_present = self.model.forward_subsets(
            *b.model_args(), subsets=self.predict_subsets,
            return_token_present=True, **b.covariates())
        return {name: self._pack(o, b, present[name], tok_present[name])
                for name, o in outs.items()}

    # ---- base-class hooks ----------------------------------------------------------------
    def train_score_metrics(self):
        return self.train_metrics

    def on_val_class_stats(self, payload):
        # "val/balanced_acc" is the old name of the same number; kept so existing
        # ModelCheckpoint / EarlyStopping monitors keep working.
        if "val/balanced_accuracy" in payload:
            self.log("val/balanced_acc", payload["val/balanced_accuracy"], prog_bar=True)

    def describe(self):
        counts = ", ".join(f"Class {i}: {int(c)}" for i, c in enumerate(self.class_counts))
        c = self.criterion
        weights = c.pos_weight if c.binary else c.weight
        return [f"Samples per class -> {counts}",
                f"Balancing factor: {self.loss_cfg.balancing_factor}",
                f"Balanced data: {self.loss_cfg.balanced_data}, "
                f"use balanced weights: {self.loss_cfg.use_balanced_weights}",
                f"Loss weights ({'pos_weight' if c.binary else 'class weights'}): "
                f"{None if weights is None else weights.tolist()}"]
