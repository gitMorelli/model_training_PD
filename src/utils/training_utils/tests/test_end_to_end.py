"""Train the original and the refactored module side by side on the same data
and require identical logged metrics and identical final weights."""
import copy
import logging

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader

import legacy_reference as legacy
from conftest import MemLogger, TinyModel, make_batches
from pd_training import (DataShape, EvalConfig, PDClassificationModule, SchedulerConfig,
                         default_callbacks, trainer_kwargs)

B, N_TRAIN, N_VAL, EPOCHS = 8, 12, 3, 2

CASES = {
    "binary_epoch_val":       dict(num_classes=1, counts=[60, 36], per_step=False),
    "binary_aligned":         dict(num_classes=1, counts=[60, 36], per_step=False,
                                   val=0.25, align=True, window=1),
    "ce2_aligned_prauc":      dict(num_classes=2, counts=[60, 36], per_step=False,
                                   val=0.5, align=True, window=1, drop_pr_auc=False),
    "multi_perstep_aligned":  dict(num_classes=3, counts=[32, 32, 32], per_step=True,
                                   val=0.5, align=True, window=1, accumulate=2),
    "multi_val_int_no_align": dict(num_classes=3, counts=[32, 32, 32], per_step=False, val=4),
}


def run(case, use_legacy):
    c = dict(window=50, val=None, align=False, accumulate=1, drop_pr_auc=True) | case
    out_dim = 1 if c["num_classes"] == 1 else c["num_classes"]
    n_labels = 2 if c["num_classes"] == 1 else c["num_classes"]
    L.seed_everything(0, verbose=False)
    model = TinyModel(out_dim, per_step=c["per_step"])
    train = DataLoader(make_batches(N_TRAIN, B, n_labels, seed=1), batch_size=None)
    val = DataLoader(make_batches(N_VAL, B, n_labels, seed=2), batch_size=None)
    logger = MemLogger()
    common = dict(accelerator="cpu", devices=1, logger=logger, enable_checkpointing=False,
                  enable_progress_bar=False, enable_model_summary=False,
                  log_every_n_steps=1, num_sanity_val_steps=2, deterministic=True)

    if use_legacy:
        module = legacy.ModelPDClassification(
            lambda *a, **k: None, copy.deepcopy(model), c["counts"],
            num_classes=c["num_classes"], total_units=N_TRAIN * B, batch_size=B,
            num_epochs=EPOCHS, accumulate_grad_batches=c["accumulate"],
            val_check_interval=c["val"], align_train_metrics_to_val=c["align"],
            min_window_steps=c["window"], drop_train_pr_auc=c["drop_pr_auc"])
        trainer = L.Trainer(max_epochs=EPOCHS, accumulate_grad_batches=c["accumulate"],
                            **module.trainer_val_kwargs, **common)
    else:
        shape = DataShape(total_units=N_TRAIN * B, batch_size=B, num_epochs=EPOCHS,
                          accumulate_grad_batches=c["accumulate"])
        ev = EvalConfig(val_check_interval=c["val"], align_train_metrics_to_val=c["align"],
                        min_window_steps=c["window"], drop_train_pr_auc=c["drop_pr_auc"])
        sched = SchedulerConfig()
        module = PDClassificationModule(copy.deepcopy(model), c["counts"], shape,
                                        num_classes=c["num_classes"], sched_cfg=sched, eval_cfg=ev)
        trainer = L.Trainer(**trainer_kwargs(shape, ev), callbacks=default_callbacks(sched), **common)

    trainer.fit(module, train, val)
    return logger.records, module.model.state_dict()


@pytest.mark.parametrize("name", list(CASES))
def test_same_metrics_and_weights(name):
    old_rec, old_w = run(CASES[name], use_legacy=True)
    new_rec, new_w = run(CASES[name], use_legacy=False)

    assert set(old_rec) == set(new_rec), (
        f"only legacy: {sorted(set(old_rec) - set(new_rec))[:10]}\n"
        f"only new:    {sorted(set(new_rec) - set(old_rec))[:10]}")
    for key in old_rec:
        assert new_rec[key] == pytest.approx(old_rec[key], rel=1e-5, abs=1e-7), key

    for k in old_w:
        assert torch.allclose(old_w[k], new_w[k], atol=1e-6), k

    # sanity: the interesting series actually exist
    keys = {k for _, k in new_rec}
    assert "val/balanced_accuracy" in keys and "lr/lr_backbone" in keys
    if CASES[name].get("align"):
        assert "train_window/balanced_accuracy" in keys


def test_two_stage_trains_end_to_end():
    L.seed_everything(0, verbose=False)
    model = TinyModel(1, batchnorm=True)
    shape = DataShape(total_units=N_TRAIN * B, batch_size=B, num_epochs=3)
    sched = SchedulerConfig(kind="two_stage", stage1_fraction=0.5)
    module = PDClassificationModule(model, [60, 36], shape, num_classes=1, sched_cfg=sched)
    backbone_before = copy.deepcopy(model.backbone.state_dict())
    seen = {}

    class Probe(L.Callback):
        def on_train_batch_end(self, trainer, pl_module, *a):
            if trainer.global_step == module.two_stage_plan.switch_step - 1:
                seen["frozen_bn_eval"] = not model.backbone[1].training
                seen["backbone_unchanged"] = all(
                    torch.equal(v, backbone_before[k]) for k, v in model.backbone.state_dict().items())

    trainer = L.Trainer(**trainer_kwargs(shape, EvalConfig()), accelerator="cpu", devices=1,
                        callbacks=default_callbacks(sched) + [Probe()], logger=False,
                        enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False)
    trainer.fit(module, DataLoader(make_batches(N_TRAIN, B, 2, seed=1), batch_size=None))

    assert seen == {"frozen_bn_eval": True, "backbone_unchanged": True}
    assert not torch.equal(model.backbone[0].weight, backbone_before["0.weight"])   # trained in stage 2
    assert all(p.requires_grad for p in model.parameters())
