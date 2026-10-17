"""Component-level equivalence with the original implementation."""
import copy
import math

import pytest
import torch

import legacy_reference as legacy
from conftest import TinyModel
from pd_training import (DataShape, EvalConfig, LossConfig, OptimConfig, OptimGroup,
                         PDClassificationModule, SchedulerConfig)
from pd_training.metrics import ClassStats, class_stats_payload
from pd_training.trainer_factory import val_check_kwargs


def noop(*a, **k):
    pass


def build_pair(num_classes, class_counts, per_step=False, legacy_kwargs=None, new_kwargs=None,
               total_units=40, batch_size=8):
    out_dim = 1 if num_classes == 1 else num_classes
    model = TinyModel(out_dim, per_step=per_step)
    old = legacy.ModelPDClassification(noop, copy.deepcopy(model), class_counts,
                                       num_classes=num_classes, total_units=total_units,
                                       batch_size=batch_size, **(legacy_kwargs or {}))
    new = PDClassificationModule(copy.deepcopy(model), class_counts,
                                 DataShape(total_units=total_units, batch_size=batch_size),
                                 num_classes=num_classes, **(new_kwargs or {}))
    return old, new


def groups_signature(module, opt):
    names = {id(p): n for n, p in module.model.named_parameters()}
    return [(g["lr_name"], g["lr"], g["weight_decay"], tuple(names[id(p)] for p in g["params"]))
            for g in opt.param_groups]


@pytest.mark.parametrize("explicit", [False, True])
def test_param_groups_match(explicit):
    lk, nk = {}, {}
    if explicit:
        lk = {"opt_groups": [{"names": ["classifier"], "lr": 5e-3, "lr_name": "head"},
                             {"names": ["backbone", "fc"], "lr": 2e-4, "lr_name": "body"}]}
        nk = {"optim_cfg": OptimConfig(groups=(OptimGroup(("classifier",), 5e-3, "head"),
                                               OptimGroup(("backbone", "fc"), 2e-4, "body")))}
    old, new = build_pair(1, [30, 10], legacy_kwargs=lk, new_kwargs=nk)
    o = old.configure_optimizers()["optimizer"]
    n = new.configure_optimizers()["optimizer"]
    assert groups_signature(old, o) == groups_signature(new, n)


def test_explicit_groups_leftover_raises():
    _, new = build_pair(1, [30, 10], new_kwargs={
        "optim_cfg": OptimConfig(groups=(OptimGroup(("classifier",), 1e-3, "head"),))})
    with pytest.raises(ValueError, match="matched no optimizer group"):
        new.configure_optimizers()


def test_cosine_schedule_matches():
    old, new = build_pair(1, [30, 10], legacy_kwargs={"num_epochs": 7})
    new.data_shape = DataShape(total_units=40, batch_size=8, num_epochs=7)
    old_cfg, new_cfg = old.configure_optimizers(), new.configure_optimizers()
    for _ in range(old.total_steps + 3):
        assert [g["lr"] for g in old_cfg["optimizer"].param_groups] == pytest.approx(
            [g["lr"] for g in new_cfg["optimizer"].param_groups], rel=1e-12)
        old_cfg["optimizer"].step(); new_cfg["optimizer"].step()
        old_cfg["lr_scheduler"]["scheduler"].step(); new_cfg["lr_scheduler"]["scheduler"].step()


def test_two_stage_legacy_crashes_new_works():
    old, new = build_pair(1, [30, 10], legacy_kwargs={"lr_scheduling": "two_stage"},
                          new_kwargs={"sched_cfg": SchedulerConfig(kind="two_stage", stage1_fraction=0.4)})
    with pytest.raises(AttributeError):          # stage1_fraction was never set
        old.configure_optimizers()
    cfg = new.configure_optimizers()
    opt, sched = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
    plan = new.two_stage_plan
    head = [g for g in opt.param_groups if g["lr_name"] == "lr_head"]
    body = [g for g in opt.param_groups if g["lr_name"] == "lr_backbone"]
    for step in range(new.data_shape.total_steps):
        if step < plan.switch_step:
            assert all(g["lr"] == 0.0 for g in body)
            assert all(g["lr"] > 0.0 for g in head)
        elif step == plan.switch_step + 1:
            assert all(g["lr"] > 0.0 for g in body)
        opt.step(); sched.step()
    assert plan.stage2_params and plan.switch_step == int(0.4 * new.data_shape.total_steps)


@pytest.mark.parametrize("num_classes,counts", [(1, [30, 10]), (2, [30, 10]), (3, [20, 5, 9])])
@pytest.mark.parametrize("balanced", [True, False])
@pytest.mark.parametrize("factor", [1.0, 0.5])
@pytest.mark.parametrize("per_step", [False, True])
def test_losses_match(num_classes, counts, balanced, factor, per_step):
    lk = {"use_balanced_weights": balanced, "balancing_factor": factor}
    nk = {"loss_cfg": LossConfig(use_balanced_weights=balanced, balancing_factor=factor)}
    old, new = build_pair(num_classes, counts, per_step=per_step, legacy_kwargs=lk, new_kwargs=nk)
    g = torch.Generator().manual_seed(1)
    B, out = 6, (1 if num_classes == 1 else num_classes)
    labels = torch.arange(B) % (2 if num_classes == 1 else num_classes)
    seq_ids = torch.tensor([0, 0, 1, 2, 2, 2, 3, 4, 5, 5])
    n = seq_ids.numel() if per_step else B
    logits = torch.randn(n, out, generator=g)
    lab = labels[seq_ids] if per_step else labels
    targets = lab.float().unsqueeze(1) if num_classes == 1 else lab.long()

    for old_crit, old_none, new_crit in [
            (old.criterion, old.criterion_none, new.criterion),
            (old.criterion_unweighted, old.criterion_unweighted_none, new.criterion_unweighted)]:
        if per_step:
            ref = old._subject_mean_loss(logits, targets, seq_ids, B, crit=old_none)
            got = new_crit(logits, targets, seq_ids, B)
        else:
            ref, got = old_crit(logits, targets), new_crit(logits, targets)
        assert got.item() == pytest.approx(ref.item(), rel=1e-6)


def test_class_stats_payload_matches_legacy():
    old, _ = build_pair(3, [20, 5, 9])
    captured = {}
    old.log = lambda name, v, **kw: captured.__setitem__(name, v)
    old._sum_across_ranks = lambda v: [int(x) for x in v]
    old.__dict__["_trainer"] = None
    labels = torch.tensor([0, 0, 1, 2, 2, 2, 1, 0])
    preds = torch.tensor([0, 1, 1, 2, 0, 2, 1, 0])
    old._update_class_counts(labels, "val", preds=preds)
    type(old).current_epoch = property(lambda self: 0)
    old._log_class_metrics("val")

    m = ClassStats(3)
    m.update(labels, preds)
    n, c = m.compute()
    payload, _, _ = class_stats_payload(n.tolist(), c.tolist(), "val")
    assert payload.keys() == captured.keys()
    for k in payload:
        assert payload[k] == pytest.approx(captured[k])


@pytest.mark.parametrize("interval", [None, 1.0, 0.25, 0.5, 3, 100])
@pytest.mark.parametrize("iterable", [True, False])
def test_val_check_kwargs_match(interval, iterable):
    old, _ = build_pair(1, [30, 10], legacy_kwargs={
        "val_check_interval": interval, "iterable_train_dataset": iterable}, total_units=200)
    shape = DataShape(total_units=200, batch_size=8, iterable_train_dataset=iterable)
    assert val_check_kwargs(shape, EvalConfig(val_check_interval=interval)) == old.trainer_val_kwargs


def test_checkpoint_contains_only_model_weights():
    _, new = build_pair(1, [30, 10])
    assert all(k.startswith("model.") for k in new.state_dict())


def test_legacy_weights_load():
    from pd_training.compat import load_legacy_state_dict
    old, new = build_pair(1, [30, 10])
    with torch.no_grad():
        for p in old.model.parameters():
            p.add_(1.0)
    load_legacy_state_dict(new, old.state_dict())
    for (k1, v1), (k2, v2) in zip(old.model.state_dict().items(), new.model.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
