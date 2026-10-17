# pd_training

Refactor of `ModelPDBase` / `ModelPDClassification`. With the same data and
seed it logs the same metrics and ends with the same weights as the original
(checked by `tests/test_end_to_end.py`).

## Layout

```
pd_training/
  config.py            DataShape, OptimConfig/OptimGroup, SchedulerConfig, LossConfig, EvalConfig
  batch.py             PDBatch (named view of the collate output), make_example_input
  trainer_factory.py   trainer_kwargs(), default_callbacks(): Trainer args from the same configs
  logging_utils.py     attach_write_log(): route the package's logging to your write_log
  compat.py            load_legacy_state_dict(): weights from old checkpoints
  modules/
    base.py            PDModuleBase: forward, optimizer wiring, class-stat / windowed reporting
    classification.py  PDClassificationModule: loss, score metrics, predict_step
  optim/
    param_groups.py    explicit groups / LoRA layer decay / backbone-head split, weight-decay split
    schedulers.py      cosine with warmup, two-stage
  losses/
    class_weights.py   count validation, inverse-frequency weights, pos_weight
    criteria.py        ClassificationLoss (BCE/CE, optional per-subject averaging)
  metrics/
    class_stats.py     ClassStats torchmetrics Metric + payload/format helpers
    scores.py          logits -> preds/probs, PR-AUC/F1/MCC collection
  callbacks/
    diagnostics.py     ParameterSummary, GradientCheck, NamedLRLogger
    two_stage.py       TwoStageFreeze
tests/                 equivalence tests against the original (legacy_reference.py)
```

## Usage

```python
import lightning as L
from pd_training import (DataShape, EvalConfig, LossConfig, OptimConfig, SchedulerConfig,
                         PDClassificationModule, attach_write_log, default_callbacks,
                         trainer_kwargs)

attach_write_log(write_log)          # keep your existing log file

shape = DataShape(total_units=n_subjects, batch_size=32, world_size=n_gpus,
                  accumulate_grad_batches=1, num_epochs=10, iterable_train_dataset=True)
eval_cfg = EvalConfig(val_check_interval=0.25, align_train_metrics_to_val=True)
sched = SchedulerConfig(kind="cosine")

module = PDClassificationModule(model, class_counts=[n_neg, n_pos], data_shape=shape,
                                num_classes=1, loss_cfg=LossConfig(balancing_factor=1.0),
                                optim_cfg=OptimConfig(lr_backbone=1e-4, lr_head=1e-3),
                                sched_cfg=sched, eval_cfg=eval_cfg)

trainer = L.Trainer(**trainer_kwargs(shape, eval_cfg),
                    callbacks=default_callbacks(sched) + [checkpoint_cb],
                    devices=n_gpus, logger=tb_logger)
```

## Old argument -> new location

| Old (`ModelPDBase` / `ModelPDClassification`) | New |
|---|---|
| `write_log` | gone; call `attach_write_log(write_log)` once |
| `total_units`, `batch_size`, `world_size`, `accumulate_grad_batches`, `num_epochs`, `iterable_train_dataset` | `DataShape` |
| `lr_backbone`, `lr_classifier_head`, `weight_decay`, `lora_tuning`, `lora_lr_decay` | `OptimConfig` (`lr_classifier_head` -> `lr_head`) |
| `opt_groups=[{"names", "lr", "lr_name"}]` | `OptimConfig(groups=(OptimGroup(names, lr, lr_name), ...))` |
| `lr_scheduling`, `warmup_fraction`, `eta_min_cosine` | `SchedulerConfig(kind, warmup_fraction, eta_min)` |
| `use_balanced_weights`, `balancing_factor`, `balanced_data` | `LossConfig` |
| `val_check_interval`, `align_train_metrics_to_val`, `min_window_steps`, `drop_train_pr_auc` | `EvalConfig` |
| `module.trainer_val_kwargs` | `trainer_kwargs(data_shape, eval_cfg)` |
| `on_train_epoch_end` parameter table | `ParameterSummary` callback (`color=True` for the old ANSI output) |
| `on_after_backward` gradient check | `GradientCheck` callback |
| `_log_lrs` | `NamedLRLogger` callback |
| `_log_epoch_summary_extras` | `describe()` returning lines |
| `_update_class_counts(labels, stage, preds)` | `update_class_stats(labels, preds, stage)` |
| `make_example_input` (static method) | `pd_training.batch.make_example_input` |

Logged metric names are unchanged.

## Behaviour changes

- **Two-stage scheduling works.** The original read `stage1_fraction` and
  `stage1_lr_names`, which were never set, so it crashed. They are now
  `SchedulerConfig` fields (defaults `0.2` and `("lr_head",)`, so choose your
  own). The freezing the original referred to but never implemented is the
  `TwoStageFreeze` callback; under DDP it needs `find_unused_parameters=True`.
- An unknown `lr_scheduling` string used to mean "no scheduler" silently; now
  it raises. Use `kind="none"` for no scheduler.
- Class-stat metrics (`*/balanced_accuracy`, recalls, counts) are logged on
  every rank instead of rank 0 only. The values were already summed across
  ranks, and this lets ModelCheckpoint/EarlyStopping monitor them under DDP.
- `val/balanced_acc` is now the same number as `val/balanced_accuracy` instead
  of a second torchmetrics computation. It is kept as an alias so existing
  monitors work; drop it once nothing uses it. It is not logged for a
  validation run in which some class has no samples.
- Checkpoints contain only `model.*` weights. Class weights are recomputed from
  `class_counts` in the hyperparameters. Old checkpoints: build the new module
  and call `load_legacy_state_dict(module, ckpt["state_dict"])`.
- `ParameterSummary` defaults to no ANSI colours.

## Porting the grouped regime

Subclass `PDModuleBase` and implement `compute_loss_and_metrics(batch, stage)`.
Inside it call `self.update_class_stats(labels, preds, stage)`. Override
`train_score_metrics()` if you have a train MetricCollection that should be
windowed, and in that case don't pass it to `self.log_dict` when
`self.train_window_stats is not None`. `classification.py` is the template.

## Tests

```
pip install torch lightning torchmetrics pytest
pytest tests -q
```
