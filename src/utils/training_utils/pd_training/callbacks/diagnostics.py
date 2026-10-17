"""Diagnostics that used to live inside the LightningModule.

None of these change training; add or drop them in the Trainer's callback list.
"""
from __future__ import annotations

import logging
from typing import Sequence

import lightning as L
import os
import psutil
import time
from collections import deque
import torch
from lightning.pytorch.callbacks import Callback

from ..logging_utils import rank_zero_logger

info, warning = rank_zero_logger(__name__)

_WHITE, _RED, _RESET = "\033[97m", "\033[91m", "\033[0m"


class ParameterSummary(L.Callback):
    """At the end of the listed epochs (0-based), log every parameter with its
    shape, size and current learning rate, plus run-level counts.

    `color=True` adds ANSI colours (white = trainable, red = frozen); leave it
    off when the log goes to a file.
    """

    def __init__(self, epochs: Sequence[int] = (0, 5), color: bool = False):
        self.epochs = set(epochs)
        self.color = color

    def _c(self, text, colour):
        return f"{colour}{text}{_RESET}" if self.color else text

    def on_train_start(self, trainer, pl_module):
        info(f"Device of model at start of training: {pl_module.device}")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch not in self.epochs:
            return

        param_lr = {}
        for opt in trainer.optimizers:
            for group in opt.param_groups:
                for p in group["params"]:
                    param_lr[id(p)] = (group["lr"], group.get("lr_name", "?"))

        lines, n_train, n_frozen = [], 0, 0
        model = getattr(pl_module, "model", pl_module)
        for name, p in model.named_parameters():
            row = f"  - {name} | Shape: {list(p.shape)} | Parameters: {p.numel():,}"
            if p.requires_grad:
                lr, lr_name = param_lr.get(id(p), (None, None))
                row += f" | LR: {lr:.3e} ({lr_name})" if lr is not None else " | LR: NOT IN OPTIMIZER"
                lines.append(self._c(row, _WHITE))
                n_train += p.numel()
            else:
                lines.append(self._c(row + " | frozen", _RED))
                n_frozen += p.numel()

        out = [f"--- Epoch {epoch + 1} summary ---",
               f"Total parameters: {n_train + n_frozen:,} | "
               f"{self._c(f'Trainable: {n_train:,}', _WHITE)} | "
               f"{self._c(f'Non-trainable: {n_frozen:,}', _RED)}",
               f"Expected number of stepping batches: {trainer.estimated_stepping_batches}"]
        if hasattr(pl_module, "train_samples_seen"):
            out.append(f"Training samples processed this epoch: {pl_module.train_samples_seen}")
        data_shape = getattr(pl_module, "data_shape", None)
        if data_shape is not None:
            out.append(f"Expected total units (subjects or groups): {data_shape.total_units}")
        if hasattr(pl_module, "describe"):
            out += pl_module.describe()
        out.append("Parameters:")
        info("\n".join(out + lines))


class GradientCheck(L.Callback):
    """After the very first backward pass, log the mean |grad| of parameters
    whose name contains one of `name_filters`, and warn below `threshold`."""

    def __init__(self, name_filters: Sequence[str] = ("layer4", "classifier"),
                 threshold: float = 1e-8):
        self.name_filters = tuple(name_filters)
        self.threshold = threshold
        self._done = False

    def on_after_backward(self, trainer, pl_module):
        # with gradient accumulation global_step stays 0 for several backwards
        if self._done:
            return
        self._done = True
        if trainer.global_step != 0:      # resumed run
            return
        lines = ["--- Gradient check (first batch) ---"]
        model = getattr(pl_module, "model", pl_module)
        for name, p in model.named_parameters():
            if p.grad is None or not p.requires_grad:
                continue
            if any(f in name for f in self.name_filters):
                g = p.grad.abs().mean().item()
                lines.append(f"{name}: mean |grad| = {g:.2e}")
                if g < self.threshold:
                    lines.append(f"  -> WARNING: potential vanishing gradient in {name}")
        info("\n".join(lines))


class NamedLRLogger(L.Callback):
    """Log `lr/<lr_name>` every step, once per lr_name (the first group with
    that name). Unlike Lightning's LearningRateMonitor, which logs one series
    per param group, this gives one curve per logical group.

    Logged at batch start: by on_train_batch_end the scheduler has already
    stepped, so the value there would belong to the next step."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        seen = set()
        for opt in trainer.optimizers:
            for g in opt.param_groups:
                name = g.get("lr_name")
                if name and name not in seen:
                    pl_module.log(f"lr/{name}", g["lr"], on_step=True, on_epoch=False)
                    seen.add(name)

# MARK -------------- Optional Callbacks -------------------
#tracking experiments
class BestMetricTracker(L.Callback):
    """Tracks best-so-far validation metrics. Selection now follows PR-AUC
    (max), matching the ModelCheckpoint / EarlyStopping monitor, so best_epoch
    points at the epoch whose checkpoint was saved."""

    # "higher is better"  -> logged metric name : attribute
    _MAX_METRICS = {
        "val/pr_auc":       "best_val_pr_auc",
        "val/f1_pos":       "best_val_f1_pos",
        "val/mcc":          "best_val_mcc",
        "val/balanced_acc": "best_val_balanced_acc",
        "val/acc":          "best_val_acc",
        "train/acc_epoch":  "best_train_acc",   # note: train keeps the _epoch suffix
    }
    # "lower is better"
    _MIN_METRICS = {
        "val/loss":            "best_val_loss",              # weighted (training objective)
        "val/loss_unweighted": "best_val_loss_unweighted",   # proper NLL, diagnostic only
    }
    # metric that defines "the best epoch"
    _ANCHOR = ("val/pr_auc", "max")

    def __init__(self):
        super().__init__()
        for attr in self._MAX_METRICS.values():
            setattr(self, attr, float("-inf"))
        for attr in self._MIN_METRICS.values():
            setattr(self, attr, float("inf"))
        self.best_epoch = 0
        self._anchor_best = float("-inf") if self._ANCHOR[1] == "max" else float("inf")

    @staticmethod
    def _get(metrics, key):
        """Fetch a metric as a float, or None if missing/NaN."""
        v = metrics.get(key)
        if v is None:
            return None
        v = v.item() if hasattr(v, "item") else float(v)
        return v if v == v else None      # drop NaN (e.g. degenerate pr-auc epoch)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:       # skip the sanity pass
            return
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        for key, attr in self._MAX_METRICS.items():
            v = self._get(metrics, key)
            if v is not None:
                setattr(self, attr, max(getattr(self, attr), v))

        for key, attr in self._MIN_METRICS.items():
            v = self._get(metrics, key)
            if v is not None:
                setattr(self, attr, min(getattr(self, attr), v))

        # anchor best_epoch to the selection metric
        anchor_key, anchor_mode = self._ANCHOR
        v = self._get(metrics, anchor_key)
        if v is not None:
            improved = (v > self._anchor_best if anchor_mode == "max"
                        else v < self._anchor_best)
            if improved:
                self._anchor_best = v
                self.best_epoch = epoch

    @property
    def best_metrics(self):
        """Convenience dict for end-of-run printing."""
        d = {attr: getattr(self, attr)
             for attr in list(self._MAX_METRICS.values()) + list(self._MIN_METRICS.values())}
        d["best_epoch"] = self.best_epoch
        return d

class ClearCache(L.Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()

class TimeLoader(Callback):
    def __init__(self):
        self.n, self.tot, self.win = 0, 0.0, []
    def on_train_batch_end(self, trainer, pl_module, *a):
        d = trainer.profiler.recorded_durations[
            "[_TrainingEpochLoop].train_dataloader_next"][-1]
        self.n += 1; self.tot += d; self.win.append(d)
        if self.n % 100 == 0:
            print(f"batch {self.n} mean={self.tot/self.n:.4f} "
                  f"last100={sum(self.win)/100:.4f} max={max(self.win):.4f}")
            self.win = []

class BatchTimer(Callback):
    """Times each training batch and pushes it to TensorBoard.
 
    Measures three things:
      * time/data_wait_s  -- how long the loop waited on the dataloader
                             (read from the profiler; needs Trainer(profiler="simple"))
      * time/batch_compute_s -- forward + backward + optimizer step
      * time/epoch_s      -- wall clock per epoch
 
    `sync_cuda=True` inserts torch.cuda.synchronize() so the compute number is
    real; without it CUDA calls are async and you mostly time the Python loop.
    """
 
    DATA_KEY = "[_TrainingEpochLoop].train_dataloader_next"
 
    def __init__(self, window=100, print_every=100, sync_cuda=False, verbose=False):
        self.window = window
        self.print_every = print_every
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.verbose = verbose
 
        self.n = 0
        self.tot_data = 0.0
        self.tot_compute = 0.0
        self.win_data = []
        self.win_compute = []
        self._t0 = None
        self._epoch_t0 = None
 
    # -- helpers --------------------------------------------------------------
    def _sync(self):
        if self.sync_cuda:
            torch.cuda.synchronize()
 
    def _data_wait(self, trainer):
        rec = getattr(trainer.profiler, "recorded_durations", None)
        if rec:
            durations = rec.get(self.DATA_KEY)
            if durations:
                return durations[-1]
        return float("nan")   # no profiler attached -> silently skip
 
    # -- hooks ----------------------------------------------------------------
    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_t0 = time.perf_counter()
 
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._sync()
        self._t0 = time.perf_counter()

        if self.lr_scheduling != "two_stage":
            return
        in_stage1 = self.global_step < self._switch_step
        # Frozen params get no .grad, so AdamW skips them entirely (no update, no weight decay).
        for p in self._stage2_params:
            p.requires_grad_(not in_stage1)
        # Lightning puts the whole model back in train() after validation, so re-apply every
        # batch: frozen BN layers must not update their running stats during stage 1.
        for m in self._stage2_bn:
            m.train(not in_stage1)
 
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync()
        compute = time.perf_counter() - self._t0
        data = self._data_wait(trainer)
 
        pl_module.log("time/batch_compute_s", compute,
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        if data == data:  # not NaN
            pl_module.log("time/data_wait_s", data,
                          on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
            frac = data / (data + compute) if (data + compute) > 0 else 0.0
            pl_module.log("time/data_wait_fraction", frac,
                          on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
            self.tot_data += data
            self.win_data.append(data)
 
        self.n += 1
        self.tot_compute += compute
        self.win_compute.append(compute)
 
        if self.verbose and self.n % self.print_every == 0:
            def stats(win, tot):
                if not win:
                    return "n/a"
                return (f"mean={tot / self.n:.4f} "
                        f"last{len(win)}={sum(win) / len(win):.4f} "
                        f"max={max(win):.4f}")
            print(f"batch {self.n} | data  {stats(self.win_data, self.tot_data)}")
            print(f"batch {self.n} | compute {stats(self.win_compute, self.tot_compute)}")
            self.win_data, self.win_compute = [], []
 
        # keep SimpleProfiler's recorded_durations from growing without bound
        rec = getattr(trainer.profiler, "recorded_durations", None)
        if rec and self.DATA_KEY in rec and len(rec[self.DATA_KEY]) > 10_000:
            rec[self.DATA_KEY] = rec[self.DATA_KEY][-100:]
 
    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_t0 is not None:
            pl_module.log("time/epoch_s", time.perf_counter() - self._epoch_t0,
                          on_step=False, on_epoch=True, batch_size=1)

class ThroughputMonitor(Callback):
    """Rolling-window training speed monitor.

    Logged every `window` batches:
      time/samples_per_s    -- wall-clock throughput. The number that matters.
      time/batches_per_s    -- same, per batch.
      time/gpu_busy_s       -- mean GPU time per batch, measured with CUDA events.
      time/gpu_utilization  -- gpu_busy / wall_clock. ~1.0 = GPU-bound,
                               < ~0.85 = the GPU is idling and you are losing time.
      time/data_wait_s      -- mean host-side wait on the dataloader. Only a
                               problem if utilization is also low; a wait that
                               overlaps with GPU work is free.

    Dataloader wait is read from the profiler when one is attached
    (Trainer(profiler="simple")); otherwise it falls back to the gap between
    consecutive batch hooks, which also includes callback/logging overhead.

    Notes / limits:
      * CUDA events time the current stream. Work issued on side streams
        (some DDP comm, custom prefetchers) is not counted, so utilization can
        read low on a run that is actually saturated.
      * Under DDP every rank logs its own numbers; these are rank-local by
        design, since stragglers are exactly what you want to see.
      * Utilization can drift slightly above 1.0 from event/wall-clock skew.
    """

    DATA_KEY = "[_TrainingEpochLoop].train_dataloader_next"

    def __init__(self, window=50, event_lag=16, samples_per_batch=None,
                 verbose=False):
        self.window = window
        self.event_lag = event_lag          # batches to wait before reading events
        self.samples_per_batch = samples_per_batch   # override if inference fails
        self.verbose = verbose
        self.cuda = torch.cuda.is_available()

        self._pending = deque()
        self._last_batch_end = None
        self._epoch_t0 = None
        self._reset_window()

    # -- internals ------------------------------------------------------------
    def _reset_window(self):
        self._wall_t0 = time.perf_counter()
        self._n_batches = 0
        self._n_samples = 0
        self._gpu_ms = 0.0
        self._gpu_n = 0
        self._data_s = 0.0
        self._data_n = 0

    @staticmethod
    def _infer_batch_size(batch):
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        if isinstance(batch, (list, tuple)):
            for item in batch:
                if isinstance(item, torch.Tensor):
                    return item.shape[0]
        if isinstance(batch, dict):
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
        return None

    def _data_wait(self, trainer):
        """Host-side dataloader wait. Profiler if available, hook gap otherwise."""
        rec = getattr(trainer.profiler, "recorded_durations", None)
        if rec:
            durations = rec.get(self.DATA_KEY)
            if durations:
                # trim so SimpleProfiler does not grow without bound
                if len(durations) > 10_000:
                    rec[self.DATA_KEY] = durations[-100:]
                return durations[-1]
        if self._last_batch_end is not None:
            return time.perf_counter() - self._last_batch_end
        return None

    def _drain(self, force=False):
        """Read back completed CUDA events. Never blocks unless force=True."""
        while self._pending:
            if not force and len(self._pending) <= self.event_lag:
                break
            start_ev, end_ev = self._pending[0]
            if not force and not end_ev.query():
                break                      # GPU still behind, try again later
            self._pending.popleft()
            end_ev.synchronize()           # no-op if query() already passed
            self._gpu_ms += start_ev.elapsed_time(end_ev)
            self._gpu_n += 1

    def _flush(self, pl_module, on_step=True):
        if self._n_batches == 0:
            return
        wall = time.perf_counter() - self._wall_t0
        if wall <= 0:
            return

        logs = {"time/batches_per_s": self._n_batches / wall}
        if self._n_samples:
            logs["time/samples_per_s"] = self._n_samples / wall
        if self._data_n:
            logs["time/data_wait_s"] = self._data_s / self._data_n
        if self._gpu_n:
            gpu_per_batch = self._gpu_ms / self._gpu_n / 1000.0
            logs["time/gpu_busy_s"] = gpu_per_batch
            # scale by batches in the window, since event readback lags
            logs["time/gpu_utilization"] = gpu_per_batch * self._n_batches / wall

        pl_module.log_dict(logs, on_step=on_step, on_epoch=not on_step,
                       prog_bar=False, batch_size=1)

        if self.verbose:
            parts = [f"{logs.get('time/samples_per_s', float('nan')):8.1f} samp/s"]
            if "time/gpu_utilization" in logs:
                parts.append(f"gpu_util {logs['time/gpu_utilization']:5.1%}")
                parts.append(f"gpu {logs['time/gpu_busy_s'] * 1000:6.1f} ms")
            if "time/data_wait_s" in logs:
                parts.append(f"data {logs['time/data_wait_s'] * 1000:6.1f} ms")
            print("  ".join(parts))

        self._reset_window()

    # -- hooks ----------------------------------------------------------------
    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_t0 = time.perf_counter()
        self._last_batch_end = None
        self._reset_window()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        wait = self._data_wait(trainer)
        if wait is not None:
            self._data_s += wait
            self._data_n += 1

        if self.cuda:
            start_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            self._start_ev = start_ev

        n = self.samples_per_batch or self._infer_batch_size(batch)
        self._pending_samples = n or 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.cuda:
            end_ev = torch.cuda.Event(enable_timing=True)
            end_ev.record()
            self._pending.append((self._start_ev, end_ev))
            self._drain()

        self._n_batches += 1
        self._n_samples += self._pending_samples
        self._last_batch_end = time.perf_counter()

        if self._n_batches >= self.window:
            self._flush(pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.cuda:
            self._drain(force=True)
        self._flush(pl_module, on_step=False)   # <-- add on_step=False
        if self._epoch_t0 is not None:
            pl_module.log("time/epoch_s", time.perf_counter() - self._epoch_t0,
                          on_step=False, on_epoch=True, batch_size=1)

class MemMonitor(L.Callback):
    def __init__(self, every_n_steps=1000):
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, *args, **kwargs):
        if trainer.global_step % self.every_n_steps != 0:
            return

        p = psutil.Process(os.getpid())
        main = p.memory_full_info().pss / 1e9

        kids = []
        for c in p.children(recursive=True):
            try:
                kids.append((c.pid, c.memory_full_info().pss / 1e9))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass  # worker died/respawned between listing and reading

        total = main + sum(m for _, m in kids)
        print(f"step {trainer.global_step}: main={main:.2f}GB "
              f"workers={[f'{m:.2f}' for _, m in kids]} "
              f"total={total:.2f}GB (PSS)")

class WriteProbe(L.Callback):
    def on_validation_end(self, trainer, pl_module):
        payload = torch.zeros(34_000_000, dtype=torch.float32)  # ~136 MB
        for tgt in ["/tmp/probe.ckpt", "/mnt/beegfs02/scratch/a_morelli/probe.ckpt"]:
            t = time.time(); torch.save(payload, tgt); d = time.time() - t
            print(f"[probe] {tgt:55s} {d:6.2f}s {136/d:7.1f} MB/s", flush=True)
            os.remove(tgt)

