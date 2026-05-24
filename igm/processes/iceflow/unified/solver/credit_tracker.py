#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Credit-tracker — the TEMPORAL lever (B) of the clever-patch study.

This module decides *when* the unified solver should retrain. It is
intentionally INDEPENDENT of the spatial lever (C, `patch_selection.py`):
both modules read the per-step input change |X(t) − X(t−1)| independently
and reduce it differently:

  - this module (B)         → global scalar Δ(t) → credit accumulator → trigger.
  - patch_selection.py (C)  → per-window aggregate → patch scoring → sampling.

The duplication is intentional and cheap (~one extra GPU reduce per step):
it keeps B and C fully decoupled. B does NOT read C's window list and C
does NOT read B's accumulators.

Mechanism (see clever-patch/archive_md/adaptive_unique_strategy.md §5–§7):

  δ_c(t) = mean_xy |X_c(t) − X_c(t−1)|              per-channel mean abs diff
  Δ̂_c(t) = δ_c(t) / σ_c                            channel-normalised
  Δ(t)   = mean_c Δ̂_c(t)  (eligible channels only)  scalar, z-units
  credit(t) = credit(t−1) + Δ(t)                    pure accumulator

`σ_c` is recomputed every `SIGMA_REFRESH_FREQ` simulation steps via a TF
`reduce_variance` over the current full field, independently of the
network's input_normalizer. Channels with `var_c < VAR_FLOOR` are excluded
from both the mean and the σ estimation — robust to ice-free bootstrap
(thk≡0 at t=0) and to genuinely-constant channels (e.g. dX).

Public functions:
  update_credit_observer(cfg, state)
      Called every simulation step BEFORE `get_status`. Updates the
      per-step accumulator. No-op when both `adaptive_training.enabled`
      and `adaptive_training.enabled_observation` are false.

  log_and_maybe_reset_credit_observer(cfg, state, status, do_solve)
      Called AFTER the optimizer step. Appends one JSONL line and, if a
      retrain just fired, resets `_ct_credit` and `_ct_steps_since_retrain`.

State attributes maintained on `state`:
  _ct_credit              float (CPU)            Σ Δ since last retrain
  _ct_steps_since_retrain int   (CPU)            steps since last retrain
  _ct_sigma_c             tf [C] (GPU)           per-channel σ
  _ct_var_c               tf [C] (GPU)           per-channel raw variance
  _ct_prev_X              tf [H, W, C] (GPU)     previous full field snapshot
  _ct_n_steps_observed    int   (CPU)            total steps (refresh cadence)
  _ct_last_delta          float (CPU)            last Δ value (for the log)
  _ct_last_delta_c        tf [C] (GPU)           last per-channel δ_c (for log)

The previous version also maintained `_ct_cumulative_delta_w` (per-window
accumulator of δ_w) and read C's `state._ap_cache["list_windows"]` to feed
it. Both were removed in the 2026-05-24 B/C separation: that signal was
exclusively C's, and C now uses its own `temporal_variance` scoring
directly. See clever-patch/CLAUDE.md (Implementation map) for context.
"""

import json

import tensorflow as tf

from igm.common import State


SIGMA_REFRESH_FREQ = 10    # refresh _ct_sigma_c every N steps. Cheap (one
                           # full-grid reduce_variance per refresh). The
                           # short cadence matters early in a run when a
                           # channel transitions from variance≈0 (ice-free
                           # bootstrap, thk=0) to a non-trivial scale: the
                           # accumulator must follow that change closely or
                           # it produces hugely-inflated z-units in the
                           # interim.
VAR_FLOOR = 1e-8           # var_c < VAR_FLOOR → channel is "constant" → it
                           # is fully excluded from both σ_c estimation AND
                           # the Δ reduction. Replaces an earlier σ_c floor
                           # that mistakenly included sqrt(epsilon)=1e-3
                           # channels in the mean.


def _ct_active(cfg_unified) -> bool:
    """Return True iff the credit observer should run.

    Active when EITHER:
      - `adaptive_time.method == "credit"` (drives the retrain trigger), OR
      - `adaptive_time.enabled_observation == true` (diagnostic only).
    """
    try:
        block = getattr(cfg_unified, "adaptive_time", None)
        if block is None:
            return False
        method = str(getattr(block, "method", "fixed_freq")).lower()
        if method == "credit":
            return True
        return bool(getattr(block, "enabled_observation", False))
    except Exception:
        return False


def _ct_get_credit_cfg(cfg_unified):
    """Safe accessor for processes.iceflow.unified.adaptive_time.credit."""
    block = getattr(cfg_unified, "adaptive_time", None)
    if block is None:
        return None
    return getattr(block, "credit", None)


def _current_step(state: State) -> int:
    if hasattr(state, "it"):
        it = state.it
        return int(it.numpy()) if hasattr(it, "numpy") else int(it)
    return 0


def _ct_full_field(cfg, state) -> tf.Tensor:
    """Build the full [H, W, C] input field as a TF float32 tensor on GPU.

    No GPU→CPU transfer here — everything downstream stays on GPU until the
    final small-scalar log step.
    """
    from igm.processes.iceflow.utils.data_preprocessing import fieldin_state_to_X
    X = fieldin_state_to_X(cfg, state)
    if not isinstance(X, tf.Tensor):
        X = tf.convert_to_tensor(X)
    return tf.cast(X, tf.float32)


def _ct_refresh_sigma_c(state, X_full: tf.Tensor) -> None:
    """Lazy refresh of per-channel σ_c, cached on state._ct_sigma_c (TF tensor).

    Computed every SIGMA_REFRESH_FREQ steps from the variance of the current
    full field across the spatial dims. Also caches the raw per-channel
    variance (state._ct_var_c) — `_ct_compute_delta` masks channels with
    `var_c < VAR_FLOOR` so genuinely constant channels are cleanly excluded.
    All ops stay on GPU; the resulting σ_c is a [C] tf.float32 tensor.
    Independent of the network's input_normalizer — see §5 of
    adaptive_unique_strategy.md.
    """
    n = int(getattr(state, "_ct_n_steps_observed", 0))
    var = getattr(state, "_ct_var_c", None)
    if var is None or (n % SIGMA_REFRESH_FREQ == 0):
        # Variance over spatial dims [H, W] → shape [C]. Stay on GPU.
        var = tf.math.reduce_variance(X_full, axis=[0, 1])
        state._ct_var_c = var
        # σ from raw variance; channels with var < VAR_FLOOR are excluded by
        # the mask in _ct_compute_delta. A tiny 1e-12 epsilon avoids
        # exactly-zero σ for cleaner downstream arithmetic.
        state._ct_sigma_c = tf.sqrt(tf.maximum(var, 1e-12))


def _ct_compute_delta(state, X_full: tf.Tensor):
    """Return (Δ_scalar_tf, δ_c_tf [C]) — both TF tensors on GPU.

    δ_c = mean_xy(|X_c - X_prev_c|), per-channel.
    Δ̂_c = δ_c / σ_c, channels with var_c < VAR_FLOOR are excluded (set to 0).
    Δ   = mean_c Δ̂_c restricted to eligible channels (hardcoded mean reduction).
    Bootstrap (no previous snapshot yet): returns zeros.

    The previous version also returned a per-cell [H, W] field for window
    aggregation. That output was removed in the 2026-05-24 B/C separation —
    C now scores its own windows via `temporal_variance` in patch_selection.py.
    """
    prev = getattr(state, "_ct_prev_X", None)
    sigma = state._ct_sigma_c                 # tf [C]
    var = getattr(state, "_ct_var_c", None)   # tf [C]
    C = int(X_full.shape[-1])

    if prev is None or tuple(prev.shape) != tuple(X_full.shape):
        zeros_c = tf.zeros([C], dtype=tf.float32)
        return tf.constant(0.0, dtype=tf.float32), zeros_c

    # All ops below stay on GPU.
    diff = tf.abs(X_full - prev)                       # [H, W, C]
    delta_c = tf.reduce_mean(diff, axis=[0, 1])         # [C]

    # Channel eligibility: var_c >= VAR_FLOOR (else channel is "constant",
    # divide by σ-floor would give huge numbers). Float mask: 1 if eligible,
    # 0 otherwise.
    mask = tf.cast(var >= VAR_FLOOR, tf.float32)        # [C]
    n_elig = tf.reduce_sum(mask)

    # z_c = δ_c / σ_c, but zero where the channel is ineligible.
    z_c = mask * (delta_c / sigma)                       # [C]
    # Δ = mean over ELIGIBLE channels only.
    Delta = tf.cond(
        n_elig > 0,
        lambda: tf.reduce_sum(z_c) / n_elig,
        lambda: tf.constant(0.0, dtype=tf.float32),
    )

    return Delta, delta_c


def update_credit_observer(cfg, state) -> None:
    """Per-step credit accumulator — fully on GPU (TF tensors throughout).

    Updates the `_ct_*` attributes on `state` (see module docstring).

    Only scalars are transferred GPU→CPU; the [H, W, C] field stays on GPU
    between steps (~1.3 GB on the exp3 grid).

    No-op unless `adaptive_time.method == "credit"` OR
    `adaptive_time.enabled_observation == true`.
    """
    cfg_unified = cfg.processes.iceflow.unified
    if not _ct_active(cfg_unified):
        return

    X_full = _ct_full_field(cfg, state)
    _ct_refresh_sigma_c(state, X_full)
    Delta_tf, delta_c_tf = _ct_compute_delta(state, X_full)

    # Per-step scalars transferred GPU→CPU here (negligible).
    Delta = float(Delta_tf.numpy())
    state._ct_credit = float(getattr(state, "_ct_credit", 0.0)) + Delta
    state._ct_steps_since_retrain = int(getattr(state, "_ct_steps_since_retrain", 0)) + 1
    state._ct_n_steps_observed = int(getattr(state, "_ct_n_steps_observed", 0)) + 1
    state._ct_prev_X = X_full                # KEPT ON GPU — no .numpy()
    state._ct_last_delta = Delta
    state._ct_last_delta_c = delta_c_tf      # tf [C], converted in the log step


def log_and_maybe_reset_credit_observer(cfg, state, status, do_solve: bool) -> None:
    """Append one JSONL line and reset accumulators if a retrain just fired.

    Called from `solve_iceflow` AFTER the optimizer step has run, so the
    log reflects the state at retrain time (with credit BEFORE the reset).

    No-op unless `adaptive_time.method == "credit"` OR
    `adaptive_time.enabled_observation == true`.
    """
    cfg_unified = cfg.processes.iceflow.unified
    if not _ct_active(cfg_unified):
        return

    credit_cfg = _ct_get_credit_cfg(cfg_unified)
    record_path = str(getattr(credit_cfg, "record_path", "credit_log.jsonl")
                      if credit_cfg is not None else "credit_log.jsonl")

    status_name = getattr(status, "name", str(status))
    is_retrain = bool(do_solve) and status_name in ("INIT", "WARM_UP", "DEFAULT")

    # σ_c and δ_c are TF tensors on GPU now; bring them to CPU once for the log.
    sigma_tf = getattr(state, "_ct_sigma_c", None)
    delta_c_tf = getattr(state, "_ct_last_delta_c", None)
    sigma_list = ([float(s) for s in sigma_tf.numpy()]
                  if isinstance(sigma_tf, tf.Tensor) else [])
    delta_c_list = ([float(d) for d in delta_c_tf.numpy()]
                    if isinstance(delta_c_tf, tf.Tensor) else [])

    rec = {
        "step": _current_step(state),
        "t": (float(state.t.numpy()) if hasattr(state, "t") and hasattr(state.t, "numpy")
              else None),
        "delta":  float(getattr(state, "_ct_last_delta", 0.0)),
        "credit": float(getattr(state, "_ct_credit", 0.0)),
        "steps_since_retrain": int(getattr(state, "_ct_steps_since_retrain", 0)),
        "sigma_c": sigma_list,
        "delta_c": delta_c_list,
        "status": status_name,
        "is_retrain": is_retrain,
    }
    try:
        with open(record_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass  # never let logging crash a run

    if is_retrain:
        # Reset the accumulator so the NEXT inter-retrain interval starts fresh.
        state._ct_credit = 0.0
        state._ct_steps_since_retrain = 0
