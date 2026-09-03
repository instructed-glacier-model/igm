#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf


def _reduce_for_cfl(x, percentile, active_mask=None):
    """Reduce abs(x) to a single representative speed.

    percentile == 100 (default) → exact maximum (legacy behaviour).
    percentile <  100           → ignore the top (100-percentile)% of cells,
                                  i.e. take the value at rank ceil(p*N/100).
    """
    abs_x = tf.abs(x)
    if active_mask is not None:
        active_mask = tf.cast(active_mask, tf.bool)
    if percentile >= 100.0:
        if active_mask is None:
            return tf.reduce_max(abs_x)
        return tf.reduce_max(
            tf.where(active_mask, abs_x, tf.zeros_like(abs_x))
        )
    flat = (
        tf.reshape(abs_x, [-1])
        if active_mask is None
        else tf.boolean_mask(abs_x, active_mask)
    )
    flat = tf.cond(
        tf.size(flat) > 0,
        lambda: flat,
        lambda: tf.zeros((1,), dtype=abs_x.dtype),
    )
    n = tf.size(flat)
    # k = number of cells in the (100-p)% tail; top_k returns them in
    # descending order, and we want the smallest of those = the percentile.
    keep_tail = tf.maximum(
        1,
        tf.cast(
            tf.math.ceil(
                tf.cast(n, tf.float32) * (100.0 - percentile) / 100.0
            ),
            tf.int32,
        ),
    )
    top = tf.math.top_k(flat, k=keep_tail, sorted=True).values
    return top[-1]


@tf.function(autograph=False, reduce_retracing=True)
def compute_dt_from_cfl(
    ubar,
    vbar,
    cfl,
    dx,
    step_max,
    percentile=100.0,
    active_mask=None,
):
    """Compute adaptive time step based on CFL condition.

    `percentile` < 100 uses the percentile of |velocity| in place of the
    exact max, so isolated outlier cells (a handful of unconverged cells
    near boundaries, or a transient instability spike) do not crash dt
    toward zero. The standard CFL guarantee then holds for everywhere
    except the top (100-percentile)% of cells.
    """
    velomax = tf.maximum(
        _reduce_for_cfl(ubar, percentile, active_mask),
        _reduce_for_cfl(vbar, percentile, active_mask),
    )
    return tf.where(
        velomax > 0,
        tf.minimum(cfl * dx / velomax, step_max),
        tf.cast(step_max, velomax.dtype),
    )


def initialize(cfg, state):

    # Initialize the time with starting time
    state.t = tf.Variable(float(cfg.processes.time.start))

    state.itsave = -1

    state.dt = tf.Variable(float(cfg.processes.time.step_max))

    state.dt_target = tf.Variable(float(cfg.processes.time.step_max))

    time_save_values = np.arange(
        cfg.processes.time.start,
        cfg.processes.time.end,
        cfg.processes.time.save,
    ).tolist() + [cfg.processes.time.end]
    state.time_save = tf.constant(time_save_values, dtype="float32")


def update(cfg, state):
    if hasattr(state, "logger"):
        # Avoid a device-to-host synchronization solely for log formatting.
        state.logger.info("Update time step")

    if cfg.processes.time.cfl > 0:
        state.dt_target = compute_dt_from_cfl(
            state.ubar,
            state.vbar,
            cfg.processes.time.cfl,
            state.dx,
            cfg.processes.time.step_max,
            percentile=float(
                getattr(cfg.processes.time, "cfl_percentile", 100.0)
            ),
            active_mask=getattr(state, "thk_active_mask", None),
        )
    else:
        state.dt_target = cfg.processes.time.step_max

    state.dt = state.dt_target

    # modify dt such that times of requested savings are reached exactly
    if state.time_save[state.itsave + 1] <= state.t + state.dt:
        state.dt = state.time_save[state.itsave + 1] - state.t
        state.saveresult = True
        state.itsave += 1
    else:
        state.saveresult = False 

    # the first loop is not advancing
    if state.it >= 0:
        state.t.assign(state.t + state.dt)

    state.continue_run = state.t < cfg.processes.time.end


def finalize(cfg, state):
    pass
