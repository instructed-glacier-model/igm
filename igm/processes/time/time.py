#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf

def _reduce_for_cfl(x, percentile):
    """Reduce abs(x) to a single representative speed.

    percentile == 100 (default) → exact maximum (legacy behaviour).
    percentile <  100           → ignore the top (100-percentile)% of cells,
                                  i.e. take the value at rank ceil(p*N/100).
    """
    abs_x = tf.abs(x)
    if percentile >= 100.0:
        return tf.reduce_max(abs_x)
    flat = tf.reshape(abs_x, [-1])
    n = tf.size(flat)
    # k = number of cells in the (100-p)% tail; top_k returns them in
    # descending order, and we want the smallest of those = the percentile.
    keep_tail = tf.maximum(
        1, tf.cast(tf.math.ceil(
            tf.cast(n, tf.float32) * (100.0 - percentile) / 100.0
        ), tf.int32)
    )
    top = tf.math.top_k(flat, k=keep_tail, sorted=True).values
    return top[-1]


def compute_dt_from_cfl(ubar, vbar, cfl, dx, step_max, percentile=100.0):
    """Compute adaptive time step based on CFL condition.

    `percentile` < 100 uses the percentile of |velocity| in place of the
    exact max, so isolated outlier cells (a handful of unconverged cells
    near boundaries, or a transient instability spike) do not crash dt
    toward zero. The standard CFL guarantee then holds for everywhere
    except the top (100-percentile)% of cells.
    """
    velomax = tf.maximum(
        _reduce_for_cfl(ubar, percentile),
        _reduce_for_cfl(vbar, percentile),
    )
    # dt_target account for both cfl and dt_max
    if (velomax > 0):
        dt_target = tf.minimum( cfl * dx / velomax, step_max )
    else:
        dt_target = step_max

    return dt_target

def initialize(cfg, state):

    # Initialize the time with starting time
    state.t = tf.Variable(float(cfg.processes.time.start))

    state.itsave = -1

    state.dt = tf.Variable(float(cfg.processes.time.step_max))

    state.dt_target = tf.Variable(float(cfg.processes.time.step_max))

    state.time_save = np.ndarray.tolist(
        np.arange(cfg.processes.time.start, cfg.processes.time.end, cfg.processes.time.save)
    ) + [cfg.processes.time.end]

    state.time_save = tf.constant(state.time_save, dtype="float32")

def update(cfg, state):
    if hasattr(state, "logger"):
        state.logger.info(
            "Update DT at time : " + str(state.t.numpy())
        )

    if (cfg.processes.time.cfl>0):
        state.dt_target = compute_dt_from_cfl(
            state.ubar, state.vbar,
            cfg.processes.time.cfl, state.dx,
            cfg.processes.time.step_max,
            percentile=float(getattr(cfg.processes.time, "cfl_percentile", 100.0)),
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

    state.continue_run = (state.t < cfg.processes.time.end)

def finalize(cfg, state):
    pass
