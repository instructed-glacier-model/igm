#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Optional

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.schedule import (
    check_schedule,
    is_positive,
    is_schedule,
    parse_schedule,
    value_at,
)


def _reduce_for_cfl(
    x: tf.Tensor, percentile: float, active_mask: Optional[tf.Tensor] = None
) -> tf.Tensor:
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
        return tf.reduce_max(tf.where(active_mask, abs_x, tf.zeros_like(abs_x)))
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
            tf.math.ceil(tf.cast(n, tf.float32) * (100.0 - percentile) / 100.0),
            tf.int32,
        ),
    )
    top = tf.math.top_k(flat, k=keep_tail, sorted=True).values
    return top[-1]


@tf.function(autograph=False, reduce_retracing=True)
def compute_dt_from_cfl(
    ubar: tf.Tensor,
    vbar: tf.Tensor,
    cfl: float,
    dx: tf.Tensor,
    step_max: float,
    percentile: float = 100.0,
    active_mask: Optional[tf.Tensor] = None,
    ablation_speed: Optional[tf.Tensor] = None,
    t: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Compute adaptive time step based on CFL condition.

    `cfl` and `step_max` are numbers or schedules over model time, best parsed
    once (``igm.utils.schedule.parse_schedule``), and are read at the model time
    `t` inside the graph: a change of value neither retraces nor synchronises
    with the host.

    `percentile` < 100 uses the percentile of |velocity| in place of the
    exact max, so isolated outlier cells (a handful of unconverged cells
    near boundaries, or a transient instability spike) do not crash dt
    toward zero. The standard CFL guarantee then holds for everywhere
    except the top (100-percentile)% of cells.

    `ablation_speed` (m/yr), the calving plus frontal-melt rate of the
    ``calving_rate`` process, bounds the time step too, so that the front
    retreats at most `cfl` cells per step relative to the ice. Its maximum
    over the whole front band is used: conservative when the largest rate
    sits on band cells no front scheme reads.
    """
    cfl = value_at(cfl, t)
    step_max = value_at(step_max, t)
    velomax = tf.maximum(
        _reduce_for_cfl(ubar, percentile, active_mask),
        _reduce_for_cfl(vbar, percentile, active_mask),
    )
    if ablation_speed is not None:
        velomax = tf.maximum(
            velomax, tf.cast(tf.reduce_max(ablation_speed), velomax.dtype)
        )
    return tf.where(
        velomax > 0,
        tf.minimum(cfl * dx / velomax, step_max),
        tf.cast(step_max, velomax.dtype),
    )


def _ablation_speed(cfg: DictConfig, state: State) -> Optional[tf.Tensor]:
    """Calving plus frontal-melt rate of the calving_rate process, if active."""
    if "calving_rate" not in cfg.processes or not hasattr(state, "calving_rate"):
        return None
    rate = state.calving_rate
    if hasattr(state, "frontal_melt_rate"):
        rate = rate + state.frontal_melt_rate
    return rate


def initialize(cfg: DictConfig, state: State) -> None:

    cfg_time = cfg.processes.time
    check_schedule(cfg_time.step_max, "time.step_max", is_positive, "positive numbers")
    check_schedule(cfg_time.cfl, "time.cfl", is_positive, "positive numbers")

    # Initialize the time with starting time
    state.t = tf.Variable(float(cfg_time.start))

    state.itsave = -1

    # Parsed once, as they are read at every time step
    state.time_cfl = parse_schedule(cfg_time.cfl)
    state.time_step_max = parse_schedule(cfg_time.step_max)

    step_max = float(value_at(state.time_step_max, cfg_time.start))

    state.dt = tf.Variable(step_max)

    state.dt_target = tf.Variable(step_max)

    time_save_values = np.arange(
        cfg_time.start,
        cfg_time.end,
        cfg_time.save,
    ).tolist() + [cfg_time.end]
    state.time_save = tf.constant(time_save_values, dtype="float32")


def update(cfg: DictConfig, state: State) -> None:
    if hasattr(state, "logger"):
        # Avoid a device-to-host synchronization solely for log formatting.
        state.logger.info("Update time step")

    # A scheduled cfl has positive values only: the CFL condition is then always on.
    cfg_time = cfg.processes.time
    if is_schedule(state.time_cfl) or state.time_cfl > 0:
        state.dt_target = compute_dt_from_cfl(
            state.ubar,
            state.vbar,
            state.time_cfl,
            state.dx,
            state.time_step_max,
            percentile=float(getattr(cfg_time, "cfl_percentile", 100.0)),
            active_mask=getattr(state, "thk_active_mask", None),
            ablation_speed=_ablation_speed(cfg, state),
            t=state.t,
        )
    else:
        state.dt_target = value_at(state.time_step_max, state.t)

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

    state.continue_run = state.t < cfg_time.end


def finalize(cfg: DictConfig, state: State) -> None:
    pass
