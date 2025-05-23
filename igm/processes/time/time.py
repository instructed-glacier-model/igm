#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf

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
        # compute maximum ice velocitiy magnitude 
        velomax = tf.maximum(
            tf.reduce_max(tf.abs(state.ubar)),
            tf.reduce_max(tf.abs(state.vbar)),
        )
        # dt_target account for both cfl and dt_max
        if (velomax > 0):
            state.dt_target =  tf.minimum(
                cfg.processes.time.cfl * state.dx / velomax, 
                cfg.processes.time.step_max
            )
        else:
            state.dt_target = cfg.processes.time.step_max
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
