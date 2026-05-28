#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


def initialize_iceflow_fields(cfg: DictConfig, state: State) -> None:
    """Initialize iceflow fields: arrhenius, slidingco/tau_ref, U, V.

    Basal-friction field: the legacy stack (emulated/solved/diagnostic
    + data_assimilation) uses `state.slidingco` initialised from
    `cfg.processes.iceflow.physics.sliding.slidingco`; the new stack
    (unified + data_assimilation_SR + pretraining) uses `state.tau_ref`
    initialised from `cfg.processes.iceflow.physics.sliding.tau_ref`.
    Cross-stack readers use `igm.common.fields.get_tau_ref(state)`.
    """

    cfg_physics = cfg.processes.iceflow.physics
    Nz = cfg.processes.iceflow.numerics.Nz
    Ny = state.thk.shape[0]
    Nx = state.thk.shape[1]
    shape_2d = (Ny, Nx)
    shape_3d = (Nz, Ny, Nx)

    if not hasattr(state, "arrhenius"):
        init_value = cfg_physics.viscosity.arrhenius * cfg_physics.viscosity.enhancement_factor
        state.arrhenius = tf.ones(shape_2d) * init_value

    method = cfg.processes.iceflow.method.lower()
    if method == "unified":
        if not hasattr(state, "tau_ref"):
            state.tau_ref = tf.ones(shape_2d) * cfg_physics.sliding.tau_ref
    else:
        if not hasattr(state, "slidingco"):
            state.slidingco = tf.ones(shape_2d) * cfg_physics.sliding.slidingco

    if not hasattr(state, "U"):
        state.U = tf.zeros(shape_3d)

    if not hasattr(state, "V"):
        state.V = tf.zeros(shape_3d)
