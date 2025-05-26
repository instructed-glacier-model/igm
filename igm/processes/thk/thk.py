#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import tensorflow as tf

from igm.utils.gradient.compute_divflux_slope_limiter import compute_divflux_slope_limiter


def initialize(cfg, state):

    if not hasattr(state, "topg"):
        raise ValueError("The 'thk' module requires an initial topography ('state.topg') to be defined. Please define it through the preprocessing steps (not yet implemented)")
        
    # define the lower ice surface
    if hasattr(state, "sealevel"):
        # ! This is not clear which modules provides state.topg and allows us to use state.thk!!!
        state.lsurf = tf.maximum(state.topg,-cfg.processes.thk.ratio_density*state.thk + state.sealevel)
    else:
        state.lsurf = tf.maximum(state.topg,-cfg.processes.thk.ratio_density*state.thk + cfg.processes.thk.default_sealevel)

    # define the upper ice surface
    state.usurf = state.lsurf + state.thk


def update(cfg, state):

    if state.it >= 0:
        if hasattr(state, "logger"):
            state.logger.info(
                "Ice thickness equation at time : " + str(state.t.numpy())
            )

        # compute the divergence of the flux
        state.divflux = compute_divflux_slope_limiter(
            state.ubar, state.vbar, state.thk, state.dx, state.dx, state.dt, slope_type=cfg.processes.thk.slope_type
        )

        # if not smb model is given, set smb to zero
        if not hasattr(state, "smb"):
            state.smb = tf.zeros_like(state.thk)

        # Forward Euler with projection to keep ice thickness non-negative
        state.thk = tf.maximum(state.thk + state.dt * (state.smb - state.divflux), 0)

        # define the lower ice surface
        if hasattr(state, "sealevel"):
            state.lsurf = tf.maximum(state.topg,-cfg.processes.thk.ratio_density*state.thk + state.sealevel)
        else:
            state.lsurf = tf.maximum(state.topg,-cfg.processes.thk.ratio_density*state.thk + cfg.processes.thk.default_sealevel)

        # define the upper ice surface
        state.usurf = state.lsurf + state.thk


def finalize(cfg, state):
    pass
