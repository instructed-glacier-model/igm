#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import tensorflow as tf

from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter


def _ensure_water_level(cfg, state):
    """Guarantee that state.water_level is a 2D tf.Variable matching topg.

    Resolution order, highest priority first:
      1. state.water_level if already loaded from the input NetCDF.
      2. cfg.processes.thk.default_sealevel, broadcast to a uniform 2D field.

    The default value of default_sealevel is a large negative sentinel so
    that the flotation term is inactive (lsurf = topg) unless the user
    either provides a 2D water_level field in the input or explicitly
    overrides default_sealevel in the config.
    """
    if hasattr(state, "water_level"):
        return

    state.water_level = tf.Variable(
        tf.ones_like(state.topg) * tf.cast(
            cfg.processes.thk.default_sealevel, state.topg.dtype
        ),
        trainable=False,
    )


def initialize(cfg, state):

    if not hasattr(state, "topg"):
        raise ValueError("The 'thk' module requires an initial topography ('state.topg') to be defined. Please define it through the preprocessing steps (not yet implemented)")

    _ensure_water_level(cfg, state)

    # define the lower ice surface (flotation constraint applied)
    state.lsurf = tf.maximum(
        state.topg,
        -cfg.processes.thk.ratio_density * state.thk + state.water_level,
    )

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

        # define the lower ice surface (flotation constraint applied)
        state.lsurf = tf.maximum(
            state.topg,
            -cfg.processes.thk.ratio_density * state.thk + state.water_level,
        )

        # define the upper ice surface
        state.usurf = state.lsurf + state.thk


def finalize(cfg, state):
    pass
