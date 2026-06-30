#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
thk
===

Stock mass-conservation update of the ice thickness `state.thk` from the
depth-averaged velocities (`state.ubar`, `state.vbar`) and the surface mass
balance (`state.smb`), using a slope-limited flux divergence. Surfaces are
updated afterwards (flotation when `state.water_level` exists).

Sibling file:
  utils.py   shared helpers (surface update)
"""

import os
import sys

import tensorflow as tf

from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter

# IGM loads this file under the bare name "thk" via SourceFileLoader, so the
# folder it lives in is not on sys.path. Add it so the sibling module is
# importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import update_surfaces


def initialize(cfg, state):

    if not hasattr(state, "topg"):
        raise ValueError(
            "The 'thk' module requires an initial topography ('state.topg')."
        )

    update_surfaces(cfg, state)


def update(cfg, state):

    if state.it >= 0:
        if hasattr(state, "logger"):
            state.logger.info(
                "Ice thickness equation at time : " + str(state.t.numpy())
            )

        p = cfg.processes.thk

        state.divflux = compute_divflux_slope_limiter(
            state.ubar,
            state.vbar,
            state.thk,
            state.dx,
            state.dx,
            state.dt,
            slope_type=p.slope_type,
        )
        if not hasattr(state, "smb"):
            state.smb = tf.zeros_like(state.thk)
        state.thk = tf.maximum(state.thk + state.dt * (state.smb - state.divflux), 0)

        update_surfaces(cfg, state)


def finalize(cfg, state):
    pass
