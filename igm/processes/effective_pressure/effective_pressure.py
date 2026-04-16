#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Compute and maintain state.effective_pressure (MPa).

This module is the single source of truth for the basal effective
pressure N consumed by the Budd and Coulomb sliding laws (via
fieldin["effective_pressure"]). Several closure modes are available;
which one to pick is a config choice.

The module is opt-in: if Weertman is the only sliding law in use, it
should not be activated.
"""

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State


VALID_MODES = ("constant_one", "percentage", "ocean_connected", "from_input")


def initialize(cfg: DictConfig, state: State) -> None:
    _compute(cfg, state)


def update(cfg: DictConfig, state: State) -> None:
    if hasattr(state, "logger"):
        state.logger.info(f"Update EFFECTIVE_PRESSURE at time: {state.t.numpy()}")
    _compute(cfg, state)


def finalize(cfg: DictConfig, state: State) -> None:
    pass


def _compute(cfg: DictConfig, state: State) -> None:
    """Write state.effective_pressure (MPa) according to the chosen mode."""

    cfg_ep = cfg.processes.effective_pressure
    mode = cfg_ep.mode

    if mode not in VALID_MODES:
        raise ValueError(
            f"❌ Unknown effective_pressure.mode: {mode!r}. "
            f"Valid modes: {VALID_MODES}."
        )

    if mode == "from_input":
        if not hasattr(state, "effective_pressure"):
            raise ValueError(
                "❌ effective_pressure.mode = 'from_input' but "
                "state.effective_pressure is not set. Provide it as a "
                "variable in the input NetCDF, or pick a different mode."
            )
        return

    cfg_phys = cfg.processes.iceflow.physics
    rho_i = tf.cast(cfg_phys.ice_density, state.thk.dtype)
    rho_w = tf.cast(cfg_phys.water_density, state.thk.dtype)
    g = tf.cast(cfg_phys.gravity_cst, state.thk.dtype)

    p_ice = rho_i * g * state.thk * 1.0e-6  # MPa

    if mode == "constant_one":
        N = tf.ones_like(state.thk)
    elif mode == "percentage":
        pct = tf.cast(cfg_ep.percentage, state.thk.dtype)
        N = (1.0 - pct) * p_ice
    elif mode == "ocean_connected":
        if not hasattr(state, "water_level"):
            raise ValueError(
                "❌ effective_pressure.mode = 'ocean_connected' requires "
                "state.water_level. Activate the 'thk' module (which "
                "creates state.water_level) before 'effective_pressure'."
            )
        Dw = tf.maximum(state.water_level - state.topg, 0.0)
        p_water = rho_w * g * Dw * 1.0e-6  # MPa
        N = p_ice - p_water

    N_min = tf.cast(cfg_ep.N_min, state.thk.dtype)
    state.effective_pressure = tf.maximum(N, N_min)
