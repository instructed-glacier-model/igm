#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Compute and maintain state.effective_pressure (MPa).

This module is the single source of truth for the basal effective
pressure N consumed by the Budd and Coulomb sliding laws (via
fieldin["effective_pressure"]). Several closure modes are available;
which one to pick is a config choice.

Unit convention: N is in MPa to match the rest of the iceflow
quantities (slidingco, gravity, viscosity costs). Literature values
for N_ref are typically O(1 MPa) (Brondex et al. 2019, Pollard &
DeConto 2012, Pattyn 2017).

The module is opt-in: if Weertman is the only sliding law in use, it
should not be activated.
"""

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from . import till_storage


VALID_MODES = (
    "constant_one",
    "percentage",
    "ocean_connected",
    "from_input",
    "till_storage",
)


def initialize(cfg: DictConfig, state: State) -> None:
    state.effective_pressure = _compute_N(cfg, state)


def update(cfg: DictConfig, state: State) -> None:
    if hasattr(state, "logger"):
        state.logger.info(f"Update EFFECTIVE_PRESSURE at time: {state.t.numpy()}")
    state.effective_pressure = _compute_N(cfg, state)


def finalize(cfg: DictConfig, state: State) -> None:
    pass


def _compute_N(cfg: DictConfig, state: State) -> tf.Tensor:
    """Return state.effective_pressure (MPa) for the configured mode."""
    cfg_ep = cfg.processes.effective_pressure
    mode = cfg_ep.mode

    if mode == "from_input":
        if not hasattr(state, "effective_pressure"):
            raise ValueError(
                "❌ effective_pressure.mode = 'from_input' but "
                "state.effective_pressure is not set. Provide it as a "
                "variable in the input NetCDF, or pick a different mode."
            )
        return state.effective_pressure

    if mode == "till_storage":
        return till_storage.compute_N_MPa(cfg, state)

    # Remaining modes all derive N from the ice-overburden pressure.
    cfg_phys = cfg.processes.iceflow.physics
    dtype = state.thk.dtype
    PA_TO_MPA = tf.cast(1.0e-6, dtype)
    rho_i = tf.cast(cfg_phys.ice_density, dtype)
    g = tf.cast(cfg_phys.gravity_cst, dtype)
    p_ice = rho_i * g * state.thk * PA_TO_MPA  # MPa

    if mode == "constant_one":
        N = tf.ones_like(state.thk)
    elif mode == "percentage":
        N = (1.0 - tf.cast(cfg_ep.percentage, dtype)) * p_ice
    elif mode == "ocean_connected":
        if not hasattr(state, "water_level"):
            raise ValueError(
                "❌ effective_pressure.mode = 'ocean_connected' requires "
                "state.water_level. Activate the 'thk' module (which "
                "creates state.water_level) before 'effective_pressure'."
            )
        rho_w = tf.cast(cfg_phys.water_density, dtype)
        Dw = tf.maximum(state.water_level - state.topg, 0.0)
        N = p_ice - rho_w * g * Dw * PA_TO_MPA
    else:
        raise ValueError(
            f"❌ Unknown effective_pressure.mode: {mode!r}. "
            f"Valid modes: {VALID_MODES}."
        )

    return tf.maximum(N, tf.cast(cfg_ep.N_min, dtype))
