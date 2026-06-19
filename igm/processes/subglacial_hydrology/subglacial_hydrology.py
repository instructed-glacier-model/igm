#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Subglacial hydrology: effective pressure and (optionally) till water layer.

Computes state.effective_pressure (MPa) consumed by the Budd / Coulomb
sliding laws, and in mode 'till_storage' also evolves state.h_water_till
(m water) each time step.

The module is opt-in. It should not be activated when Weertman is the only
sliding law, since N cancels out of the Weertman cost.
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
    if cfg.processes.subglacial_hydrology.mode == "till_storage":
        if not hasattr(state, "h_water_till"):
            state.h_water_till = tf.zeros_like(state.thk)
        if not hasattr(state, "basal_melt_rate"):
            state.basal_melt_rate = tf.zeros_like(state.thk)
    state.effective_pressure = _compute_N(cfg, state)


def update(cfg: DictConfig, state: State) -> None:
    if hasattr(state, "logger"):
        state.logger.info(f"Update SUBGLACIAL_HYDROLOGY at time: {state.t.numpy()}")
    if cfg.processes.subglacial_hydrology.mode == "till_storage":
        state.h_water_till = till_storage.update_h_water_till(cfg, state)
    state.effective_pressure = _compute_N(cfg, state)


def finalize(cfg: DictConfig, state: State) -> None:
    pass


def _compute_N(cfg: DictConfig, state: State) -> tf.Tensor:
    """Return effective pressure (MPa) for the configured mode."""
    cfg_sh = cfg.processes.subglacial_hydrology
    mode = cfg_sh.mode

    if mode == "from_input":
        if not hasattr(state, "effective_pressure"):
            raise ValueError(
                "❌ subglacial_hydrology.mode = 'from_input' but "
                "state.effective_pressure is not set. Provide it as a "
                "variable in the input NetCDF, or pick a different mode."
            )
        return state.effective_pressure

    if mode == "till_storage":
        return till_storage.compute_N_MPa(cfg, state)

    cfg_phys = cfg.processes.iceflow.physics
    dtype = state.thk.dtype
    PA_TO_MPA = tf.cast(1.0e-6, dtype)
    rho_ice = tf.cast(cfg_phys.ice_density, dtype)
    g = tf.cast(cfg_phys.gravity_cst, dtype)
    p_ice = rho_ice * g * state.thk * PA_TO_MPA  # MPa

    if mode == "constant_one":
        N = tf.ones_like(state.thk)
    elif mode == "percentage":
        N = (1.0 - tf.cast(cfg_sh.percentage, dtype)) * p_ice
    elif mode == "ocean_connected":
        if not hasattr(state, "water_level"):
            raise ValueError(
                "❌ subglacial_hydrology.mode = 'ocean_connected' requires "
                "state.water_level. Activate the 'thk' module (which "
                "creates state.water_level) before 'subglacial_hydrology'."
            )
        rho_water = tf.cast(cfg_phys.water_density, dtype)
        ocean_depth = tf.maximum(state.water_level - state.topg, 0.0)
        N = p_ice - rho_water * g * ocean_depth * PA_TO_MPA
    else:
        raise ValueError(
            f"❌ Unknown subglacial_hydrology.mode: {mode!r}. "
            f"Valid modes: {VALID_MODES}."
        )

    return tf.maximum(N, tf.cast(cfg_sh.N_min, dtype))
