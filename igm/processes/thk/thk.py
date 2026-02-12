#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter


def _get_sealevel(cfg: DictConfig, state: State) -> tf.Tensor:
    """Get sea level from state or use default from config."""
    return getattr(state, "sealevel", cfg.processes.thk.default_sealevel)


def _get_smb(cfg: DictConfig, state: State) -> tf.Tensor:
    """Get surface mass balance from state or use zero tensor."""
    return getattr(state, "smb", tf.zeros_like(state.thk))


def _compute_surfs(cfg: DictConfig, state: State) -> None:
    """Update lower and upper ice surfaces."""
    delta = cfg.processes.thk.ratio_density
    sealevel = _get_sealevel(cfg, state)
    state.lsurf = tf.maximum(state.topg, -delta * state.thk + sealevel)
    state.usurf = state.lsurf + state.thk


def initialize(cfg: DictConfig, state: State) -> None:
    if not hasattr(state, "topg"):
        raise ValueError(
            "The 'thk' module requires an initial topography ('state.topg') to be defined. "
            "Please define it through the preprocessing steps (not yet implemented)"
        )

    _compute_surfs(cfg, state)


def update(cfg: DictConfig, state: State) -> None:
    if state.it >= 0:
        if hasattr(state, "logger"):
            state.logger.info(f"Ice thickness equation at time: {state.t.numpy()}")

        # Compute the divergence of the flux
        state.divflux = compute_divflux_slope_limiter(
            state.ubar,
            state.vbar,
            state.thk,
            state.dx,
            state.dx,
            state.dt,
            slope_type=cfg.processes.thk.slope_type,
        )

        # Get surface mass balance
        state.smb = _get_smb(cfg, state)

        # Forward Euler with projection to keep ice thickness non-negative
        state.thk = tf.maximum(state.thk + state.dt * (state.smb - state.divflux), 0)

        _compute_surfs(cfg, state)


def finalize(cfg: DictConfig, state: State) -> None:
    pass
