#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Slope-limited forward-Euler thickness evolution backend.

:func:`divergence` is the backend's flux divergence with the configured
boundaries and active domain. The calving-front schemes call the same
limiter kernels directly (one fused graph with their routing), with the
boundary options cached here, so they honour the same boundary contract.
"""

import math
from typing import Optional

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.grad.compute_divflux_slope_limiter import (
    compute_divflux_slope_limiter,
    compute_divflux_slope_limiter_boundaries,
)

from ..domains import face_masks
from ..sources import mass_balance

SUPPORTED_BOUNDARY_MODES = ("dirichlet", "symmetric", "zero")
SUPPORTS_ACTIVE_DOMAIN = True
SUPPORTS_DIVFLUX_SMOOTHING = True


def _ghost(thk: tf.Tensor, side: str, mode: str) -> Optional[tf.Tensor]:
    """Exterior thickness of a Dirichlet side: its initial edge cells."""
    if mode != "dirichlet":
        return None
    edge = {
        "left": thk[:, :1],
        "right": thk[:, -1:],
        "top": thk[:1, :],
        "bottom": thk[-1:, :],
    }[side]
    return tf.identity(edge)


def initialize(cfg: DictConfig, state: State) -> None:
    """Validate and cache static options (and the Dirichlet ghost thickness)."""
    p = cfg.processes.thk
    slope_type = str(getattr(p, "slope_type", "superbee")).strip().lower()
    if slope_type not in ("godunov", "minmod", "superbee"):
        raise ValueError(
            "cfg.processes.thk.slope_type must be godunov, minmod, or "
            f"superbee; got {slope_type!r}."
        )
    smooth_sigma = float(getattr(p, "divflux_smooth_sigma", 0.0))
    if not math.isfinite(smooth_sigma) or smooth_sigma < 0.0:
        raise ValueError(
            "cfg.processes.thk.divflux_smooth_sigma must be finite and " "nonnegative."
        )
    boundaries = state.thk_components.boundaries
    thk = tf.convert_to_tensor(state.thk)
    state.thk_components.transport_options = {
        "slope_type": slope_type,
        "smooth_sigma": smooth_sigma,
        "has_boundary_condition": any(mode != "zero" for mode in boundaries),
        "left_symmetric": boundaries.left == "symmetric",
        "right_symmetric": boundaries.right == "symmetric",
        "top_symmetric": boundaries.top == "symmetric",
        "bottom_symmetric": boundaries.bottom == "symmetric",
        "left_ghost": _ghost(thk, "left", boundaries.left),
        "right_ghost": _ghost(thk, "right", boundaries.right),
        "top_ghost": _ghost(thk, "top", boundaries.top),
        "bottom_ghost": _ghost(thk, "bottom", boundaries.bottom),
    }


def divergence(
    state: State,
    thk: tf.Tensor,
    ubar: tf.Tensor,
    vbar: tf.Tensor,
    first_order_mask: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """Flux divergence of ``thk`` with the cached boundaries and active domain.

    ``first_order_mask`` selects cells whose outgoing fluxes are first-order
    upwind (zero reconstruction slope), as the calving fronts do next to the
    ice front in their own fused kernel.
    """
    options = state.thk_components.transport_options
    active_mask = getattr(state, "thk_active_mask", None)
    if active_mask is None:
        x_face_mask = y_face_mask = None
    else:
        x_face_mask, y_face_mask = face_masks(active_mask)

    if not options["has_boundary_condition"]:
        return compute_divflux_slope_limiter(
            ubar,
            vbar,
            thk,
            state.dx,
            state.dx,
            state.dt,
            slope_type=options["slope_type"],
            smooth_sigma=options["smooth_sigma"],
            x_face_mask=x_face_mask,
            y_face_mask=y_face_mask,
            first_order_mask=first_order_mask,
        )
    return compute_divflux_slope_limiter_boundaries(
        ubar,
        vbar,
        thk,
        state.dx,
        state.dx,
        state.dt,
        slope_type=options["slope_type"],
        smooth_sigma=options["smooth_sigma"],
        x_face_mask=x_face_mask,
        y_face_mask=y_face_mask,
        left=options["left_symmetric"],
        right=options["right_symmetric"],
        top=options["top_symmetric"],
        bottom=options["bottom_symmetric"],
        first_order_mask=first_order_mask,
        left_ghost=options["left_ghost"],
        right_ghost=options["right_ghost"],
        top_ghost=options["top_ghost"],
        bottom_ghost=options["bottom_ghost"],
    )


def update(cfg: DictConfig, state: State) -> None:
    """Advance ice thickness by one slope-limited forward-Euler step."""
    del cfg
    source = mass_balance(state)
    state.divflux = divergence(state, state.thk, state.ubar, state.vbar)

    active_mask = getattr(state, "thk_active_mask", None)
    if active_mask is None:
        # Exact historical update path: no mask allocation, no tf.where.
        state.thk = tf.maximum(state.thk + state.dt * (source - state.divflux), 0.0)
    else:
        candidate = tf.maximum(
            state.thk + state.dt * (tf.where(active_mask, source, 0.0) - state.divflux),
            0.0,
        )
        state.thk = tf.where(active_mask, candidate, state.thk)
