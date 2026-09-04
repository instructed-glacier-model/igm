#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Slope-limited forward-Euler thickness evolution backend."""

import math

import tensorflow as tf

from igm.utils.grad.compute_divflux_slope_limiter import (
    compute_divflux_slope_limiter,
    compute_divflux_slope_limiter_symmetric,
)

from ..domains import face_masks


SUPPORTED_BOUNDARY_MODES = ("zero", "symmetric")
SUPPORTS_ACTIVE_DOMAIN = True
SUPPORTS_DIVFLUX_SMOOTHING = True


def initialize(cfg, state):
    """Validate and cache static options without creating device tensors."""
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
            "cfg.processes.thk.divflux_smooth_sigma must be finite and "
            "nonnegative."
        )
    boundaries = state.thk_components.boundaries
    state.thk_components.transport_options = {
        "slope_type": slope_type,
        "smooth_sigma": smooth_sigma,
        "has_symmetric_boundary": "symmetric" in boundaries,
        "left_symmetric": boundaries.left == "symmetric",
        "right_symmetric": boundaries.right == "symmetric",
        "top_symmetric": boundaries.top == "symmetric",
        "bottom_symmetric": boundaries.bottom == "symmetric",
    }


def update(cfg, state):
    """Advance ice thickness by one slope-limited forward-Euler step."""
    del cfg
    if not hasattr(state, "smb"):
        state.smb = tf.zeros_like(state.thk)

    options = state.thk_components.transport_options
    active_mask = getattr(state, "thk_active_mask", None)
    if active_mask is None:
        x_face_mask = y_face_mask = None
    else:
        x_face_mask, y_face_mask = face_masks(active_mask)

    if options["has_symmetric_boundary"]:
        state.divflux = compute_divflux_slope_limiter_symmetric(
            state.ubar,
            state.vbar,
            state.thk,
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
        )
    else:
        state.divflux = compute_divflux_slope_limiter(
            state.ubar,
            state.vbar,
            state.thk,
            state.dx,
            state.dx,
            state.dt,
            slope_type=options["slope_type"],
            smooth_sigma=options["smooth_sigma"],
            x_face_mask=x_face_mask,
            y_face_mask=y_face_mask,
        )

    if active_mask is None:
        # Exact historical update path: no mask allocation, no tf.where.
        state.thk = tf.maximum(
            state.thk + state.dt * (state.smb - state.divflux), 0.0
        )
    else:
        candidate = tf.maximum(
            state.thk
            + state.dt
            * (tf.where(active_mask, state.smb, 0.0) - state.divflux),
            0.0,
        )
        state.thk = tf.where(active_mask, candidate, state.thk)
