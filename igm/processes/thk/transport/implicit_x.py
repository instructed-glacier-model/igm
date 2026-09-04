#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Batched tridiagonal theta solver for independent x-flowlines.

This backend advances

    dH/dt + d(u H)/dx = SMB

and intentionally contains no y-transport term. Every row is one independent
flowline, and all rows are solved together on the GPU by parallel cyclic
reduction (PCR), exposing all flowlines and x-nodes at each of only
``ceil(log2(Nx))`` dependent stages.
"""

from typing import NamedTuple

import tensorflow as tf

from igm.utils.math.tridiagonal import solve_tridiagonal_pcr

from .. import boundary


SUPPORTED_BOUNDARY_MODES = ("zero", "symmetric")


class ImplicitXStepResult(NamedTuple):
    """Thickness and conservation diagnostics from one x-flowline step."""

    thickness: tf.Tensor
    divflux: tf.Tensor
    transport_divflux: tf.Tensor
    nonnegative_correction_volume: tf.Tensor


class _XCoefficients(NamedTuple):
    """Positive magnitudes of a first-order upwind x-divergence stencil."""

    diagonal: tf.Tensor
    west: tf.Tensor
    east: tf.Tensor


def _build_x_coefficients(ubar, dx, left, right):
    """Build ``D_x`` such that D_x(H)=diag*H-west*H_W-east*H_E."""
    u_face = boundary.x_face_velocities(ubar, left, right)
    u_left, u_right = u_face[:, :-1], u_face[:, 1:]
    inverse_dx = tf.math.reciprocal(dx)

    west = inverse_dx * tf.nn.relu(u_left)
    east = inverse_dx * tf.nn.relu(-u_right)
    diagonal = inverse_dx * (tf.nn.relu(u_right) + tf.nn.relu(-u_left))

    # Both supported policies are non-periodic: ghost-cell couplings are
    # discarded. Open-boundary outflow remains in diagonal, while symmetric
    # face velocities are already zero.
    zero_column = tf.zeros_like(west[:, :1])
    west = tf.concat([zero_column, west[:, 1:]], axis=1)
    east = tf.concat([east[:, :-1], zero_column], axis=1)
    return _XCoefficients(diagonal, west, east)


def _apply_x(field, coefficients):
    west_neighbor = tf.pad(field[:, :-1], [[0, 0], [1, 0]])
    east_neighbor = tf.pad(field[:, 1:], [[0, 0], [0, 1]])
    return (
        coefficients.diagonal * field
        - coefficients.west * west_neighbor
        - coefficients.east * east_neighbor
    )


@tf.function(autograph=False, jit_compile=True)
def _solve_theta_x_step(
    ubar,
    thickness,
    dx,
    dt,
    smb,
    theta,
    left="zero",
    right="zero",
):
    """Advance all independent x-flowlines in one GPU/XLA kernel."""
    coefficients = _build_x_coefficients(ubar, dx, left, right)
    divflux_old = _apply_x(thickness, coefficients)
    rhs = thickness - dt * (1.0 - theta) * divflux_old + dt * smb

    scale = dt * theta
    diagonal = 1.0 + scale * coefficients.diagonal
    lower = -scale * coefficients.west
    upper = -scale * coefficients.east
    thickness_solved = solve_tridiagonal_pcr(lower, diagonal, upper, rhs)

    divflux_new = _apply_x(thickness_solved, coefficients)
    transport_divflux = theta * divflux_new + (1.0 - theta) * divflux_old
    thickness_new = tf.nn.relu(thickness_solved)
    correction = thickness_new - thickness_solved
    nonnegative_correction_volume = tf.reduce_sum(correction) * dx * dx
    divflux = smb - tf.math.divide_no_nan(thickness_new - thickness, dt)
    return ImplicitXStepResult(
        thickness_new,
        divflux,
        transport_divflux,
        nonnegative_correction_volume,
    )


def _solve_with_options(state, smb, options):
    """Adapt state tensors and cached options to the compiled kernel."""
    dtype = state.thk.dtype
    return _solve_theta_x_step(
        tf.cast(state.ubar, dtype),
        state.thk,
        tf.cast(state.dx, dtype),
        tf.cast(state.dt, dtype),
        tf.cast(smb, dtype),
        tf.cast(options["theta"], dtype),
        options["left"],
        options["right"],
    )


def _options(cfg, state, boundaries=None):
    p = cfg.processes.thk
    if boundaries is None:
        boundaries = boundary.get_boundary_conditions(cfg)
    boundary.validate_backend(boundaries, SUPPORTED_BOUNDARY_MODES, "implicit_x")
    theta = float(p.implicit.theta)
    if not 0.5 <= theta <= 1.0:
        raise ValueError(
            "cfg.processes.thk.implicit.theta must be between 0.5 and 1.0, "
            f"got {theta}."
        )
    if state.thk.shape.rank != 2:
        raise ValueError("scheme: implicit_x requires a rank-two thickness field.")
    return {
        "theta": theta,
        "left": boundaries.left,
        "right": boundaries.right,
    }


def solve(state, cfg, smb):
    """Solve one step, parsing configuration for direct/test callers."""
    return _solve_with_options(state, smb, _options(cfg, state))


def initialize(cfg, state):
    """Validate the x-flowline backend and initialize its diagnostics."""
    state.thk_components.transport_options = _options(
        cfg, state, state.thk_components.boundaries
    )
    state.thk_transport_divflux = tf.zeros_like(state.thk)
    state.thk_nonnegative_correction_volume = tf.zeros([], dtype=state.thk.dtype)


def update(cfg, state):
    """Advance each row independently along x; ``state.vbar`` is not used."""
    del cfg
    if not hasattr(state, "smb"):
        state.smb = tf.zeros_like(state.thk)
    result = _solve_with_options(
        state, state.smb, state.thk_components.transport_options
    )
    state.thk = result.thickness
    state.divflux = result.divflux
    state.thk_transport_divflux = result.transport_divflux
    state.thk_nonnegative_correction_volume = result.nonnegative_correction_volume
