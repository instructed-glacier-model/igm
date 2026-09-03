#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Peaceman--Rachford ADI thickness evolution.

With frozen velocities, split the first-order upwind flux divergence into
directional linear operators ``D = D_x + D_y``. For ``h = dt / 2``, the
standard two-stage Peaceman--Rachford step is

    (I + h D_x) H*       = (I - h D_y) H_old + h smb,
    (I + h D_y) H_new    = (I - h D_x) H*    + h smb.

Each left-hand side is tridiagonal along one grid direction. All rows or
columns are solved as a batch with TensorFlow's XLA-compatible tridiagonal
solver. The complete two-stage step is one compiled graph and uses no Python
loops, NumPy operations, or device-to-host synchronization.

PR is second-order and A-stable for each one-dimensional split operator, but
it is not L-stable or monotonicity preserving. At extremely large advective
CFL numbers it can therefore remain bounded while developing oscillations;
the fully implicit theta backend is preferable when strong damping is wanted.
"""

from typing import NamedTuple

import tensorflow as tf


class ADIStepResult(NamedTuple):
    """Result of one Peaceman--Rachford step."""

    thickness: tf.Tensor
    divflux: tf.Tensor
    transport_divflux: tf.Tensor
    nonnegative_correction_volume: tf.Tensor


class _DirectionalCoefficients(NamedTuple):
    """Coefficients of the directional upwind divergence operators."""

    x_diagonal: tf.Tensor
    west: tf.Tensor
    east: tf.Tensor
    y_diagonal: tf.Tensor
    north: tf.Tensor
    south: tf.Tensor


# These helpers intentionally remain undecorated: _peaceman_rachford_step
# traces them inline into one XLA graph, avoiding nested graph-call boundaries.
def _build_directional_coefficients(ubar, vbar, dx):
    """Build ``D_x`` and ``D_y`` from velocities interpolated to faces."""
    u_face = tf.concat(
        [ubar[:, 0:1], 0.5 * (ubar[:, :-1] + ubar[:, 1:]), ubar[:, -1:]],
        axis=1,
    )
    v_face = tf.concat(
        [vbar[0:1, :], 0.5 * (vbar[:-1, :] + vbar[1:, :]), vbar[-1:, :]],
        axis=0,
    )

    inverse_dx = tf.math.reciprocal(dx)
    u_left, u_right = u_face[:, :-1], u_face[:, 1:]
    v_north, v_south = v_face[:-1, :], v_face[1:, :]

    west = inverse_dx * tf.nn.relu(u_left)
    east = inverse_dx * tf.nn.relu(-u_right)
    north = inverse_dx * tf.nn.relu(v_north)
    south = inverse_dx * tf.nn.relu(-v_south)

    return _DirectionalCoefficients(
        inverse_dx * (tf.nn.relu(u_right) + tf.nn.relu(-u_left)),
        west,
        east,
        inverse_dx * (tf.nn.relu(v_south) + tf.nn.relu(-v_north)),
        north,
        south,
    )


def _apply_x(field, coefficients):
    """Apply the x-direction upwind divergence with zero exterior thickness."""
    west_neighbor = tf.pad(field[:, :-1], [[0, 0], [1, 0]])
    east_neighbor = tf.pad(field[:, 1:], [[0, 0], [0, 1]])
    return (
        coefficients.x_diagonal * field
        - coefficients.west * west_neighbor
        - coefficients.east * east_neighbor
    )


def _apply_y(field, coefficients):
    """Apply the y-direction upwind divergence with zero exterior thickness."""
    north_neighbor = tf.pad(field[:-1, :], [[1, 0], [0, 0]])
    south_neighbor = tf.pad(field[1:, :], [[0, 1], [0, 0]])
    return (
        coefficients.y_diagonal * field
        - coefficients.north * north_neighbor
        - coefficients.south * south_neighbor
    )


def _solve_x(rhs, coefficients, half_dt):
    """Solve all systems ``(I + half_dt D_x) x = rhs`` as a row batch."""
    zero_column = tf.zeros_like(rhs[:, 0:1])
    lower = tf.concat(
        [zero_column, -half_dt * coefficients.west[:, 1:]], axis=1
    )
    upper = tf.concat(
        [-half_dt * coefficients.east[:, :-1], zero_column], axis=1
    )
    diagonal = 1.0 + half_dt * coefficients.x_diagonal
    return tf.linalg.tridiagonal_solve(
        (upper, diagonal, lower),
        rhs,
        diagonals_format="sequence",
        partial_pivoting=False,
    )


def _solve_y(rhs, coefficients, half_dt):
    """Solve all systems ``(I + half_dt D_y) x = rhs`` as a column batch."""
    zero_row = tf.zeros_like(rhs[0:1, :])
    lower = tf.transpose(
        tf.concat(
            [zero_row, -half_dt * coefficients.north[1:, :]], axis=0
        )
    )
    upper = tf.transpose(
        tf.concat(
            [-half_dt * coefficients.south[:-1, :], zero_row], axis=0
        )
    )
    diagonal = tf.transpose(1.0 + half_dt * coefficients.y_diagonal)
    solution = tf.linalg.tridiagonal_solve(
        (upper, diagonal, lower),
        tf.transpose(rhs),
        diagonals_format="sequence",
        partial_pivoting=False,
    )
    return tf.transpose(solution)


@tf.function(autograph=False, reduce_retracing=True, jit_compile=True)
def _peaceman_rachford_step(ubar, vbar, thickness, dx, dt, smb):
    """Execute one tensor-only, XLA-compiled Peaceman--Rachford step."""
    coefficients = _build_directional_coefficients(ubar, vbar, dx)
    half_dt = 0.5 * dt
    half_source = half_dt * smb

    # Stage 1: x implicit, y explicit.
    first_rhs = thickness - half_dt * _apply_y(thickness, coefficients)
    intermediate = _solve_x(first_rhs + half_source, coefficients, half_dt)

    # Stage 2: y implicit, x explicit. Together with stage 1 this is the
    # symmetric, second-order Peaceman--Rachford factorization.
    second_rhs = intermediate - half_dt * _apply_x(intermediate, coefficients)
    thickness_solved = _solve_y(
        second_rhs + half_source, coefficients, half_dt
    )

    # PR is not monotone at high CFL. Keep both the raw transport divergence
    # and the conservative effective divergence after the physical H >= 0
    # projection, as is done by the other implicit backends.
    transport_divflux = smb - tf.math.divide_no_nan(
        thickness_solved - thickness, dt
    )
    thickness_new = tf.nn.relu(thickness_solved)
    correction = thickness_new - thickness_solved
    nonnegative_correction_volume = tf.reduce_sum(correction) * dx * dx
    divflux = smb - tf.math.divide_no_nan(thickness_new - thickness, dt)
    return ADIStepResult(
        thickness_new,
        divflux,
        transport_divflux,
        nonnegative_correction_volume,
    )


def solve(state, smb):
    """Adapt state tensors to the compiled Peaceman--Rachford kernel."""
    dtype = state.thk.dtype
    return _peaceman_rachford_step(
        tf.cast(state.ubar, dtype),
        tf.cast(state.vbar, dtype),
        state.thk,
        tf.cast(state.dx, dtype),
        tf.cast(state.dt, dtype),
        tf.cast(smb, dtype),
    )


def _validate_config(cfg):
    """Validate options relevant to the ADI backend."""
    p = cfg.processes.thk
    if p.calving_front:
        raise ValueError(
            "cfg.processes.thk.scheme: adi cannot currently be combined with "
            "calving_front: true."
        )


def initialize(cfg, state):
    """Validate the Peaceman--Rachford backend configuration."""
    _validate_config(cfg)
    state.thk_transport_divflux = tf.zeros_like(state.thk)
    state.thk_nonnegative_correction_volume = tf.zeros(
        [], dtype=state.thk.dtype
    )


def update(cfg, state):
    """Advance thickness by one Peaceman--Rachford ADI step."""
    if not hasattr(state, "smb"):
        state.smb = tf.zeros_like(state.thk)

    result = solve(state, state.smb)
    state.thk = result.thickness
    state.divflux = result.divflux
    state.thk_transport_divflux = result.transport_divflux
    state.thk_nonnegative_correction_volume = (
        result.nonnegative_correction_volume
    )
