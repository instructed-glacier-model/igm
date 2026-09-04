#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Conservative flux-form semi-Lagrangian thickness evolution.

This backend combines three properties which are useful for ice-thickness
transport: exact flux-form conservation, a monotone piecewise-linear
reconstruction, and no advective CFL stability restriction.  A face flux is
the integral of the reconstructed thickness over the complete swept interval,
so a characteristic may cross any number of cells in one step.

The two-dimensional update uses symmetric Strang splitting.  Its leading
direction alternates between steps to avoid a persistent directional bias.
Uniform translation therefore needs no internal substeps, irrespective of its
CFL number.  For spatially varying flow, device-side substepping instead
limits a dimensionless velocity-deformation number; this keeps neighbouring
departure faces ordered, which is the condition needed for monotonicity.

The complete update is a single XLA-compiled TensorFlow graph.  It contains no
NumPy operations, Python time loop, or device-to-host synchronization.
"""

import math
from typing import NamedTuple

import tensorflow as tf

from .. import boundary


_DEFAULT_MAX_DEFORMATION = 0.5
_DEFAULT_MAX_SUBSTEPS = 128
SUPPORTED_BOUNDARY_MODES = ("zero", "symmetric", "periodic")


class FFSLStepResult(NamedTuple):
    """Result and device-resident diagnostics from one FFSL step."""

    thickness: tf.Tensor
    divflux: tf.Tensor
    substeps: tf.Tensor
    required_substeps: tf.Tensor
    max_deformation: tf.Tensor
    substep_limit_reached: tf.Tensor
    source_limiter_volume: tf.Tensor
    step_accepted: tf.Tensor


# The helpers below are intentionally undecorated.  They are traced inline in
# _ffsl_step, leaving a single graph/XLA compilation boundary for the backend.
def _minimum_modulus(left, right):
    """Return the minmod-limited value of two tensors."""
    same_sign = left * right > 0.0
    return tf.where(
        same_sign,
        tf.sign(left) * tf.minimum(tf.abs(left), tf.abs(right)),
        tf.zeros_like(left),
    )


def _superbee_slope(lines, left, right):
    """Return monotone Superbee slopes for a batch of one-dimensional lines."""
    padded = boundary.pad_lines(lines, left, right)
    backward = lines - padded[:, :-2]
    forward = padded[:, 2:] - lines
    first = _minimum_modulus(forward, 2.0 * backward)
    second = _minimum_modulus(backward, 2.0 * forward)
    same_sign = first * second > 0.0
    return tf.where(
        same_sign,
        tf.sign(first) * tf.maximum(tf.abs(first), tf.abs(second)),
        tf.zeros_like(first),
    )


def _reconstruction_integral_base(lines, slopes, prefix, position):
    """Integrate reconstruction inside one copy of the finite domain."""
    line_length = tf.shape(lines)[1]
    clipped = tf.clip_by_value(
        position, 0.0, tf.cast(line_length, lines.dtype)
    )
    cell = tf.minimum(
        tf.cast(tf.floor(clipped), tf.int32), line_length - 1
    )
    fraction = clipped - tf.cast(cell, lines.dtype)

    cell_average = tf.gather(lines, cell, axis=1, batch_dims=1)
    cell_slope = tf.gather(slopes, cell, axis=1, batch_dims=1)
    integral_before = tf.gather(prefix, cell, axis=1, batch_dims=1)
    return (
        integral_before
        + fraction * cell_average
        + 0.5 * (fraction * fraction - fraction) * cell_slope
    )


def _reconstruction_integral(lines, slopes, prefix, position, left):
    """Integrate from zero to a possibly out-of-domain departure position."""
    if left != "periodic":
        # A zero exterior contains no ice. At a symmetric boundary the normal
        # face is fixed and ordered departure faces remain inside the domain.
        return _reconstruction_integral_base(
            lines, slopes, prefix, position
        )

    line_length = tf.cast(tf.shape(lines)[1], lines.dtype)
    cycles = tf.floor(position / line_length)
    wrapped_position = position - cycles * line_length
    total_mass = prefix[:, -1:]
    return cycles * total_mass + _reconstruction_integral_base(
        lines, slopes, prefix, wrapped_position
    )


def _repair_nonnegative_roundoff(thickness):
    """Remove tiny negative roundoff while retaining the total flux budget."""
    target_mass = tf.reduce_sum(thickness)
    positive = tf.nn.relu(thickness)
    positive_mass = tf.reduce_sum(positive)
    scale = tf.math.divide_no_nan(tf.nn.relu(target_mass), positive_mass)
    return positive * scale


def _transport_lines(lines, face_velocity, dt, dx, left, right):
    """Conservatively transport a batch of lines through arbitrary face CFLs."""
    slopes = _superbee_slope(lines, left, right)
    prefix = tf.concat(
        [tf.zeros_like(lines[:, :1]), tf.cumsum(lines, axis=1)], axis=1
    )
    line_length = tf.shape(lines)[1]
    faces = tf.cast(tf.range(line_length + 1)[None, :], lines.dtype)
    departure = faces - face_velocity * dt / dx

    # swept_mass is signed: positive velocities move mass to the right and
    # negative velocities move it to the left.  Clipping the reconstruction
    # integral implements a zero-thickness exterior at open boundaries.
    swept_mass = prefix - _reconstruction_integral(
        lines, slopes, prefix, departure, left
    )
    return lines + swept_mass[:, :-1] - swept_mass[:, 1:]


def _transport_x(thickness, u_face, dt, dx, left, right):
    return _transport_lines(thickness, u_face, dt, dx, left, right)


def _transport_y(thickness, v_face, dt, dx, top, bottom):
    return tf.transpose(
        _transport_lines(
            tf.transpose(thickness),
            tf.transpose(v_face),
            dt,
            dx,
            top,
            bottom,
        )
    )


def _maximum_absolute_difference(field, axis):
    """Maximum adjacent difference, also valid for a one-cell dimension."""
    if axis == 0:
        difference = field[1:, :] - field[:-1, :]
    else:
        difference = field[:, 1:] - field[:, :-1]
    values = tf.concat(
        [tf.reshape(tf.abs(difference), (-1,)), tf.zeros((1,), field.dtype)],
        axis=0,
    )
    return tf.reduce_max(values)


def _velocity_deformation(ubar, vbar, u_face, v_face, dt, dx):
    """Return an upper bound on ``dt * ||grad(velocity)||_infinity``."""
    # Directional face differences include a velocity jump introduced by a
    # reflecting boundary. This ensures those boundary cells are substepped
    # even when the cell-centred velocity is spatially uniform.
    du_dx = _maximum_absolute_difference(u_face, axis=1)
    du_dy = _maximum_absolute_difference(ubar, axis=0)
    dv_dx = _maximum_absolute_difference(vbar, axis=1)
    dv_dy = _maximum_absolute_difference(v_face, axis=0)
    jacobian_bound = tf.maximum(du_dx + du_dy, dv_dx + dv_dy) / dx
    return tf.abs(dt) * jacobian_bound


def _substep_count(maximum_deformation, deformation_limit, max_substeps):
    """Choose a bounded device-side step count without a host synchronization."""
    requested = tf.math.ceil(maximum_deformation / deformation_limit)
    requested = tf.where(
        tf.math.is_finite(requested),
        requested,
        tf.cast(2_000_000_000, requested.dtype),
    )
    # Avoid an undefined float-to-int conversion for pathologically large
    # input while preserving a useful 'larger than allowed' diagnostic.
    requested = tf.minimum(
        requested, tf.cast(2_000_000_000, requested.dtype)
    )
    required = tf.maximum(tf.cast(requested, tf.int32), 1)
    return required, tf.minimum(required, max_substeps)


def _apply_source(thickness, increment):
    """Apply SMB and report mass added by the non-negativity constraint."""
    unconstrained = thickness + increment
    constrained = tf.nn.relu(unconstrained)
    correction = tf.reduce_sum(constrained - unconstrained)
    return constrained, correction


def _strang_x_y(
    thickness, u_face, v_face, dt, dx, left, right, top, bottom
):
    first = _transport_x(thickness, u_face, 0.5 * dt, dx, left, right)
    second = _transport_y(first, v_face, dt, dx, top, bottom)
    return _transport_x(second, u_face, 0.5 * dt, dx, left, right)


def _strang_y_x(
    thickness, u_face, v_face, dt, dx, left, right, top, bottom
):
    first = _transport_y(thickness, v_face, 0.5 * dt, dx, top, bottom)
    second = _transport_x(first, u_face, dt, dx, left, right)
    return _transport_y(second, v_face, 0.5 * dt, dx, top, bottom)


@tf.function(autograph=False, reduce_retracing=True, jit_compile=True)
def _ffsl_step(
    ubar,
    vbar,
    thickness,
    dx,
    dt,
    smb,
    step_index,
    deformation_limit,
    max_substeps,
    left="zero",
    right="zero",
    top="zero",
    bottom="zero",
    limit_policy="accept",
):
    """Execute one conservative, arbitrary-CFL FFSL thickness step."""
    u_face, v_face = boundary.face_velocities(
        ubar, vbar, left, right, top, bottom
    )
    maximum_deformation = _velocity_deformation(
        ubar, vbar, u_face, v_face, dt, dx
    )
    required_substeps, substeps = _substep_count(
        maximum_deformation, deformation_limit, max_substeps
    )
    substep_dt = dt / tf.cast(substeps, thickness.dtype)
    half_source_increment = 0.5 * substep_dt * smb

    def condition(index, field, source_correction):
        del field, source_correction
        return index < substeps

    def body(index, field, source_correction):
        field, correction_before = _apply_source(
            field, half_source_increment
        )
        x_first = tf.equal(tf.math.floormod(step_index + index, 2), 0)
        field = tf.cond(
            x_first,
            lambda: _strang_x_y(
                field,
                u_face,
                v_face,
                substep_dt,
                dx,
                left,
                right,
                top,
                bottom,
            ),
            lambda: _strang_y_x(
                field,
                u_face,
                v_face,
                substep_dt,
                dx,
                left,
                right,
                top,
                bottom,
            ),
        )
        # A valid deformation step is monotone analytically.  Float32 prefix
        # sums can nevertheless leave millimetre-scale negative roundoff on a
        # kilometre-scale field.  Repair once per complete Strang step and
        # preserve its global flux-form mass, avoiding three line reductions.
        field = _repair_nonnegative_roundoff(field)
        field, correction_after = _apply_source(
            field, half_source_increment
        )
        return (
            index + 1,
            field,
            source_correction + correction_before + correction_after,
        )

    _, thickness_new, source_limiter_correction = tf.while_loop(
        condition,
        body,
        (
            tf.constant(0, tf.int32),
            thickness,
            tf.zeros((), thickness.dtype),
        ),
        parallel_iterations=1,
    )

    divflux = smb - tf.math.divide_no_nan(thickness_new - thickness, dt)
    source_limiter_volume = source_limiter_correction * dx * dx
    substep_limit_reached = required_substeps > max_substeps
    if limit_policy == "stop":
        step_accepted = tf.logical_not(substep_limit_reached)
        thickness_new = tf.where(step_accepted, thickness_new, thickness)
        divflux = tf.where(step_accepted, divflux, smb)
        source_limiter_volume = tf.where(
            step_accepted,
            source_limiter_volume,
            tf.zeros_like(source_limiter_volume),
        )
    elif limit_policy == "accept":
        step_accepted = tf.constant(True)
    else:
        raise ValueError(
            "limit_policy must be 'stop' or 'accept'; "
            f"got {limit_policy!r}."
        )
    return FFSLStepResult(
        thickness_new,
        divflux,
        substeps,
        required_substeps,
        maximum_deformation,
        substep_limit_reached,
        source_limiter_volume,
        step_accepted,
    )


def _options(cfg):
    """Read FFSL options with defaults for older or minimal configurations."""
    options = getattr(cfg.processes.thk, "ffsl", None)
    if options is None:
        return _DEFAULT_MAX_DEFORMATION, _DEFAULT_MAX_SUBSTEPS, "stop"
    return (
        float(getattr(options, "max_deformation", _DEFAULT_MAX_DEFORMATION)),
        int(getattr(options, "max_substeps", _DEFAULT_MAX_SUBSTEPS)),
        str(getattr(options, "limit_policy", "stop")).strip().lower(),
    )


def _solve_with_options(state, smb, options):
    """Adapt state tensors and cached options to the compiled kernel."""
    dtype = state.thk.dtype
    return _ffsl_step(
        tf.cast(state.ubar, dtype),
        tf.cast(state.vbar, dtype),
        state.thk,
        tf.cast(state.dx, dtype),
        tf.cast(state.dt, dtype),
        tf.cast(smb, dtype),
        tf.cast(state.it, tf.int32),
        tf.cast(options["deformation_limit"], dtype),
        tf.cast(options["max_substeps"], tf.int32),
        options["left"],
        options["right"],
        options["top"],
        options["bottom"],
        options["limit_policy"],
    )


def solve(state, cfg, smb):
    """Solve one step, parsing configuration for direct/test callers."""
    deformation_limit, max_substeps, limit_policy = _options(cfg)
    boundaries = boundary.get_boundary_conditions(cfg)
    options = {
        "deformation_limit": deformation_limit,
        "max_substeps": max_substeps,
        "limit_policy": limit_policy,
        "left": boundaries.left,
        "right": boundaries.right,
        "top": boundaries.top,
        "bottom": boundaries.bottom,
    }
    return _solve_with_options(state, smb, options)


def _validate_config(cfg, boundaries=None):
    """Validate options relevant to the FFSL backend."""
    p = cfg.processes.thk
    if boundaries is None:
        boundaries = boundary.get_boundary_conditions(cfg)
    boundary.validate_backend(boundaries, SUPPORTED_BOUNDARY_MODES, "ffsl")
    if str(getattr(p, "slope_type", "superbee")).strip().lower() != "superbee":
        raise ValueError(
            "cfg.processes.thk.scheme: ffsl currently requires "
            "slope_type: superbee."
        )

    deformation_limit, max_substeps, limit_policy = _options(cfg)
    if not math.isfinite(deformation_limit) or deformation_limit <= 0.0:
        raise ValueError(
            "cfg.processes.thk.ffsl.max_deformation must be positive and finite."
        )
    if max_substeps < 1:
        raise ValueError(
            "cfg.processes.thk.ffsl.max_substeps must be at least one."
        )
    if limit_policy not in ("accept", "stop"):
        raise ValueError(
            "cfg.processes.thk.ffsl.limit_policy must be 'stop' or 'accept'; "
            f"got {limit_policy!r}."
        )
    return boundaries


def initialize(cfg, state):
    """Validate the flux-form semi-Lagrangian backend configuration."""
    components = getattr(state, "thk_components", None)
    boundaries = _validate_config(
        cfg, None if components is None else components.boundaries
    )
    deformation_limit, max_substeps, limit_policy = _options(cfg)
    components.transport_options = {
        "deformation_limit": deformation_limit,
        "max_substeps": max_substeps,
        "limit_policy": limit_policy,
        "left": boundaries.left,
        "right": boundaries.right,
        "top": boundaries.top,
        "bottom": boundaries.bottom,
    }
    state.thk_step_accepted = tf.constant(True)


def update(cfg, state):
    """Advance thickness and publish FFSL conservation diagnostics."""
    del cfg
    if not hasattr(state, "smb"):
        state.smb = tf.zeros_like(state.thk)

    options = state.thk_components.transport_options
    result = _solve_with_options(state, state.smb, options)
    state.thk = result.thickness
    state.divflux = result.divflux
    state.ffsl_substeps = result.substeps
    state.ffsl_required_substeps = result.required_substeps
    state.ffsl_max_deformation = result.max_deformation
    state.ffsl_substep_limit_reached = result.substep_limit_reached
    state.ffsl_source_limiter_volume = result.source_limiter_volume
    state.thk_step_accepted = result.step_accepted
    if options["limit_policy"] == "stop":
        state.continue_run = tf.logical_and(
            tf.cast(getattr(state, "continue_run", True), tf.bool),
            result.step_accepted,
        )
