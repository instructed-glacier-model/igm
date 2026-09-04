#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Matrix-free theta method for the ice-thickness transport equation.

For velocities frozen over one time step, first-order upwind transport is a
linear operator ``D``.  The theta method therefore gives the linear system

    [I + dt * theta * D] H_new
        = H_old - dt * (1 - theta) * D(H_old) + dt * smb.

The system is solved by Jacobi-preconditioned BiCGSTAB.  The complete solve is
one TensorFlow graph containing a ``tf.while_loop``: there are no Python
iterations, NumPy operations, or device-to-host convergence checks.  BiCGSTAB
also has fixed O(N) storage, unlike restarted GMRES's O(restart * N) basis.

The upwind operator is assembled once per thickness step as a five-point
stencil.  Reusing that stencil avoids rebuilding face velocities and upwind
masks for every Krylov matrix-vector product.
"""

import math
from typing import NamedTuple

import tensorflow as tf

from .. import boundary
from ..domains import face_masks


SUPPORTED_BOUNDARY_MODES = ("zero", "symmetric", "periodic")
SUPPORTS_ACTIVE_DOMAIN = True


class ImplicitStepResult(NamedTuple):
    """Result and device-resident diagnostics from one implicit step."""

    thickness: tf.Tensor
    divflux: tf.Tensor
    iterations: tf.Tensor
    restarts: tf.Tensor
    relative_residual: tf.Tensor
    effective_tolerance: tf.Tensor
    converged: tf.Tensor
    breakdown: tf.Tensor
    transport_divflux: tf.Tensor
    nonnegative_correction_volume: tf.Tensor
    step_accepted: tf.Tensor


class _OperatorCoefficients(NamedTuple):
    """Five-point coefficients of ``I + dt * theta * D_upwind``.

    The four off-diagonal arrays are stored as positive magnitudes and are
    subtracted in :func:`_apply_operator`.
    """

    diagonal: tf.Tensor
    west: tf.Tensor
    east: tf.Tensor
    north: tf.Tensor
    south: tf.Tensor
    left: str
    right: str
    top: str
    bottom: str


# Keep the tensor helpers below as ordinary Python functions: they are traced
# inline into _solve_theta_step's single tf.function/XLA graph.  Giving every
# helper its own decorator would create independent trace caches and graph-call
# boundaries, which can inhibit fusion without moving any eager work to-device.
def _build_operator_coefficients(
    ubar,
    vbar,
    dx,
    dt_theta,
    left="zero",
    right="zero",
    top="zero",
    bottom="zero",
    active_mask=None,
):
    """Assemble the five-point upwind transport stencil once per solve."""
    u_face, v_face = boundary.face_velocities(
        ubar, vbar, left, right, top, bottom
    )
    if active_mask is not None:
        x_face_mask, y_face_mask = face_masks(active_mask)
        u_face = u_face * tf.cast(x_face_mask, u_face.dtype)
        v_face = v_face * tf.cast(y_face_mask, v_face.dtype)

    scale = dt_theta / dx
    u_left, u_right = u_face[:, :-1], u_face[:, 1:]
    v_north, v_south = v_face[:-1, :], v_face[1:, :]

    west = scale * tf.nn.relu(u_left)
    east = scale * tf.nn.relu(-u_right)
    north = scale * tf.nn.relu(v_north)
    south = scale * tf.nn.relu(-v_south)
    west, east, north, south = boundary.remove_nonperiodic_corner_couplings(
        west, east, north, south, left, right, top, bottom
    )
    diagonal = 1.0 + scale * (
        tf.nn.relu(u_right)
        + tf.nn.relu(-u_left)
        + tf.nn.relu(v_south)
        + tf.nn.relu(-v_north)
    )

    return _OperatorCoefficients(
        diagonal, west, east, north, south, left, right, top, bottom
    )


def _apply_operator(field, coefficients):
    """Apply a stencil whose retained edge couplings wrap periodically."""
    west, east, north, south = boundary.neighbor_fields(
        field,
        coefficients.left,
        coefficients.right,
        coefficients.top,
        coefficients.bottom,
    )
    return (
        coefficients.diagonal * field
        - coefficients.west * west
        - coefficients.east * east
        - coefficients.north * north
        - coefficients.south * south
    )


def _dot(left, right):
    return tf.reduce_sum(left * right)


def _safe_denominator(value):
    """Return a division-safe scalar and whether the original was usable."""
    valid = tf.math.is_finite(value) & tf.not_equal(value, 0.0)
    return tf.where(valid, value, tf.ones_like(value)), valid


def _bicgstab(
    coefficients,
    rhs,
    initial,
    initial_residual,
    tolerance,
    max_iter,
    max_restarts,
):
    """TensorFlow-native, right-Jacobi-preconditioned BiCGSTAB.

    All loop-carried large objects are fields shaped like ``rhs``.  No Krylov
    basis is retained, and ``parallel_iterations=1`` prevents TensorFlow from
    keeping multiple iterations live concurrently.
    """
    dtype = rhs.dtype
    zero = tf.zeros_like(rhs)
    one = tf.ones([], dtype=dtype)

    rhs_norm_sq = _dot(rhs, rhs)
    # A relative tolerance remains meaningful for a zero RHS without imposing a
    # macroscopic absolute tolerance on normally scaled thickness fields.
    norm_floor = tf.cast(1.0e-30 if dtype == tf.float32 else 1.0e-300, dtype)
    machine_epsilon = tf.cast(
        1.1920928955078125e-7 if dtype == tf.float32 else 2.220446049250313e-16,
        dtype,
    )
    # At high CFL the operator norm grows with its diagonal. In finite
    # precision, a smaller b-relative residual is not attainable reliably.
    # Honouring this floor prevents false-convergence/restart cycles while
    # retaining the requested tolerance whenever the dtype can represent it.
    effective_tolerance = tf.maximum(
        tolerance,
        2.0 * machine_epsilon * tf.reduce_max(tf.abs(coefficients.diagonal)),
    )
    target_norm_sq = tf.square(effective_tolerance) * tf.maximum(
        rhs_norm_sq, norm_floor
    )
    initial_norm_sq = _dot(initial_residual, initial_residual)
    initially_converged = tf.math.is_finite(initial_norm_sq) & (
        initial_norm_sq <= target_norm_sq
    )

    def condition(
        iteration,
        _solution,
        _residual,
        _shadow_residual,
        _search,
        _operator_search,
        _rho_old,
        _alpha,
        _omega,
        converged,
        breakdown,
        restarts,
        _best_solution,
        _best_norm_sq,
    ):
        return (
            (iteration < max_iter)
            & tf.logical_not(converged)
            & tf.logical_not(breakdown)
        )

    def body(
        iteration,
        solution,
        residual,
        shadow_residual,
        search,
        operator_search,
        rho_old,
        alpha,
        omega,
        _converged,
        _breakdown,
        restarts,
        best_solution,
        best_norm_sq,
    ):
        rho = _dot(shadow_residual, residual)
        rho_den, rho_old_valid = _safe_denominator(rho_old)
        omega_den, omega_old_valid = _safe_denominator(omega)
        rho_valid = tf.math.is_finite(rho) & tf.not_equal(rho, 0.0)

        beta = (rho / rho_den) * (alpha / omega_den)
        search_next = residual + beta * (search - omega * operator_search)
        preconditioned_search = search_next / coefficients.diagonal
        operator_search_next = _apply_operator(preconditioned_search, coefficients)

        shadow_dot_operator = _dot(shadow_residual, operator_search_next)
        alpha_den, alpha_den_valid = _safe_denominator(shadow_dot_operator)
        alpha_next = rho / alpha_den
        intermediate_residual = residual - alpha_next * operator_search_next
        intermediate_norm_sq = _dot(intermediate_residual, intermediate_residual)

        first_half_valid = (
            rho_valid
            & rho_old_valid
            & omega_old_valid
            & alpha_den_valid
            & tf.math.is_finite(beta)
            & tf.math.is_finite(alpha_next)
            & tf.math.is_finite(intermediate_norm_sq)
        )
        converged_after_first_half = first_half_valid & (
            intermediate_norm_sq <= target_norm_sq
        )

        def verify_or_restart(candidate_solution):
            """Verify convergence or recover from a numerical breakdown.

            Recursive Krylov residuals can drift from ``b - A(x)`` in single
            precision at high CFL. Recomputing the true residual only at an
            apparent convergence or breakdown keeps the common path fast while
            ensuring that ``tolerance`` always refers to the true residual.
            """
            true_residual = rhs - _apply_operator(candidate_solution, coefficients)
            true_norm_sq = _dot(true_residual, true_residual)
            true_residual_valid = tf.math.is_finite(true_norm_sq)
            truly_converged = true_residual_valid & (true_norm_sq <= target_norm_sq)
            can_restart = (
                true_residual_valid
                & tf.logical_not(truly_converged)
                & (restarts < max_restarts)
            )
            better = true_residual_valid & (true_norm_sq < best_norm_sq)
            return (
                iteration + 1,
                candidate_solution,
                true_residual,
                true_residual,
                zero,
                zero,
                one,
                one,
                one,
                truly_converged,
                tf.logical_not(truly_converged | can_restart),
                restarts + tf.cast(can_restart, tf.int32),
                tf.where(better, candidate_solution, best_solution),
                tf.where(better, true_norm_sq, best_norm_sq),
            )

        def finish_after_first_half():
            solution_next = solution + alpha_next * preconditioned_search
            return verify_or_restart(solution_next)

        def finish_full_iteration():
            preconditioned_intermediate = intermediate_residual / coefficients.diagonal
            operator_intermediate = _apply_operator(
                preconditioned_intermediate, coefficients
            )
            operator_norm_sq = _dot(operator_intermediate, operator_intermediate)
            omega_den_next, omega_den_valid = _safe_denominator(operator_norm_sq)
            omega_next = (
                _dot(operator_intermediate, intermediate_residual) / omega_den_next
            )

            solution_candidate = (
                solution
                + alpha_next * preconditioned_search
                + omega_next * preconditioned_intermediate
            )
            residual_candidate = (
                intermediate_residual - omega_next * operator_intermediate
            )
            residual_norm_sq = _dot(residual_candidate, residual_candidate)
            omega_valid = tf.math.is_finite(omega_next) & tf.not_equal(omega_next, 0.0)
            step_valid = (
                first_half_valid
                & omega_den_valid
                & omega_valid
                & tf.math.is_finite(residual_norm_sq)
            )

            def accept_step():
                def accept_unconverged_step():
                    better = residual_norm_sq < best_norm_sq
                    return (
                        iteration + 1,
                        solution_candidate,
                        residual_candidate,
                        shadow_residual,
                        search_next,
                        operator_search_next,
                        rho,
                        alpha_next,
                        omega_next,
                        tf.constant(False),
                        tf.constant(False),
                        restarts,
                        tf.where(better, solution_candidate, best_solution),
                        tf.minimum(residual_norm_sq, best_norm_sq),
                    )

                return tf.cond(
                    residual_norm_sq <= target_norm_sq,
                    lambda: verify_or_restart(solution_candidate),
                    accept_unconverged_step,
                )

            def restart_from_current_solution():
                # BiCGSTAB may lose bi-orthogonality at high CFL.  A cheap
                # residual replacement gives it a fresh shadow direction while
                # preserving fixed memory.  This is only taken on breakdown.
                return verify_or_restart(solution)

            return tf.cond(
                step_valid,
                accept_step,
                restart_from_current_solution,
            )

        return tf.cond(
            converged_after_first_half,
            finish_after_first_half,
            finish_full_iteration,
        )

    loop_result = tf.while_loop(
        condition,
        body,
        (
            tf.zeros([], dtype=tf.int32),
            initial,
            initial_residual,
            initial_residual,
            zero,
            zero,
            one,
            one,
            one,
            initially_converged,
            tf.constant(False),
            tf.zeros([], dtype=tf.int32),
            initial,
            initial_norm_sq,
        ),
        parallel_iterations=1,
        swap_memory=False,
    )

    (
        iterations,
        solution,
        residual,
        _shadow_residual,
        _search,
        _operator_search,
        _rho,
        _alpha,
        _omega,
        converged,
        breakdown,
        restarts,
        best_solution,
        best_norm_sq,
    ) = loop_result

    residual_norm_sq = _dot(residual, residual)

    def return_converged():
        # Convergence exits only through verify_or_restart, so this residual is
        # already the true b - A(x) residual.
        return solution, residual, residual_norm_sq

    def return_best_failed_iterate():
        # A max-iteration or breakdown exit is rare, so spend two matvecs here
        # to return and report the genuinely better of the last and best-seen
        # recursive-residual iterates.
        last_residual = rhs - _apply_operator(solution, coefficients)
        last_norm_sq = _dot(last_residual, last_residual)
        saved_residual = rhs - _apply_operator(best_solution, coefficients)
        saved_norm_sq = _dot(saved_residual, saved_residual)
        use_saved = saved_norm_sq < last_norm_sq
        return (
            tf.where(use_saved, best_solution, solution),
            tf.where(use_saved, saved_residual, last_residual),
            tf.where(use_saved, saved_norm_sq, last_norm_sq),
        )

    solution, residual, residual_norm_sq = tf.cond(
        converged, return_converged, return_best_failed_iterate
    )
    relative_residual = tf.sqrt(residual_norm_sq / tf.maximum(rhs_norm_sq, norm_floor))
    return (
        solution,
        residual,
        iterations,
        restarts,
        relative_residual,
        effective_tolerance,
        converged,
        breakdown,
    )


@tf.function(autograph=False, reduce_retracing=True, jit_compile=True)
def _solve_theta_step(
    ubar,
    vbar,
    thickness,
    dx,
    dt,
    smb,
    theta,
    tolerance,
    max_iter,
    max_restarts,
    left="zero",
    right="zero",
    top="zero",
    bottom="zero",
    active_mask=None,
    failure_policy="accept",
):
    """Compiled tensor-only implementation of one theta-method step."""
    dt_theta = dt * theta
    if active_mask is None:
        active_smb = smb
    else:
        active_mask = tf.cast(active_mask, tf.bool)
        active_smb = tf.where(active_mask, smb, tf.zeros_like(smb))
    coefficients = _build_operator_coefficients(
        ubar,
        vbar,
        dx,
        dt_theta,
        left,
        right,
        top,
        bottom,
        active_mask,
    )

    # The warm start is H_old.  Its operator application is also reused to form
    # the explicit part of the theta-method RHS, avoiding a separate divflux.
    operator_old = _apply_operator(thickness, coefficients)
    explicit_to_implicit_ratio = (1.0 - theta) / theta
    rhs = (
        thickness
        - explicit_to_implicit_ratio * (operator_old - thickness)
        + dt * active_smb
    )
    initial_residual = rhs - operator_old

    (
        thickness_solved,
        residual,
        iterations,
        restarts,
        relative_residual,
        effective_tolerance,
        converged,
        breakdown,
    ) = _bicgstab(
        coefficients,
        rhs,
        thickness,
        initial_residual,
        tolerance,
        max_iter,
        max_restarts,
    )

    # The carried residual is r = b - A(H_new).  Reusing it gives the actual
    # theta-weighted flux divergence without one more operator application.
    scaled_divflux_new = rhs - residual - thickness_solved
    scaled_divflux_old = operator_old - thickness
    dt_divflux = scaled_divflux_new + explicit_to_implicit_ratio * scaled_divflux_old
    if active_mask is None:
        transport_divflux = tf.math.divide_no_nan(dt_divflux, dt)
    else:
        transport_divflux = tf.where(
            active_mask,
            tf.math.divide_no_nan(dt_divflux, dt),
            tf.zeros_like(dt_divflux),
        )

    # Theta < 1 is not monotone at a sharp margin. Keep the physical thickness
    # nonnegative, but expose the correction and make the public divflux close
    # the *actual* post-projection thickness budget exactly.
    if active_mask is None:
        thickness_new = tf.nn.relu(thickness_solved)
        correction = thickness_new - thickness_solved
    else:
        thickness_new = tf.where(
            active_mask, tf.nn.relu(thickness_solved), thickness
        )
        correction = tf.where(
            active_mask,
            thickness_new - thickness_solved,
            tf.zeros_like(thickness),
        )
    nonnegative_correction_volume = tf.reduce_sum(correction) * dx * dx
    if active_mask is None:
        divflux = smb - tf.math.divide_no_nan(thickness_new - thickness, dt)
    else:
        divflux = tf.where(
            active_mask,
            smb - tf.math.divide_no_nan(thickness_new - thickness, dt),
            tf.zeros_like(smb),
        )

    solved = converged & tf.logical_not(breakdown)
    if failure_policy == "stop":
        step_accepted = solved
        thickness_new = tf.where(step_accepted, thickness_new, thickness)
        divflux = tf.where(step_accepted, divflux, smb)
        transport_divflux = tf.where(
            step_accepted,
            transport_divflux,
            tf.zeros_like(transport_divflux),
        )
        nonnegative_correction_volume = tf.where(
            step_accepted,
            nonnegative_correction_volume,
            tf.zeros_like(nonnegative_correction_volume),
        )
    elif failure_policy == "accept":
        step_accepted = tf.constant(True)
    else:
        raise ValueError(
            "failure_policy must be 'stop' or 'accept'; "
            f"got {failure_policy!r}."
        )

    return ImplicitStepResult(
        thickness_new,
        divflux,
        iterations,
        restarts,
        relative_residual,
        effective_tolerance,
        converged,
        breakdown,
        transport_divflux,
        nonnegative_correction_volume,
        step_accepted,
    )


def _solve_with_options(state, smb, options):
    """Adapt state tensors and cached options to the compiled solver."""
    dtype = state.thk.dtype
    return _solve_theta_step(
        tf.cast(state.ubar, dtype),
        tf.cast(state.vbar, dtype),
        state.thk,
        tf.cast(state.dx, dtype),
        tf.cast(state.dt, dtype),
        tf.cast(smb, dtype),
        tf.cast(options["theta"], dtype),
        tf.cast(options["tolerance"], dtype),
        tf.cast(options["max_iter"], tf.int32),
        tf.cast(options["max_restarts"], tf.int32),
        options["left"],
        options["right"],
        options["top"],
        options["bottom"],
        getattr(state, "thk_active_mask", None),
        options["failure_policy"],
    )


def _options(cfg, boundaries=None):
    """Validate and return the static options owned by this backend."""
    p = cfg.processes.thk
    if boundaries is None:
        boundaries = boundary.get_boundary_conditions(cfg)
    boundary.validate_backend(
        boundaries, SUPPORTED_BOUNDARY_MODES, "implicit"
    )

    theta = float(p.implicit.theta)
    if not 0.5 <= theta <= 1.0:
        raise ValueError(
            "cfg.processes.thk.implicit.theta must be between 0.5 and 1.0, "
            f"got {theta}."
        )

    tolerance = float(p.implicit.solver.tol)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "cfg.processes.thk.implicit.solver.tol must be positive and finite."
        )

    max_iter = int(p.implicit.solver.max_iter)
    if max_iter <= 0:
        raise ValueError(
            "cfg.processes.thk.implicit.solver.max_iter must be positive."
        )

    max_restarts = int(p.implicit.solver.max_restarts)
    if max_restarts < 0:
        raise ValueError(
            "cfg.processes.thk.implicit.solver.max_restarts cannot be negative."
        )
    failure_policy = str(
        getattr(p.implicit.solver, "failure_policy", "stop")
    ).strip().lower()
    if failure_policy not in ("accept", "stop"):
        raise ValueError(
            "cfg.processes.thk.implicit.solver.failure_policy must be "
            f"'stop' or 'accept'; got {failure_policy!r}."
        )
    return {
        "theta": theta,
        "tolerance": tolerance,
        "max_iter": max_iter,
        "max_restarts": max_restarts,
        "failure_policy": failure_policy,
        "left": boundaries.left,
        "right": boundaries.right,
        "top": boundaries.top,
        "bottom": boundaries.bottom,
    }


def solve(state, cfg, smb):
    """Solve one step, parsing configuration for direct/test callers."""
    return _solve_with_options(state, smb, _options(cfg))


def initialize(cfg, state):
    """Validate the scheme and initialize device-resident diagnostics."""
    state.thk_components.transport_options = _options(
        cfg, state.thk_components.boundaries
    )
    state.thk_solver_iterations = tf.zeros([], dtype=tf.int32)
    state.thk_solver_restarts = tf.zeros([], dtype=tf.int32)
    state.thk_solver_relative_residual = tf.zeros([], dtype=state.thk.dtype)
    state.thk_solver_effective_tolerance = tf.zeros([], dtype=state.thk.dtype)
    state.thk_solver_converged = tf.constant(True)
    state.thk_solver_breakdown = tf.constant(False)
    state.thk_step_accepted = tf.constant(True)
    state.thk_transport_divflux = tf.zeros_like(state.thk)
    state.thk_nonnegative_correction_volume = tf.zeros(
        [], dtype=state.thk.dtype
    )


def update(cfg, state):
    """Advance thickness by one theta-method step."""
    if not hasattr(state, "smb"):
        state.smb = tf.zeros_like(state.thk)

    del cfg
    options = state.thk_components.transport_options
    result = _solve_with_options(state, state.smb, options)
    state.thk = result.thickness
    state.divflux = result.divflux
    state.thk_transport_divflux = result.transport_divflux
    state.thk_nonnegative_correction_volume = (
        result.nonnegative_correction_volume
    )
    state.thk_solver_iterations = result.iterations
    state.thk_solver_restarts = result.restarts
    state.thk_solver_relative_residual = result.relative_residual
    state.thk_solver_effective_tolerance = result.effective_tolerance
    state.thk_solver_converged = result.converged
    state.thk_solver_breakdown = result.breakdown
    state.thk_step_accepted = result.step_accepted
    if options["failure_policy"] == "stop":
        state.continue_run = tf.logical_and(
            tf.cast(getattr(state, "continue_run", True), tf.bool),
            result.step_accepted,
        )
