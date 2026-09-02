#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Callable, NamedTuple, Optional

import tensorflow as tf

from .optimizer import Optimizer
from ..mappings import Mapping
from ..halt import Halt, HaltState, HaltStatus, StepState
from ..halt.criteria import CriterionRelTol
from ..halt.metrics import MetricCost
from .line_searches import LineSearches
from igm.utils.math.tridiagonal import (
    solve_block_tridiagonal,
    solve_tridiagonal_pcr,
)

from ..operators import Operator, Tridiag1DADOperator


class _NewtonLoopResult(NamedTuple):
    iterations: tf.Tensor
    theta: tf.Tensor
    cost: tf.Tensor
    U: tf.Tensor
    V: tf.Tensor
    grad_theta_norm: tf.Tensor
    halt_status: tf.Tensor
    relative_change: tf.Tensor
    criterion_satisfied: tf.Tensor


class OptimizerTridiagNewton(Optimizer):
    """Newton optimizer with direct linear solves for y-invariant SSA.

    Exploits the ``Ny=2`` + periodic-north-south grid convention (see
    ``tridiag1d.py``): the Hessian, restricted to the live degrees of
    freedom, is an exact 2x2-block tridiagonal system along x. Each Newton
    step is solved by parallel cyclic reduction instead of an iterative CG
    solve.
    """

    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        halt: Optional[Halt] = None,
        print_cost: bool = True,
        print_cost_freq: int = 1,
        precision: str = "float32",
        ord_grad_u: str = "l2_weighted",
        ord_grad_theta: str = "l2_weighted",
        alpha_min: float = 0.0,
        iter_max: int = 100,
        damping: float = 2e-2,
        damping_adaptive: bool = False,
        damping_min: float = 1e-12,
        damping_max: float = 1e2,
        damping_down: float = 0.25,
        damping_up: float = 4.0,
        operator: Optional[Operator] = None,
        scalar_flowline: bool = False,
        **kwargs,
    ):
        super().__init__(
            cost_fn,
            map,
            halt,
            print_cost,
            print_cost_freq,
            precision,
            ord_grad_u,
            ord_grad_theta,
            **kwargs,
        )

        self.name = "tridiag_newton"
        self.line_search = LineSearches["armijo"]()

        self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
        self._iter_max_value = int(iter_max)
        self.alpha_min = float(alpha_min)

        self.damping = tf.constant(damping, dtype=self.precision)
        self._damping_value = float(damping)
        self.damping_adaptive = bool(damping_adaptive)
        self.damping_min = float(damping_min)
        self.damping_max = float(damping_max)
        self.damping_down = float(damping_down)
        self.damping_up = float(damping_up)

        self.operator: Operator = operator or Tridiag1DADOperator(
            cost_fn, self.map, precision
        )
        self.scalar_flowline = bool(scalar_flowline)
        self._scalar_flowline_validated = False
        self._damping_var = tf.Variable(
            self.damping, dtype=self.precision, trainable=False
        )

        self._xla_rel_tol = None
        if self._supports_compiled_loop():
            self._xla_rel_tol = float(self.halt.crit_success[0].tol)

    def update_parameters(self, iter_max: int, damping: float) -> None:
        iter_max = int(iter_max)
        if iter_max != self._iter_max_value:
            self.iter_max.assign(iter_max)
            self._iter_max_value = iter_max
        damping = float(damping)
        if damping != self._damping_value:
            self.damping = tf.cast(damping, self.precision)
            self._damping_value = damping

    def _compiled_loop_enabled(self) -> bool:
        return (
            self._supports_compiled_loop()
            and not self.display.enabled
            and not self.debug_mode
        )

    def minimize(self, inputs: tf.Tensor) -> tf.Tensor:
        """Skip eager optimizer bookkeeping unused by the stateless XLA loop."""
        if self._compiled_loop_enabled():
            if self._iter_max_value == 0:
                return tf.zeros([0], dtype=self.precision)
            return self.minimize_impl(inputs)
        return super().minimize(inputs)

    def _cost_and_grad(self, inputs: tf.Tensor):
        return self.operator.cost_and_grad(inputs)

    def _supports_compiled_loop(self) -> bool:
        """Whether the XLA-compiled production loop is applicable."""
        if not hasattr(self.operator, "assemble_bands_at"):
            return False
        if self.halt is None or self.halt.freq != 1:
            return False
        if self.halt.crit_failure or len(self.halt.crit_success) != 1:
            return False
        criterion = self.halt.crit_success[0]
        return isinstance(criterion, CriterionRelTol) and isinstance(
            criterion.metric, MetricCost
        )

    def _validate_scalar_flowline(self, inputs: tf.Tensor) -> None:
        """Validate the scalar path's symmetry assumptions before first use."""
        if self._scalar_flowline_validated or not self.scalar_flowline:
            return
        if not all(
            hasattr(self.operator, name)
            for name in ("scalar_system_at", "scalar_cost_at")
        ):
            raise ValueError(
                "tridiag_newton.scalar_flowline requires analytic assembly."
            )
        if not bool(getattr(self.operator, "supports_scalar_flowline", False)):
            raise ValueError(
                "tridiag_newton.scalar_flowline analytic kernels currently "
                "support viscosity, gravity, floating, and power-law sliding."
            )
        if inputs.shape[1] != 2:
            raise ValueError(
                "tridiag_newton.scalar_flowline requires exactly two y rows."
            )
        _, raw_v = self.map.get_theta()
        max_v = float(tf.reduce_max(tf.abs(raw_v)).numpy())
        max_y_difference = float(
            tf.reduce_max(tf.abs(inputs[:, 1, :, :] - inputs[:, 0, :, :])).numpy()
        )
        if max_v > 1e-12 or max_y_difference > 1e-12:
            raise ValueError(
                "tridiag_newton.scalar_flowline requires zero transverse "
                "velocity and exactly y-invariant inputs; got max |V|="
                f"{max_v:.3e}, max y difference={max_y_difference:.3e}."
            )
        self._scalar_flowline_validated = True

    @tf.function(reduce_retracing=True)
    def _solve(self, grad_theta_flat: tf.Tensor, damping: tf.Tensor) -> tf.Tensor:
        """Solve ``(H + damping I) p = -g`` exactly via cyclic reduction."""
        bands = self.operator.assemble_bands(None, damping)
        g_u, g_v = self.operator.split_row0(grad_theta_flat)
        rhs = -tf.stack([g_u, g_v], axis=1)
        p = solve_block_tridiagonal(bands["west"], bands["center"], bands["east"], rhs)
        return self.operator.join_row0(p[:, 0, :], p[:, 1, :])

    @tf.function(reduce_retracing=True)
    def _newton_step(
        self,
        theta_flat: tf.Tensor,
        inputs: tf.Tensor,
        damping: tf.Variable,
        iteration: tf.Tensor,
    ):
        """One full Newton step -- grad, stencil, direct solve, descent guard,
        Armijo search, the parameter update (including the map's state
        variables), and the gradient-norm bookkeeping -- as a single
        compiled call with no host round-trip anywhere inside it.

        The Hessian is reassembled at every iteration.
        """
        cost, grad_u, grad_theta = self._cost_and_grad(inputs)
        grad_theta_flat = self.map.flatten_theta(grad_theta)
        self.operator.prepare(inputs, damping)
        p_flat = self._solve(grad_theta_flat, damping)
        p_flat = self._descent_direction(p_flat, grad_theta_flat)

        df0 = self._dot(grad_theta_flat, p_flat)
        alpha = self._armijo_search(theta_flat, p_flat, inputs, cost, df0)
        alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))
        if self.damping_adaptive:
            damping.assign(self._updated_damping(damping, alpha))

        new_theta_flat = theta_flat + alpha * p_flat
        self.map.set_theta(self.map.unflatten_theta(new_theta_flat))
        U, V = self.map.get_UV(inputs)
        # Pure tensor bookkeeping and halt-criterion evaluation stay in this
        # graph. The outer loop only publishes the returned tensors to the
        # Python step/halt state objects and performs one scalar break check.
        grad_u_norm, grad_theta_norm = self._get_grad_norm(grad_u, grad_theta)
        step_state = StepState(
            iter=iteration,
            u=[U, V],
            theta=new_theta_flat,
            cost=cost,
            grad_u_norm=grad_u_norm,
            grad_theta_norm=grad_theta_norm,
        )
        if self.halt is None:
            halt_status = tf.constant(HaltStatus.CONTINUE.value, tf.int32)
            criterion_values = []
            criterion_satisfied = []
        else:
            halt_status, criterion_values, criterion_satisfied = self.halt.check(
                iteration, step_state
            )
            # Criteria are scalar diagnostics, but stateful criteria use
            # validate_shape=False variables and can therefore lose their
            # static shape while tracing a surrounding tf.while_loop.
            criterion_values = [tf.reshape(value, []) for value in criterion_values]
            criterion_satisfied = [
                tf.reshape(satisfied, []) for satisfied in criterion_satisfied
            ]
        max_iter_reached = tf.greater_equal(iteration + 1, self.iter_max)
        halt_status = tf.where(
            tf.not_equal(halt_status, HaltStatus.CONTINUE.value),
            halt_status,
            tf.where(
                max_iter_reached,
                tf.constant(HaltStatus.COMPLETED.value, tf.int32),
                tf.constant(HaltStatus.CONTINUE.value, tf.int32),
            ),
        )

        return (
            new_theta_flat,
            cost,
            U,
            V,
            grad_u_norm,
            grad_theta_norm,
            halt_status,
            criterion_values,
            criterion_satisfied,
        )

    @tf.function(reduce_retracing=True, jit_compile=True)
    def _xla_newton_loop(
        self,
        theta_flat: tf.Tensor,
        inputs: tf.Tensor,
        iter_max: tf.Tensor,
        damping: tf.Tensor,
    ):
        """Pure analytic Newton loop compiled into one GPU executable.

        Unlike the regular step loop, this path carries velocity, damping,
        and the relative-cost convergence state as tensors. It does not
        update a TensorFlow resource or TensorArray inside the loop, allowing
        XLA to fuse the small flowline algebra instead of scheduling a graph body
        once per Newton iteration.
        """
        dtype = theta_flat.dtype
        tol = tf.cast(self._xla_rel_tol, dtype)
        nan = tf.constant(float("nan"), dtype)
        iteration0 = tf.constant(0, tf.int32)
        initialized0 = tf.constant(False)
        satisfied0 = tf.constant(False)
        previous_cost0 = nan
        cost0 = nan
        relative_change0 = nan
        grad_theta_norm0 = nan

        def cond(
            iteration,
            theta,
            damping_value,
            initialized,
            satisfied,
            previous_cost,
            cost,
            relative_change,
            grad_theta_norm,
        ):
            del (
                theta,
                damping_value,
                initialized,
                previous_cost,
                cost,
                relative_change,
                grad_theta_norm,
            )
            return tf.logical_and(iteration < iter_max, tf.logical_not(satisfied))

        def body(
            iteration,
            theta,
            damping_value,
            initialized,
            satisfied,
            previous_cost,
            cost,
            relative_change,
            grad_theta_norm,
        ):
            del satisfied, cost, relative_change, grad_theta_norm
            cost, grad_flat = self.operator.cost_grad_at(inputs, theta)
            bands = self.operator.assemble_bands_at(
                inputs, theta, damping_value
            )
            g_u, g_v = self.operator.split_row0(grad_flat)
            rhs = -tf.stack([g_u, g_v], axis=1)
            p = solve_block_tridiagonal(
                bands["west"], bands["center"], bands["east"], rhs
            )
            p_flat = self.operator.join_row0(p[:, 0, :], p[:, 1, :])
            p_flat = self._descent_direction(p_flat, grad_flat)
            df0 = self._dot(grad_flat, p_flat)
            alpha = self._armijo_gpu(theta, p_flat, inputs, cost, df0)
            alpha = tf.maximum(alpha, tf.cast(self.alpha_min, dtype))
            theta = theta + alpha * p_flat

            if self.damping_adaptive:
                damping_value = self._updated_damping(damping_value, alpha)

            relative_change = tf.where(
                initialized,
                tf.abs(cost - previous_cost)
                / (tf.abs(previous_cost) + tf.cast(1e-12, dtype)),
                nan,
            )
            satisfied = tf.logical_and(initialized, relative_change < tol)
            grad_theta_norm = tf.norm(grad_flat, ord=float("inf"))
            return (
                iteration + 1,
                theta,
                damping_value,
                tf.constant(True),
                satisfied,
                cost,
                cost,
                relative_change,
                grad_theta_norm,
            )

        (
            iteration,
            theta_flat,
            damping,
            _,
            satisfied,
            _,
            cost,
            relative_change,
            grad_theta_norm,
        ) = tf.while_loop(
            cond,
            body,
            (
                iteration0,
                theta_flat,
                damping,
                initialized0,
                satisfied0,
                previous_cost0,
                cost0,
                relative_change0,
                grad_theta_norm0,
            ),
            parallel_iterations=1,
        )
        halt_status = tf.where(
            satisfied,
            tf.constant(HaltStatus.SUCCESS.value, tf.int32),
            tf.constant(HaltStatus.COMPLETED.value, tf.int32),
        )
        U, V = self.map.unflatten_theta(theta_flat)
        for apply_bc in self.map.apply_bcs:
            U, V = apply_bc(U, V)
        return _NewtonLoopResult(
            iteration,
            theta_flat,
            cost,
            U,
            V,
            grad_theta_norm,
            halt_status,
            relative_change,
            satisfied,
        )

    @tf.function(reduce_retracing=True, jit_compile=True)
    def _xla_scalar_newton_loop(
        self,
        u_row: tf.Tensor,
        inputs: tf.Tensor,
        iter_max: tf.Tensor,
        damping: tf.Tensor,
    ):
        """XLA Newton loop for an exactly y-invariant scalar flowline.

        With zero transverse velocity and identical y rows, the SSA energy is
        invariant on ``V=0`` and its u-v Hessian blocks vanish.  Solving only
        the flow-parallel tridiagonal system is therefore an exact symmetry
        reduction, not an approximation to this experiment's equations.
        """
        dtype = u_row.dtype
        tol = tf.cast(self._xla_rel_tol, dtype)
        nan = tf.constant(float("nan"), dtype)

        def cond(
            iteration,
            u,
            damping_value,
            initialized,
            satisfied,
            previous_cost,
            cost,
            relative_change,
            grad_norm,
        ):
            del (
                u,
                damping_value,
                initialized,
                previous_cost,
                cost,
                relative_change,
                grad_norm,
            )
            return tf.logical_and(iteration < iter_max, tf.logical_not(satisfied))

        def body(
            iteration,
            u,
            damping_value,
            initialized,
            satisfied,
            previous_cost,
            cost,
            relative_change,
            grad_norm,
        ):
            del satisfied, cost, relative_change, grad_norm
            cost, grad, lower, diagonal, upper = self.operator.scalar_system_at(
                inputs, u, damping_value
            )
            p = solve_tridiagonal_pcr(lower, diagonal, upper, -grad)
            p = tf.where(tf.reduce_sum(grad * p) >= 0.0, -grad, p)
            df0 = tf.reduce_sum(grad * p)
            alpha = self._scalar_armijo_search(inputs, u, p, cost, df0)
            alpha = tf.maximum(alpha, tf.cast(self.alpha_min, dtype))
            u = u + alpha * p

            if self.damping_adaptive:
                damping_value = self._updated_damping(damping_value, alpha)

            relative_change = tf.where(
                initialized,
                tf.abs(cost - previous_cost)
                / (tf.abs(previous_cost) + tf.cast(1e-12, dtype)),
                nan,
            )
            satisfied = tf.logical_and(initialized, relative_change < tol)
            return (
                iteration + 1,
                u,
                damping_value,
                tf.constant(True),
                satisfied,
                cost,
                cost,
                relative_change,
                tf.reduce_max(tf.abs(grad)),
            )

        (
            iteration,
            u_row,
            damping,
            _,
            satisfied,
            _,
            cost,
            relative_change,
            grad_theta_norm,
        ) = tf.while_loop(
            cond,
            body,
            (
                tf.constant(0, tf.int32),
                u_row,
                damping,
                tf.constant(False),
                tf.constant(False),
                nan,
                nan,
                nan,
                nan,
            ),
            parallel_iterations=1,
        )
        halt_status = tf.where(
            satisfied,
            tf.constant(HaltStatus.SUCCESS.value, tf.int32),
            tf.constant(HaltStatus.COMPLETED.value, tf.int32),
        )
        theta_flat = self.operator.join_row0(u_row, tf.zeros_like(u_row))
        U, V = self.map.unflatten_theta(theta_flat)
        for apply_bc in self.map.apply_bcs:
            U, V = apply_bc(U, V)
        return _NewtonLoopResult(
            iteration,
            theta_flat,
            cost,
            U,
            V,
            grad_theta_norm,
            halt_status,
            relative_change,
            satisfied,
        )

    def _descent_direction(
        self, p_flat: tf.Tensor, grad_theta_flat: tf.Tensor
    ) -> tf.Tensor:
        dot_gp = self._dot(grad_theta_flat, p_flat)
        return tf.cond(dot_gp >= 0.0, lambda: -grad_theta_flat, lambda: p_flat)

    @tf.function(reduce_retracing=True)
    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        dtype = self.precision
        return tf.tensordot(tf.cast(a, dtype), tf.cast(b, dtype), axes=1)

    def _cost_at(self, inputs: tf.Tensor, theta_flat: tf.Tensor) -> tf.Tensor:
        return self.operator.cost_at(inputs, theta_flat)

    def _updated_damping(self, damping: tf.Tensor, alpha: tf.Tensor) -> tf.Tensor:
        dtype = damping.dtype
        decreased = tf.maximum(
            damping * tf.cast(self.damping_down, dtype),
            tf.cast(self.damping_min, dtype),
        )
        increased = tf.minimum(
            damping * tf.cast(self.damping_up, dtype),
            tf.cast(self.damping_max, dtype),
        )
        return tf.where(
            alpha >= tf.cast(0.5, alpha.dtype),
            decreased,
            tf.where(alpha < tf.cast(0.1, alpha.dtype), increased, damping),
        )

    @tf.function(reduce_retracing=True)
    def _armijo_search(
        self,
        theta_flat: tf.Tensor,
        p_flat: tf.Tensor,
        inputs: tf.Tensor,
        f0: tf.Tensor,
        df0: tf.Tensor,
    ) -> tf.Tensor:
        """Backtracking Armijo search with no host round-trip.

        Same sequence of trial step sizes as the eager Python loop it
        replaces (test alpha, and only shrink by ``rho`` on rejection), but
        driven entirely by ``tf.while_loop`` so acceptance is decided by a
        tensor comparison instead of a per-trial ``float()`` GPU->CPU sync.
        """
        dtype = theta_flat.dtype
        c1 = tf.cast(self.line_search.c1, dtype)
        rho = tf.cast(self.line_search.rho, dtype)
        max_iter = tf.constant(self.line_search.max_iter, tf.int32)

        def cond(i, alpha, accepted):
            del alpha
            return tf.logical_and(i < max_iter, tf.logical_not(accepted))

        def body(i, alpha, accepted):
            del accepted
            f_alpha = self._cost_at(inputs, theta_flat + alpha * p_flat)
            ok = f_alpha <= f0 + c1 * alpha * df0
            return i + 1, tf.where(ok, alpha, alpha * rho), ok

        alpha0 = tf.cast(self.line_search.step_size_initial, dtype)
        _, alpha, _ = tf.while_loop(
            cond,
            body,
            [tf.constant(0, tf.int32), alpha0, tf.constant(False)],
            parallel_iterations=1,
        )
        return alpha

    def _scalar_armijo_search(
        self,
        inputs: tf.Tensor,
        u: tf.Tensor,
        direction: tf.Tensor,
        f0: tf.Tensor,
        df0: tf.Tensor,
    ) -> tf.Tensor:
        """Armijo search using the scalar analytic cost."""
        dtype = u.dtype
        c1 = tf.cast(self.line_search.c1, dtype)
        rho = tf.cast(self.line_search.rho, dtype)
        max_iter = tf.constant(self.line_search.max_iter, tf.int32)

        def cond(i, alpha, accepted):
            del alpha
            return tf.logical_and(i < max_iter, tf.logical_not(accepted))

        def body(i, alpha, accepted):
            del accepted
            trial_cost = self.operator.scalar_cost_at(
                inputs, u + alpha * direction
            )
            accepted = trial_cost <= f0 + c1 * alpha * df0
            return i + 1, tf.where(accepted, alpha, alpha * rho), accepted

        _, alpha, _ = tf.while_loop(
            cond,
            body,
            (
                tf.constant(0, tf.int32),
                tf.cast(self.line_search.step_size_initial, dtype),
                tf.constant(False),
            ),
            parallel_iterations=1,
        )
        return alpha

    @tf.function(reduce_retracing=True)
    def _run_scalar_loop(
        self,
        inputs: tf.Tensor,
        iter_max: tf.Tensor,
        damping: tf.Tensor,
    ):
        """Read/update identity-map resources around one compiled solve call."""
        u_row = self.map.get_theta()[0][:, 0, 0, :]
        result = self._xla_scalar_newton_loop(
            u_row, inputs, iter_max, damping
        )
        self.map.set_theta(self.map.unflatten_theta(result.theta))
        return result

    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        inputs = inputs[0:1, :, :, :]
        compiled_loop = self._compiled_loop_enabled()

        if self.scalar_flowline and not compiled_loop:
            raise ValueError(
                "tridiag_newton.scalar_flowline requires the analytic XLA "
                "loop, a single relative-cost stopping rule, and disabled "
                "iteration display and debug output."
            )
        if compiled_loop:
            self._validate_scalar_flowline(inputs)
            if self.scalar_flowline:
                result = self._run_scalar_loop(
                    inputs,
                    tf.constant(self._iter_max_value, tf.int32),
                    tf.cast(self.damping, self.precision),
                )
            else:
                theta = self.map.flatten_theta(self.map.get_theta())
                result = self._xla_newton_loop(
                    theta,
                    inputs,
                    tf.constant(self._iter_max_value, tf.int32),
                    tf.cast(self.damping, self.precision),
                )
                self.map.set_theta(self.map.unflatten_theta(result.theta))

            grad_u_norm = tf.constant(float("nan"), self.precision)
            self._update_step_state(
                result.iterations - 1,
                result.U,
                result.V,
                result.theta,
                result.cost,
                grad_u_norm,
                result.grad_theta_norm,
            )
            self.halt_state = HaltState(
                result.halt_status,
                [result.relative_change],
                [result.criterion_satisfied],
            )
            # The solver stores this only as a diagnostic; retaining every
            # intermediate cost would require a TensorArray in the compiled
            # loop and inhibits XLA's device-side execution.
            return tf.reshape(result.cost, [1])

        theta = self.map.flatten_theta(self.map.get_theta())
        self._damping_var.assign(self.damping)
        U, V = self.map.get_UV(inputs)
        self._init_step_state(U, V, theta)

        halt_status = HaltStatus.CONTINUE.value
        costs = []

        for iteration in range(self._iter_max_value):
            (
                theta,
                cost,
                U,
                V,
                grad_u_norm,
                grad_theta_norm,
                halt_status,
                criterion_values,
                criterion_satisfied,
            ) = self._newton_step(
                theta,
                inputs,
                self._damping_var,
                tf.constant(iteration, tf.int32),
            )

            costs.append(cost)

            self._update_step_state(
                iteration,
                U,
                V,
                theta,
                cost,
                grad_u_norm,
                grad_theta_norm,
            )

            self.halt_state = HaltState(
                halt_status, criterion_values, criterion_satisfied
            )
            if self.display.enabled:
                self._update_display()

            if halt_status != HaltStatus.CONTINUE.value:
                break

        self._finalize_display(halt_status)
        return tf.stack(costs)
