#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Callable, Optional, Tuple, List

import tensorflow as tf

from .optimizer import Optimizer
from ..mappings import Mapping
from ..halt import Halt, HaltStatus
from .line_searches import LineSearches, ValueAndGradient
from .preconditioner import build_preconditioner, Preconditioner
from .energy_operator import Operator, ADOperator


class OptimizerCGNewton(Optimizer):
    """
    Matrix-free preconditioned Newton-CG optimizer.

    Each outer iteration solves ``(H + damping I) p = -g`` with a compiled PCG
    loop. Stable callables and variable-backed damping/preconditioner state let
    that graph be reused across Newton iterations.
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
        line_search_method: str = "armijo",
        line_search_compile: bool = True,
        alpha_min: float = 0.0,
        iter_max: int = 100,
        damping: float = 2e-2,
        damping_adaptive: bool = False,
        damping_min: float = 1e-12,
        damping_max: float = 1e2,
        damping_down: float = 0.25,
        damping_up: float = 4.0,
        cg_max_iter: int = 100,
        cg_tol: float = 1e-10,
        warm_start: bool = True,
        operator: Optional[Operator] = None,
        preconditioner: str = "block_jacobi",
        preconditioner_obj: Optional[Preconditioner] = None,
        precond_update_freq: int = 1,
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

        self.name = "cg_newton"
        self.line_search = LineSearches[line_search_method]()
        self.line_search_compile = bool(line_search_compile)
        if not self.line_search_compile and self.line_search.name != "armijo":
            raise ValueError(
                "cg_newton.line_search_compile=false currently supports only "
                "line_search=armijo; use line_search_compile=true for "
                f"line_search={self.line_search.name}."
            )

        self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
        self.alpha_min = tf.Variable(alpha_min, dtype=self.precision)

        self.damping = tf.constant(damping, dtype=self.precision)
        self.damping_adaptive = bool(damping_adaptive)
        self.damping_min = float(damping_min)
        self.damping_max = float(damping_max)
        self.damping_down = float(damping_down)
        self.damping_up = float(damping_up)
        self.cg_max_iter = tf.constant(cg_max_iter, dtype=tf.int32)
        self.cg_tol = tf.constant(cg_tol, dtype=self.precision)
        self.warm_start = bool(warm_start)
        self._p_prev: Optional[tf.Variable] = None
        self.last_cg_iterations = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.last_cg_relative_residual = tf.Variable(
            tf.cast(float("nan"), self.precision), trainable=False
        )

        self.operator: Operator = operator or ADOperator(cost_fn, self.map, precision)

        self.precond_update_freq = int(precond_update_freq)
        if preconditioner_obj is not None:
            self.preconditioner: Preconditioner = preconditioner_obj
        else:
            self.preconditioner = build_preconditioner(
                kind=preconditioner,
                mapping=self.map,
                precision=precision,
            )
        if getattr(self.preconditioner, "needs_operator", False):
            self.preconditioner.set_operator(self.operator)

        self._cg_A: Optional[Callable] = None
        self._line_search_eval_fn: Optional[Callable] = None
        self._line_search_theta: Optional[tf.Variable] = None
        self._line_search_p: Optional[tf.Variable] = None

    def update_parameters(self, iter_max: int, damping: float) -> None:
        self.iter_max.assign(iter_max)
        self.damping = tf.cast(damping, self.precision)

    def _cost_and_grad(self, inputs: tf.Tensor):
        return self.operator.cost_and_grad(inputs)

    def _refresh_preconditioner(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        if self.preconditioner.name != "identity":
            self.preconditioner.update(inputs, damping)

    @tf.function(reduce_retracing=True)
    def _cg_solve(
        self,
        b: tf.Tensor,
        x0: tf.Tensor,
        iter_max: tf.Tensor,
        tol: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Solve ``(H + damping I) x = b`` with compiled PCG."""
        A = self._cg_A
        pre = self.preconditioner.apply

        x = x0
        r = b - A(x)
        z = pre(r)
        d = z
        delta_new = tf.tensordot(r, z, axes=1)
        # A cold-start reference keeps the tolerance meaningful after warm starts.
        delta_0 = tf.tensordot(b, pre(b), axes=1)

        def cond(i, x, r, d, z, delta_new):
            del x, r, d, z
            return tf.logical_and(i < iter_max, delta_new > tol * tol * delta_0)

        def body(i, x, r, d, z, delta_new):
            q = A(d)
            alpha = delta_new / tf.tensordot(d, q, axes=1)
            x = x + alpha * d
            r = r - alpha * q
            z = pre(r)
            delta_old = delta_new
            delta_new = tf.tensordot(r, z, axes=1)
            beta = delta_new / delta_old
            d = z + beta * d
            return i + 1, x, r, d, z, delta_new

        i, x, r, _, _, _ = tf.while_loop(
            cond,
            body,
            [tf.constant(0, tf.int32), x, r, d, z, delta_new],
            parallel_iterations=1,
        )

        # Recompute the residual because recursive CG updates accumulate error.
        r_true = b - A(x)
        b_sq = tf.tensordot(b, b, axes=1)
        rs = tf.tensordot(r_true, r_true, axes=1)
        tiny = tf.cast(1e-30, b.dtype)
        relres = tf.sqrt(rs / tf.maximum(b_sq, tiny))
        return x, i, relres

    @tf.function
    def _get_grad_cg_newton(
        self, inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, List[tf.Tensor | tf.Variable], List[tf.Tensor | tf.Variable]]:
        cost, grad_u, grad_theta = self._cost_and_grad(inputs)
        return cost, grad_u, grad_theta

    def _force_descent(
        self, p_flat: tf.Tensor, grad_theta_flat: tf.Tensor, _: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        dot_gp = self._dot(grad_theta_flat, p_flat)
        return tf.cond(dot_gp >= 0.0, lambda: -grad_theta_flat, lambda: p_flat), None

    def _apply_step(
        self, theta_flat: tf.Tensor, alpha: tf.Tensor, p_flat: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        return theta_flat + alpha * p_flat, None

    @tf.function(reduce_retracing=True)
    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        dtype = self.precision
        return tf.tensordot(tf.cast(a, dtype), tf.cast(b, dtype), axes=1)

    def _cost_at(self, input: tf.Tensor, theta_flat: tf.Tensor) -> tf.Tensor:
        """Evaluate a trial point through the shared operator graph."""
        cost, _ = self.operator.cost_grad_at(input, theta_flat)
        return cost

    def _adapt_damping(self, damping: tf.Variable, alpha: tf.Tensor) -> None:
        dtype = damping.dtype
        decreased = tf.maximum(
            damping * tf.cast(self.damping_down, dtype),
            tf.cast(self.damping_min, dtype),
        )
        increased = tf.minimum(
            damping * tf.cast(self.damping_up, dtype),
            tf.cast(self.damping_max, dtype),
        )
        damping.assign(
            tf.where(
                alpha >= tf.cast(0.5, alpha.dtype),
                decreased,
                tf.where(alpha < tf.cast(0.1, alpha.dtype), increased, damping),
            )
        )

    def _line_search(
        self,
        theta_flat: tf.Tensor,
        p_flat: tf.Tensor,
        input: tf.Tensor,
        vg0: Optional[ValueAndGradient] = None,
    ) -> tf.Tensor:
        """Run the previous compiled search or a CG-local eager Armijo search."""
        if self.line_search_compile:
            self._line_search_theta.assign(theta_flat)
            self._line_search_p.assign(p_flat)
            return self.line_search.search(
                theta_flat,
                p_flat,
                self._line_search_eval_fn,
                val_0=vg0,
            )

        if vg0 is None:
            f0, grad0 = self.operator.cost_grad_at(input, theta_flat)
            vg0 = ValueAndGradient(
                x=tf.constant(0.0, dtype=theta_flat.dtype),
                f=f0,
                df=self._dot(grad0, p_flat),
            )

        f0 = float(vg0.f)
        df0 = float(vg0.df)
        alpha = float(self.line_search.step_size_initial)
        for _ in range(self.line_search.max_iter):
            alpha_tensor = tf.constant(alpha, dtype=theta_flat.dtype)
            f_alpha = float(
                self._cost_at(input, theta_flat + alpha_tensor * p_flat)
            )
            if f_alpha <= f0 + self.line_search.c1 * alpha * df0:
                break
            alpha *= self.line_search.rho

        return tf.constant(alpha, dtype=theta_flat.dtype)

    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        input = inputs[0:1, :, :, :]

        theta_flat = self.map.flatten_theta(self.map.get_theta())

        if self.warm_start and (
            self._p_prev is None or self._p_prev.shape != theta_flat.shape
        ):
            with tf.device(theta_flat.device):
                self._p_prev = tf.Variable(tf.zeros_like(theta_flat), trainable=False)

        # Keep the HVP callable stable; damping changes through its variable.
        _damping_var = tf.Variable(self.damping, dtype=self.precision, trainable=False)

        def _cg_A(v_flat: tf.Tensor) -> tf.Tensor:
            return self.operator.hvp(input, v_flat, _damping_var)

        self._cg_A = _cg_A

        # A stable callback avoids retracing; variables carry each trial state.
        with tf.device(theta_flat.device):
            self._line_search_theta = tf.Variable(theta_flat, trainable=False)
            self._line_search_p = tf.Variable(tf.zeros_like(theta_flat), trainable=False)

        def _line_search_eval(alpha: tf.Tensor) -> ValueAndGradient:
            trial = self._line_search_theta + alpha * self._line_search_p
            f, grad = self.operator.cost_grad_at(input, trial)
            return ValueAndGradient(
                x=alpha,
                f=f,
                df=self._dot(grad, self._line_search_p),
            )

        self._line_search_eval_fn = _line_search_eval

        U, V = self.map.get_UV(input)
        self._init_step_state(U, V, theta_flat)

        halt_status = HaltStatus.CONTINUE.value
        costs = []

        for iter in range(int(self.iter_max)):
            cost, grad_u, grad_theta = self._get_grad_cg_newton(input)
            grad_theta_flat = self.map.flatten_theta(grad_theta)

            # Freeze a banded Hessian once per Newton step; exact AD is a no-op.
            self.operator.prepare(input, _damping_var)

            if (iter % self.precond_update_freq) == 0:
                self._refresh_preconditioner(input, _damping_var)

            x0 = self._p_prev if self.warm_start else tf.zeros_like(grad_theta_flat)
            p_flat, _cg_iters, _cg_relres = self._cg_solve(
                b=-grad_theta_flat,
                x0=x0,
                iter_max=self.cg_max_iter,
                tol=self.cg_tol,
            )
            self.last_cg_iterations.assign(_cg_iters)
            self.last_cg_relative_residual.assign(_cg_relres)
            if self.warm_start:
                self._p_prev.assign(p_flat)

            p_flat, _ = self._force_descent(p_flat, grad_theta_flat, theta_flat)

            # Reuse the known value and slope at alpha=0.
            _vg0 = ValueAndGradient(
                x=tf.constant(0.0, dtype=theta_flat.dtype),
                f=cost,
                df=self._dot(grad_theta_flat, p_flat),
            )
            alpha = self._line_search(
                theta_flat=theta_flat,
                p_flat=p_flat,
                input=input,
                vg0=_vg0,
            )
            alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))

            theta_flat, _ = self._apply_step(theta_flat, alpha, p_flat)
            self.map.set_theta(self.map.unflatten_theta(theta_flat))

            costs.append(cost)

            U, V = self.map.get_UV(input)
            grad_u_norm, step_norm = self._get_grad_norm(grad_u, grad_theta)
            self._update_step_state(
                iter, U, V, theta_flat, cost, grad_u_norm, step_norm
            )

            halt_status = self._check_stopping()
            self._update_display()

            if self.damping_adaptive:
                self._adapt_damping(_damping_var, alpha)

            if halt_status != HaltStatus.CONTINUE.value:
                break

        self._finalize_display(halt_status)
        return tf.stack(costs)
