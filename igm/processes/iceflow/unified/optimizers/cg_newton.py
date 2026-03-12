#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, Optional, Tuple, List

from .optimizer import Optimizer
from ..mappings import Mapping
from ..halt import Halt, HaltStatus
from .line_searches import LineSearches, ValueAndGradient

from dataclasses import dataclass


@dataclass
class CGContext:
    """Context for CG solver to avoid passing many arguments."""

    inputs: tf.Tensor
    damping: tf.Tensor


class OptimizerCGNewton(Optimizer):
    """
    Matrix-free Newton–CG optimizer.

    Outer loop: Newton
    Inner loop: Truncated CG solve of (H + damping I) p = -g
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
        alpha_min: float = 0.0,
        iter_max: int = 100,
        damping: float = 2e-2,
        cg_max_iter: int = 100,
        cg_tol: float = 1e-10,
        truncated: bool = True,
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
        self.cost_fn = cost_fn

        self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
        self.alpha_min = tf.Variable(alpha_min, dtype=self.precision)

        self.damping = tf.constant(damping, dtype=self.precision)
        self.cg_max_iter = tf.constant(cg_max_iter, dtype=tf.int32)
        self.cg_tol = tf.constant(cg_tol, dtype=self.precision)
        self.truncated = tf.constant(truncated, dtype=tf.bool)
        self._p_prev = None

    def update_parameters(self, iter_max: int, damping: float) -> None:
        self.iter_max.assign(iter_max)
        self.damping = tf.cast(damping, self.precision)

    def _cost_and_grad(
        self,
        inputs: tf.Tensor
    ):
        """Compute cost and gradient w.r.t. theta."""
        theta = self.map.get_theta()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for theta_i in theta:
                tape.watch(theta_i)
            U, V = self.map.get_UV(inputs)
            cost = self.cost_fn(U, V, inputs)

        grads = tape.gradient(cost, [U, V] + theta)
        grad_u = tuple(grads[:2])
        grad_theta = grads[2:]
    
        
        return cost, grad_u, grad_theta

    def _hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: float,
    ) -> tf.Tensor:
        """Compute (H + damping I) v using Reverse-over-reverse hessian-vector product.
        We choose to not do the perlmutter trick (forward-over-reverse) as it
        complicates the graph structure despite it being more memory efficient."""

        theta = self.map.get_theta()

        with tf.GradientTape(watch_accessed_variables=False) as outer_tape:
            for theta_i in theta:
                outer_tape.watch(theta_i)
            with tf.GradientTape(watch_accessed_variables=False) as inner_tape:
                for theta_i in theta:
                    inner_tape.watch(theta_i)
                U, V = self.map.get_UV(
                    inputs
                )  # This line adds so much time for tracing ...
                cost = self.cost_fn(U, V, inputs)

            grad_theta = inner_tape.gradient(cost, theta)

        v = self.map.unflatten_theta(
            v_flat
        )  # v has same structure as theta in parameter space (and function space) (hence why we use unflatten_theta)
        
        Hv_theta = outer_tape.gradient(
            grad_theta,
            theta,
            output_gradients=v,
        )

        Hv_theta_flat = tf.concat(
            [tf.reshape(h, (-1,)) for h in Hv_theta],
            axis=0,
        )

        return Hv_theta_flat + damping * v_flat

    def _explicit_residual(
        self,
        inputs: tf.Tensor,
        x: tf.Tensor,
        b: tf.Tensor,
        damping: float
    ) -> tf.Tensor:
        """Compute explicit residual: r = b - Ax."""
        Ax = self._hvp(inputs, x, damping)
        return b - Ax

    def _implicit_residual(
        self,
        r: tf.Tensor,
        alpha: tf.Tensor,
        Ap: tf.Tensor,
    ) -> tf.Tensor:
        """Compute implicit residual: r = r - alpha * Ap."""
        return r - alpha * Ap

    def _cg_solve(
        self,
        inputs: tf.Tensor,
        b: tf.Tensor,
        damping: float,
        cg_max_iter: int,
        cg_tol: float,
        is_truncated: bool,
        explicit_freq: int = 50,
    ) -> tf.Tensor:
        """Solve (H + damping I) x = b using truncated linear CG + Fletcher Reeves quotient.
        Using explicit and implicit residual updates for stopping criteria with a tradeoff of speed.
        """

        # Warm start
        if self._p_prev is not None:
            x = 1.0 * self._p_prev  # scale down for stability?
        else:
            x = tf.zeros_like(b)
            
        r = b
        p = r
        rs = tf.tensordot(r, r, axes=1)

        for i in tf.range(cg_max_iter):
            if rs <= cg_tol:
                break

            Ap = self._hvp(inputs, p, damping)
            pAp = tf.tensordot(p, Ap, axes=1)

            if is_truncated and pAp <= 0.0:
                if i == 0:
                    x = r
                break

            alpha = rs / pAp
            x = x + alpha * p

            should_compute_explicit = tf.logical_and(
                tf.equal(tf.math.floormod(i, explicit_freq), 0), i > 0
            )

            r_new = tf.cond(
                should_compute_explicit,
                lambda: self._explicit_residual(inputs, x, b, damping),
                lambda: self._implicit_residual(r, alpha, Ap),
            )

            rs_new = tf.tensordot(r_new, r_new, axes=1)
            beta = rs_new / rs
            p = r_new + beta * p

            r = r_new
            rs = rs_new

        self._p_prev = x 
        
        return x

    def _get_grad_cg_newton(
        self,
        inputs: tf.Tensor
    ) -> Tuple[tf.Tensor, List[tf.Tensor | tf.Variable], List[tf.Tensor | tf.Variable]]:
        """Compute cost and gradients (function and parameter space)."""

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

    def _line_search(
        self,
        theta_flat: tf.Tensor,
        p_flat: tf.Tensor,
        input: tf.Tensor,
    ) -> tf.Tensor:
        def eval_fn(alpha: tf.Tensor) -> ValueAndGradient:
            theta_backup = self.map.copy_theta(self.map.get_theta())
            theta_alpha, _ = self._apply_step(theta_flat, alpha, p_flat)

            self.map.set_theta(self.map.unflatten_theta(theta_alpha))
            f, _, grad_theta = self._get_grad_cg_newton(input)
            grad_flat = self.map.flatten_theta(grad_theta)
            df = self._dot(grad_flat, p_flat)

            self.map.set_theta(theta_backup)
            return ValueAndGradient(x=alpha, f=f, df=df)

        return self.line_search.search(theta_flat, p_flat, eval_fn)

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        
        
        first_batch = self.sampler(inputs)
        if first_batch.shape[0] != 1:
            raise NotImplementedError("Newton–CG requires a single batch.")

        input = first_batch[0, :, :, :, :]

        # Extract values to pass as arguments
        damping = self.damping
        cg_max_iter = self.cg_max_iter
        cg_tol = self.cg_tol
        is_truncated = self.truncated

        theta_flat = self.map.flatten_theta(self.map.get_theta())

        cost, grad_u, grad_theta = self._get_grad_cg_newton(input)

        U, V = self.map.get_UV(input)
        self._init_step_state(U, V, theta_flat)

        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = tf.constant(-1, dtype=tf.int32)
        costs = tf.TensorArray(dtype=cost.dtype, size=int(self.iter_max))

        for iter in tf.range(self.iter_max):

            cost, grad_u, grad_theta = self._get_grad_cg_newton(input)
            grad_theta_flat = self.map.flatten_theta(grad_theta)

            p_flat = self._cg_solve(
                inputs=inputs,
                b=-grad_theta_flat,
                damping=damping,
                cg_max_iter=cg_max_iter,
                cg_tol=cg_tol,
                is_truncated=is_truncated,
            )

            p_flat, _ = self._force_descent(p_flat, grad_theta_flat, theta_flat)

            alpha = self._line_search(
                theta_flat=theta_flat,
                p_flat=p_flat,
                input=input,
            )
            alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))

            theta_flat, _ = self._apply_step(theta_flat, alpha, p_flat)
            self.map.set_theta(self.map.unflatten_theta(theta_flat))

            costs = costs.write(iter, cost)

            U, V = self.map.get_UV(input) # unecessarily doing another forward step - but cleaner for now..
            grad_u_norm, step_norm = self._get_grad_norm(grad_u, grad_theta)
            self._update_step_state(
                iter, U, V, theta_flat, cost, grad_u_norm, step_norm
            )

            halt_status = self._check_stopping()
            self._update_display()

            iter_last = iter
            if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
                break

        self._finalize_display(halt_status)
        return costs.stack()[: iter_last + 1]