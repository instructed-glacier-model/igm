#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""GPU-oriented generalized Gauss--Newton optimization of network weights."""

import time
from typing import Callable, Optional, Tuple

import tensorflow as tf

from .cg_newton import OptimizerCGNewton
from ..halt import Halt, HaltStatus
from ..mappings import Mapping
from ..operators import Operator
from ..preconditioners import Preconditioner


class OptimizerGaussNewton(OptimizerCGNewton):
    """Inexact Gauss--Newton optimizer with LM damping and matrix-free PCG."""

    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        operator: Operator,
        halt: Optional[Halt] = None,
        print_cost: bool = True,
        print_cost_freq: int = 1,
        precision: str = "float32",
        ord_grad_u: str = "l2_weighted",
        ord_grad_theta: str = "l2_weighted",
        iter_max: int = 100,
        cg_max_iter: int = 50,
        preconditioner: str = "identity",
        preconditioner_obj: Optional[Preconditioner] = None,
        precond_update_freq: int = 3,
        print_timing: bool = False,
        debug_mode: bool = False,
        debug_freq: int = 100,
        forcing_eta_min: float = 1.0e-2,
        forcing_eta_max: float = 0.5,
        forcing_gamma: float = 0.9,
        forcing_power: float = 1.5,
        damping_rel_init: float = 1.0e-2,
        damping_rel_min: float = 1.0e-8,
        damping_rel_max: float = 1.0e2,
        lm_ratio_low: float = 0.25,
        lm_ratio_high: float = 0.75,
        damping_down: float = 0.25,
        damping_up: float = 4.0,
        solve_retries: int = 3,
        power_iterations: int = 6,
        armijo_c1: float = 1.0e-4,
        armijo_contraction: float = 0.5,
        line_search_max_iter: int = 12,
    ):
        super().__init__(
            cost_fn=cost_fn,
            map=map,
            halt=halt,
            print_cost=print_cost,
            print_cost_freq=print_cost_freq,
            precision=precision,
            ord_grad_u=ord_grad_u,
            ord_grad_theta=ord_grad_theta,
            line_search_method="armijo",
            line_search_compile=False,
            print_timing=print_timing,
            iter_max=iter_max,
            damping=1.0,
            damping_adaptive=False,
            damping_down=damping_down,
            damping_up=damping_up,
            cg_max_iter=cg_max_iter,
            cg_tol=forcing_eta_min,
            warm_start=False,
            operator=operator,
            preconditioner=preconditioner,
            preconditioner_obj=preconditioner_obj,
            operator_update_freq=1,
            precond_update_freq=precond_update_freq,
            debug_mode=debug_mode,
            debug_freq=debug_freq,
        )
        self.name = "gauss_newton"
        self.damping = tf.Variable(
            self.damping,
            dtype=self.precision,
            trainable=False,
            name="gauss_newton_damping",
        )

        if not 0.0 < forcing_eta_min <= forcing_eta_max <= 1.0:
            raise ValueError(
                "Gauss-Newton forcing tolerances must satisfy "
                "0 < eta_min <= eta_max <= 1."
            )
        if forcing_gamma <= 0.0 or forcing_power <= 0.0:
            raise ValueError(
                "Gauss-Newton forcing gamma and power must be positive."
            )
        if not 0.0 < damping_rel_min <= damping_rel_init <= damping_rel_max:
            raise ValueError(
                "Gauss-Newton relative damping must satisfy min <= init <= max."
            )
        if not 0.0 <= lm_ratio_low < lm_ratio_high:
            raise ValueError(
                "Gauss-Newton LM ratio thresholds must be ordered and nonnegative."
            )
        if not 0.0 < damping_down < 1.0 or damping_up <= 1.0:
            raise ValueError(
                "Gauss-Newton damping factors must satisfy "
                "0 < damping_down < 1 < damping_up."
            )
        if solve_retries < 0 or power_iterations < 1:
            raise ValueError(
                "Gauss-Newton retries must be nonnegative and power "
                "iterations positive."
            )
        if not 0.0 < armijo_c1 < 1.0:
            raise ValueError("Gauss-Newton Armijo c1 must be in (0, 1).")
        if not 0.0 < armijo_contraction < 1.0:
            raise ValueError("Gauss-Newton Armijo contraction must be in (0, 1).")
        if line_search_max_iter < 1:
            raise ValueError(
                "Gauss-Newton line_search_max_iter must be positive."
            )

        dtype = self.precision
        self.forcing_eta_min = tf.constant(forcing_eta_min, dtype)
        self.forcing_eta_max = tf.constant(forcing_eta_max, dtype)
        self.forcing_gamma = tf.constant(forcing_gamma, dtype)
        self.forcing_power = tf.constant(forcing_power, dtype)
        self.damping_rel_init = tf.constant(damping_rel_init, dtype)
        self.damping_rel_min = tf.constant(damping_rel_min, dtype)
        self.damping_rel_max = tf.constant(damping_rel_max, dtype)
        self.lm_ratio_low = tf.constant(lm_ratio_low, dtype)
        self.lm_ratio_high = tf.constant(lm_ratio_high, dtype)
        self.solve_retries = int(solve_retries)
        self.power_iterations = int(power_iterations)
        self.armijo_c1 = tf.constant(armijo_c1, dtype)
        self.armijo_contraction = tf.constant(armijo_contraction, dtype)
        self.line_search_max_iter = tf.constant(line_search_max_iter, tf.int32)
        self._lambda_scale = tf.Variable(
            tf.cast(1.0, dtype),
            trainable=False,
            name="gauss_newton_lambda_scale",
        )
        self.last_step_length = tf.Variable(
            tf.cast(float("nan"), dtype), trainable=False
        )
        self.last_lm_ratio = tf.Variable(
            tf.cast(float("nan"), dtype), trainable=False
        )
        self.last_forcing_tolerance = tf.Variable(
            self.forcing_eta_max, trainable=False
        )

    def update_parameters(self, iter_max: int) -> None:
        self.iter_max.assign(iter_max)

    def _set_preconditioner_damping(self) -> None:
        setter = getattr(self.preconditioner, "set_damping", None)
        if setter is not None:
            setter(self.damping)

    def _solve(
        self,
        inputs: tf.Tensor,
        b: tf.Tensor,
        damping: tf.Tensor,
        tolerance: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Solve the damped Gauss-Newton system and retain its residual."""
        # Use one damping tensor signature for spectrum estimation and PCG.
        damping = tf.convert_to_tensor(damping, dtype=self.precision)
        precondition = self.preconditioner.apply
        x = tf.zeros_like(b)
        residual = b
        z = precondition(residual)
        direction = z
        delta = tf.tensordot(residual, z, axes=1)
        delta_initial = tf.maximum(
            tf.tensordot(b, precondition(b), axes=1),
            tf.cast(0.0, b.dtype),
        )
        negative_curvature = tf.constant(False)
        threshold = tolerance * tolerance * delta_initial
        iterations = 0

        # An eager recurrence reuses the HVP graph without tracing a nested copy.
        for iteration in range(int(self.cg_max_iter)):
            keep_going = tf.logical_and(
                delta > threshold,
                tf.logical_and(
                    tf.math.is_finite(delta),
                    tf.logical_not(negative_curvature),
                ),
            )
            if not bool(keep_going.numpy()):
                break

            product = self.operator.hvp(inputs, direction, damping)
            curvature = tf.tensordot(direction, product, axes=1)
            bad = tf.logical_or(
                curvature <= tf.cast(0.0, curvature.dtype),
                tf.logical_not(tf.math.is_finite(curvature)),
            )
            iterations = iteration + 1
            if bool(bad.numpy()):
                negative_curvature = tf.constant(True)
                break

            alpha = delta / curvature
            x = x + alpha * direction
            residual = residual - alpha * product
            z = precondition(residual)
            next_delta = tf.tensordot(residual, z, axes=1)
            beta = next_delta / tf.maximum(
                delta, tf.cast(1.0e-30, delta.dtype)
            )
            direction = z + beta * direction
            delta = next_delta

        b_norm_squared = tf.tensordot(b, b, axes=1)
        residual_norm_squared = tf.tensordot(residual, residual, axes=1)
        relative_residual = tf.sqrt(
            residual_norm_squared
            / tf.maximum(b_norm_squared, tf.cast(1.0e-30, b.dtype))
        )
        return (
            x,
            tf.constant(iterations, tf.int32),
            relative_residual,
            residual,
            negative_curvature,
        )

    def _power_iteration(self, inputs: tf.Tensor, shape: tf.Tensor) -> tf.Tensor:
        vector = tf.random.stateless_normal(
            shape, seed=[1729, 0], dtype=self.precision
        )
        tiny = tf.cast(1.0e-30, self.precision)
        vector /= tf.maximum(tf.norm(vector), tiny)
        eigenvalue = tf.cast(0.0, self.precision)
        zero = tf.cast(0.0, self.precision)

        # Reuse the standalone HVP graph also used by PCG.
        for _ in range(self.power_iterations):
            product = self.operator.hvp(inputs, vector, zero)
            norm = tf.norm(product)
            vector = product / tf.maximum(norm, tiny)
            eigenvalue = tf.abs(tf.tensordot(vector, product, axes=1))
        return eigenvalue

    @tf.function(autograph=False, reduce_retracing=True, jit_compile=True)
    def _armijo_search(
        self,
        inputs: tf.Tensor,
        theta_flat: tf.Tensor,
        direction: tf.Tensor,
        slope: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # Keep reference and trial costs on the same stateless fp32 path.
        reference_cost = self.operator.cost_at(inputs, theta_flat)
        alpha = tf.cast(1.0, theta_flat.dtype)
        trial_cost = reference_cost
        accepted = tf.constant(False)

        def cond(iteration, alpha, trial_cost, accepted):
            del alpha, trial_cost
            return tf.logical_and(
                iteration < self.line_search_max_iter,
                tf.logical_not(accepted),
            )

        def body(iteration, alpha, trial_cost, accepted):
            del accepted
            candidate = theta_flat + alpha * direction
            candidate_cost = self.operator.cost_at(inputs, candidate)
            sufficient = candidate_cost <= (
                reference_cost + self.armijo_c1 * alpha * slope
            )
            good = tf.logical_and(tf.math.is_finite(candidate_cost), sufficient)
            next_alpha = tf.where(
                good, alpha, alpha * self.armijo_contraction
            )
            return iteration + 1, next_alpha, candidate_cost, good

        evaluations, alpha, trial_cost, accepted = tf.while_loop(
            cond,
            body,
            (tf.constant(0, tf.int32), alpha, trial_cost, accepted),
            parallel_iterations=1,
        )
        return alpha, trial_cost, reference_cost, accepted, evaluations

    def _increase_damping(self) -> None:
        maximum = self.damping_rel_max * self._lambda_scale
        self.damping.assign(
            tf.minimum(
                self.damping * tf.cast(self.damping_up, self.precision),
                maximum,
            )
        )
        self._set_preconditioner_damping()

    def _adapt_lm_damping(self, ratio: tf.Tensor) -> None:
        minimum = self.damping_rel_min * self._lambda_scale
        maximum = self.damping_rel_max * self._lambda_scale
        reduced = tf.maximum(
            self.damping * tf.cast(self.damping_down, self.precision), minimum
        )
        increased = tf.minimum(
            self.damping * tf.cast(self.damping_up, self.precision), maximum
        )
        self.damping.assign(
            tf.where(
                ratio > self.lm_ratio_high,
                reduced,
                tf.where(ratio < self.lm_ratio_low, increased, self.damping),
            )
        )
        self._set_preconditioner_damping()

    def tracing_counts(self) -> dict[str, int]:
        """Return concrete-function counts for the expensive graphs."""
        functions = {
            "gradient": self.operator.cost_and_grad,
            "weight_hvp": self.operator.hvp,
            "armijo": self._armijo_search,
            "cost": self.operator.cost_at,
            "preconditioner": self.preconditioner.apply,
        }
        velocity_operator = getattr(self.operator, "velocity_operator", None)
        velocity_ad = getattr(velocity_operator, "_ad", None)
        if velocity_ad is not None:
            functions["velocity_hvp"] = velocity_ad._hvp

        counts = {}
        for name, function in functions.items():
            counter = getattr(function, "experimental_get_tracing_count", None)
            counts[name] = int(counter()) if counter is not None else 0
        return counts

    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        # Optimize the aggregate objective over all input patches.
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        U, V = self.map.get_UV(inputs)
        self._init_step_state(U, V, theta_flat)

        zero = tf.cast(0.0, self.precision)
        setup_start = time.perf_counter()
        self.operator.prepare(inputs, zero)
        self._synchronize_setup()
        initial_prepare_seconds = time.perf_counter() - setup_start
        initial_sketch_seconds = 0.0
        if self.preconditioner.name != "identity":
            sketch_start = time.perf_counter()
            self.preconditioner.update(inputs, zero)
            self._synchronize_setup()
            initial_sketch_seconds = time.perf_counter() - sketch_start
        spectrum = getattr(self.preconditioner, "lambda_max", None)
        initial_spectrum_seconds = 0.0
        if spectrum is None:
            power_start = time.perf_counter()
            spectrum = self._power_iteration(inputs, tf.shape(theta_flat))
            spectrum.numpy()
            initial_spectrum_seconds = time.perf_counter() - power_start
        lambda_floor = tf.cast(
            1.0e-12 if self.precision == tf.float64 else 1.0e-6,
            self.precision,
        )
        self._lambda_scale.assign(tf.maximum(spectrum, lambda_floor))
        self.damping.assign(self.damping_rel_init * self._lambda_scale)
        self._set_preconditioner_damping()
        self._synchronize_setup()

        previous_gradient_norm = None
        halt_status = HaltStatus.CONTINUE.value
        costs = []

        for iteration in range(int(self.iter_max)):
            iteration_start = time.perf_counter()
            gradient_start = time.perf_counter()
            cost, grad_u, grad_theta = self.operator.cost_and_grad(inputs)
            grad_flat = self.map.flatten_theta(grad_theta)
            gradient_norm_euclidean = tf.norm(grad_flat)
            cost.numpy()
            gradient_seconds = time.perf_counter() - gradient_start

            if previous_gradient_norm is None:
                forcing = self.forcing_eta_max
            else:
                ratio = gradient_norm_euclidean / tf.maximum(
                    previous_gradient_norm,
                    tf.cast(1.0e-30, self.precision),
                )
                forcing = tf.clip_by_value(
                    self.forcing_gamma * tf.pow(ratio, self.forcing_power),
                    self.forcing_eta_min,
                    self.forcing_eta_max,
                )
            self.last_forcing_tolerance.assign(forcing)

            prepare_seconds = 0.0
            if iteration > 0:
                start = time.perf_counter()
                self.operator.prepare(inputs, zero)
                self._synchronize_setup()
                prepare_seconds = time.perf_counter() - start

            if iteration == 0:
                prepare_seconds = initial_prepare_seconds
            sketch_seconds = initial_sketch_seconds if iteration == 0 else 0.0
            if (
                iteration > 0
                and self.preconditioner.name != "identity"
                and iteration % self.precond_update_freq == 0
            ):
                start = time.perf_counter()
                self.preconditioner.update(inputs, self.damping)
                self._synchronize_setup()
                sketch_seconds = time.perf_counter() - start

            accepted = False
            cg_seconds = 0.0
            line_search_seconds = 0.0
            cg_iterations = 0
            line_evaluations = 0
            relative_residual = float("nan")
            alpha_value = float("nan")
            lm_ratio_value = float("nan")
            accepted_cost = cost
            last_negative = False
            last_slope = float("nan")

            for _ in range(self.solve_retries + 1):
                self._set_preconditioner_damping()
                start = time.perf_counter()
                direction, cg_it, cg_relres, residual, negative = (
                    self._solve(
                        inputs,
                        -grad_flat,
                        self.damping,
                        forcing,
                    )
                )
                direction[0:1].numpy()
                cg_seconds += time.perf_counter() - start
                cg_iterations = int(cg_it.numpy())
                relative_residual = float(cg_relres.numpy())
                last_negative = bool(negative.numpy())
                self.last_cg_iterations.assign(cg_it)
                self.last_cg_relative_residual.assign(cg_relres)

                slope = self._dot(grad_flat, direction)
                last_slope = float(slope.numpy())
                direction_ok = bool(
                    tf.logical_and(
                        tf.logical_not(negative),
                        tf.logical_and(
                            tf.math.is_finite(slope), slope < tf.cast(0.0, slope.dtype)
                        ),
                    ).numpy()
                )
                if not direction_ok:
                    self._increase_damping()
                    continue

                start = time.perf_counter()
                alpha, trial_cost, reference_cost, armijo_ok, evaluations = (
                    self._armijo_search(
                        inputs, theta_flat, direction, slope
                    )
                )
                trial_cost.numpy()
                line_search_seconds += time.perf_counter() - start
                line_evaluations = int(evaluations.numpy())

                # Recover p.T G p from the CG residual without another HVP.
                damped_product = -grad_flat - residual
                p_A_p = self._dot(direction, damped_product)
                p_G_p = p_A_p - self.damping * self._dot(direction, direction)
                predicted = (
                    -alpha * slope
                    - tf.cast(0.5, self.precision) * alpha * alpha * p_G_p
                )
                actual = reference_cost - trial_cost
                lm_ratio = actual / tf.maximum(
                    predicted, tf.cast(1.0e-30, self.precision)
                )
                model_ok = tf.logical_and(
                    armijo_ok,
                    tf.logical_and(
                        tf.logical_and(tf.math.is_finite(predicted), predicted > 0.0),
                        tf.logical_and(tf.math.is_finite(lm_ratio), actual >= 0.0),
                    ),
                )
                if not bool(model_ok.numpy()):
                    self._increase_damping()
                    continue

                accepted = True
                accepted_cost = trial_cost
                alpha_value = float(alpha.numpy())
                lm_ratio_value = float(lm_ratio.numpy())
                self.last_step_length.assign(alpha)
                self.last_lm_ratio.assign(lm_ratio)
                theta_flat = theta_flat + alpha * direction
                self.map.set_theta(self.map.unflatten_theta(theta_flat))
                self._adapt_lm_damping(lm_ratio)
                break

            if not accepted:
                raise RuntimeError(
                    "gauss_newton could not obtain a finite descent step after "
                    f"{self.solve_retries + 1} damped solves at iteration "
                    f"{iteration}: damping={float(self.damping.numpy()):.3e}, "
                    f"cg_iterations={cg_iterations}, "
                    f"cg_relative_residual={relative_residual:.3e}, "
                    f"negative_curvature={last_negative}, "
                    f"slope={last_slope:.3e}, "
                    f"line_evaluations={line_evaluations}."
                )

            costs.append(accepted_cost)
            previous_gradient_norm = gradient_norm_euclidean
            U, V = self.map.get_UV(inputs)
            grad_u_norm, grad_theta_norm = self._get_grad_norm(
                grad_u, grad_theta
            )
            self._update_step_state(
                iteration,
                U,
                V,
                theta_flat,
                accepted_cost,
                grad_u_norm,
                grad_theta_norm,
            )
            halt_status = self._check_stopping()
            self._update_display()

            if self.print_timing:
                total_seconds = time.perf_counter() - iteration_start
                spectrum_seconds = (
                    initial_spectrum_seconds if iteration == 0 else 0.0
                )
                if iteration == 0:
                    total_seconds += (
                        initial_prepare_seconds
                        + initial_sketch_seconds
                        + initial_spectrum_seconds
                    )
                print(
                    f"[gauss_newton_timing] iter={iteration:3d} "
                    f"grad={gradient_seconds:.3f}s "
                    f"prepare={prepare_seconds:.3f}s "
                    f"sketch={sketch_seconds:.3f}s "
                    f"spectrum={spectrum_seconds:.3f}s "
                    f"cg={cg_seconds:.3f}s({cg_iterations}it, "
                    f"relres={relative_residual:.2e}) "
                    f"line={line_search_seconds:.3f}s({line_evaluations}eval) "
                    f"total={total_seconds:.3f}s "
                    f"eta={float(forcing.numpy()):.2e} "
                    f"mu={float(self.damping.numpy()):.2e} "
                    f"alpha={alpha_value:.2e} rho={lm_ratio_value:.2e} "
                    f"cost={float(accepted_cost.numpy()):.8e} "
                    f"grad_theta={float(gradient_norm_euclidean.numpy()):.2e}",
                    flush=True,
                )

            if halt_status != HaltStatus.CONTINUE.value:
                break

        if self.print_timing:
            final_cost, _, final_gradient = self.operator.cost_and_grad(inputs)
            final_gradient_norm = tf.norm(
                self.map.flatten_theta(final_gradient)
            )
            final_cost.numpy()
            print(
                f"[gauss_newton_final] cost={float(final_cost.numpy()):.8e} "
                f"grad_theta={float(final_gradient_norm.numpy()):.8e}",
                flush=True,
            )
            counts = " ".join(
                f"{name}={count}" for name, count in self.tracing_counts().items()
            )
            print(f"[gauss_newton_traces] {counts}", flush=True)

        self._finalize_display(halt_status)
        return tf.stack(costs)
