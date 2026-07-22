# optimizer_cg.py
import tensorflow as tf
from typing import Callable
from ..mappings import Mapping
from .optimizer import Optimizer
from .line_searches import LineSearches, ValueAndGradient

class OptimizerCG(Optimizer):
    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        print_cost: bool,
        print_cost_freq: int,
        precision: str,
        line_search_method: str = "strong_wolfe",
        iter_max: int = int(1e5),
        alpha_min: float = 0.0,
        variant: str = "PR+",
        restart_every: int = 50,
    ):
        super().__init__(cost_fn, map, print_cost, print_cost_freq, precision)
        self.name = "cg"
        self.iter_max = tf.Variable(iter_max)
        self.alpha_min = tf.Variable(tf.cast(alpha_min, self.precision), dtype=self.precision)
        self.variant = variant
        self.restart_every = restart_every
        self.line_search = LineSearches[line_search_method]()

        # Optional bounds support if mapping provides them (like in LBFGS)
        self._has_box_bounds = hasattr(map, "get_box_bounds_flat")

    def update_parameters(self, iter_max: int, alpha_min: float, **_):
        self.iter_max.assign(iter_max)
        self.alpha_min.assign(tf.cast(alpha_min, self.precision))

    @tf.function
    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.tensordot(tf.cast(a, self.precision), tf.cast(b, self.precision), axes=1)

    def _beta(self, g_new, g_old, p_old):
        if self.variant == "FR":
            return self._dot(g_new, g_new) / (self._dot(g_old, g_old) + 1e-30)
        if self.variant == "HS":
            y = g_new - g_old
            return self._dot(g_new, y) / (self._dot(p_old, y) + 1e-30)
        if self.variant == "DY":
            y = g_new - g_old
            return self._dot(g_new, g_new) / (self._dot(p_old, y) + 1e-30)
        # PR+ default
        y = g_new - g_old
        beta_pr = self._dot(g_new, y) / (self._dot(g_old, g_old) + 1e-30)
        return tf.maximum(beta_pr, tf.cast(0.0, self.precision))

    @tf.function(reduce_retracing=True)
    def _project_box(self, theta_flat: tf.Tensor, L_flat: tf.Tensor, U_flat: tf.Tensor) -> tf.Tensor:
        """Project parameters onto box constraints [L, U]."""
        return tf.minimum(tf.maximum(theta_flat, L_flat), U_flat)

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        # Mirror LBFGS: assume single batch first
        n_batches = inputs.shape[0]
        tf.debugging.assert_equal(n_batches, 1, "CG currently expects a single batch (like LBFGS).")
        x = inputs[0, :, :, :, :]

        # Start point
        theta_flat = self.map.flatten_theta(self.map.get_theta())

        # Optional θ-space bounds (only if mapping provides them)
        if self._has_box_bounds:
            L_flat, U_flat = self.map.get_box_bounds_flat()
            # Validate dtype matches optimizer precision
            tf.debugging.assert_type(L_flat, self.precision, message="[CG] lower bounds must match optimizer dtype")
            tf.debugging.assert_type(U_flat, self.precision, message="[CG] upper bounds must match optimizer dtype")
            # Start feasible
            theta_flat = self._project_box(theta_flat, L_flat, U_flat)
            self.map.set_theta(self.map.unflatten_theta(theta_flat))
        else:
            # Dummy tensors to satisfy TF signature; never used when _has_box_bounds is False
            L_flat = tf.zeros_like(theta_flat)
            U_flat = tf.zeros_like(theta_flat)

        # Initial value/grad
        cost, grad_theta = self._get_grad(x)
        g = self.map.flatten_theta(grad_theta)
        p = -g

        costs = tf.TensorArray(dtype=self.precision, size=int(self.iter_max))

        for it in tf.range(self.iter_max, dtype=tf.int32):
            # Record current cost at start of iteration
            costs = costs.write(it, cost)
            
            # Check stopping criteria
            grad_norm = self._get_grad_norm(grad_theta)
            self.map.on_step_end(it)
            should_stop = self._progress_update(it, cost, grad_norm)
            if should_stop:
                break
            
            # Line search along p (projected-path if bounded)
            def val_grad(alpha):
                theta_bak = self.map.copy_theta(self.map.get_theta())
                theta_trial = theta_flat + alpha * p
                if self._has_box_bounds:
                    theta_trial = self._project_box(theta_trial, L_flat, U_flat)
                self.map.set_theta(self.map.unflatten_theta(theta_trial))
                f, grad = self._get_grad(x)
                df = self._dot(self.map.flatten_theta(grad), p)
                self.map.set_theta(theta_bak)
                return ValueAndGradient(x=alpha, f=f, df=df)

            def value_fn(alpha):
                theta_bak = self.map.copy_theta(self.map.get_theta())
                theta_trial = theta_flat + alpha * p
                if self._has_box_bounds:
                    theta_trial = self._project_box(theta_trial, L_flat, U_flat)
                self.map.set_theta(self.map.unflatten_theta(theta_trial))
                f = self._get_cost(x)
                self.map.set_theta(theta_bak)
                return f

            val_0 = ValueAndGradient(
                x=tf.constant(0.0, dtype=theta_flat.dtype),
                f=tf.cast(cost, theta_flat.dtype),
                df=tf.cast(self._dot(g, p), theta_flat.dtype),
            )
            alpha = self.line_search.search(
                theta_flat,
                p,
                val_grad,
                val_0=val_0,
                value_fn=self.line_search.select_value_fn(value_fn),
            )
            
            # Safeguard against NaN or invalid step sizes
            alpha = tf.where(tf.math.is_nan(alpha) | tf.math.is_inf(alpha), 
                            tf.constant(0.0, dtype=alpha.dtype), 
                            alpha)
            alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))

            # Step (and project if bounded)
            theta_new = theta_flat + alpha * p
            if self._has_box_bounds:
                theta_new = self._project_box(theta_new, L_flat, U_flat)
            self.map.set_theta(self.map.unflatten_theta(theta_new))

            # New grad/cost for next iteration
            cost, grad_theta = self._get_grad(x)
            g_new = self.map.flatten_theta(grad_theta)
            
            # Check for NaN/Inf in cost or gradient - stop if detected (use tf.cond for graph mode)
            cost_is_invalid = tf.math.is_nan(cost) | tf.math.is_inf(cost)
            grad_is_invalid = tf.reduce_any(tf.math.is_nan(g_new) | tf.math.is_inf(g_new))
            numerical_issues = cost_is_invalid | grad_is_invalid
            
            # Use previous valid cost/grad if NaN detected, and stop
            cost = tf.where(numerical_issues, costs.read(it), cost)
            g_new = tf.where(numerical_issues, g, g_new)

            # Restart conditions: periodic restart or loss of conjugacy
            # Loss of conjugacy: when gradient becomes too aligned with previous search direction
            periodic_restart = tf.equal((it + 1) % self.restart_every, 0)
            loss_of_conjugacy = self._dot(g_new, g) <= 0.0  # New grad orthogonal/opposite to old grad
            restart = tf.logical_or(periodic_restart, loss_of_conjugacy)
            
            # Compute beta and new search direction
            beta = self._beta(g_new, g, p)
            p = tf.where(restart, -g_new, -g_new + beta * p)

            # Update loop vars
            theta_flat = theta_new
            g = g_new

        # Fill remaining slots with final cost (in case of early termination)
        # This ensures costs[-1] gives the actual final cost, not 0.0
        final_cost = cost
        for fill_it in tf.range(it + 1, self.iter_max):
            costs = costs.write(fill_it, final_cost)

        return costs.stack()
