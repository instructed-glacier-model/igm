#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, Optional, Tuple

from .optimizer import Optimizer
from ..mappings import Mapping
from ..halt import Halt, HaltStatus
from .line_searches import LineSearches, ValueAndGradient


class OptimizerNewton(Optimizer):
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
        damping: float = 1e-4,
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
        self.name = "newton"
        self.line_search = LineSearches[line_search_method]()
        self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
        self.alpha_min = tf.Variable(alpha_min, dtype=self.precision)
        self.damping = damping

    def update_parameters(self, iter_max: int, damping: float) -> None:
        self.iter_max.assign(iter_max)
        self.damping = damping

    def _analyze_hessian(
        self, h_mat: tf.Tensor, iter: tf.Tensor, n_eigs: int = 10
    ) -> None:
        """Print smallest N eigenvalues of the Hessian."""
        eigenvalues = tf.linalg.eigvalsh(h_mat)
        smallest_eigs = eigenvalues[:n_eigs]

        tf.print(
            "Iter:",
            iter,
            "| Smallest",
            n_eigs,
            "eigenvalues:",
            smallest_eigs,
            summarize=-1,
        )

        # Also print condition number
        cond_number = eigenvalues[-1] / (eigenvalues[0] + 1e-12)
        tf.print("Condition number (λ_max/λ_min):", cond_number)

    def _plot_hessian_wrapper(
        self, h_mat_np: tf.Tensor, iter_np: tf.Tensor
    ) -> tf.Tensor:
        """Wrapper for plotting (called via py_function)."""
        import matplotlib.pyplot as plt
        import numpy as np

        h_mat = h_mat_np.numpy() if hasattr(h_mat_np, "numpy") else h_mat_np
        iter_val = int(iter_np.numpy() if hasattr(iter_np, "numpy") else iter_np)

        plt.figure(figsize=(10, 8))
        plt.imshow(h_mat, cmap="RdBu_r", aspect="auto")
        plt.colorbar(label="Hessian value")
        plt.title(f"Hessian Matrix at Iteration {iter_val}")
        plt.xlabel("Parameter index")
        plt.ylabel("Parameter index")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"hessian_iter_{iter_val:04d}.png", dpi=150)
        plt.close()

        return tf.constant(0)  # Return dummy value

    def _plot_hessian(self, h_mat: tf.Tensor, iter: tf.Tensor) -> None:
        """Plot Hessian matrix as an image."""
        tf.py_function(
            func=self._plot_hessian_wrapper, inp=[h_mat, iter], Tout=tf.int32
        )

    def _get_grads_and_hessian(self, inputs: tf.Tensor):
        """Compute cost, grad_theta, and Hessian."""

        theta = self.map.get_theta()

        with tf.GradientTape(watch_accessed_variables=False) as outer_tape:
            for theta_i in theta:
                outer_tape.watch(theta_i)
            with tf.GradientTape(watch_accessed_variables=False) as inner_tape:
                for theta_i in theta:
                    inner_tape.watch(theta_i)
                U, V = self.map.get_UV(inputs)
                cost = self.cost_fn(U, V, inputs)

            # First derivatives
            grads = inner_tape.gradient(cost, [U, V] + theta)
            grad_u = tuple(grads[:2])
            grad_theta = grads[2:]
            
            # ! Flattening here (on network mapping) causes an incredibly complex graph and memory leaks... (despite it being the cleanest...)
            grad_theta_flat = tf.concat(
                [tf.reshape(g, (-1,)) for g in grad_theta], axis=0
            )

        # Hessian blocks
        h_blocks_theta = outer_tape.jacobian(
            target=grad_theta_flat,  # jacobian needs a flat tensor and not a list (compared to gradient)
            sources=theta,
        )

        # Full Hessian
        n_unknowns = tf.add_n([tf.size(t) for t in theta])
        h_mat = tf.concat(
            [tf.reshape(h, (n_unknowns, -1)) for h in h_blocks_theta], axis=1
        )

        return cost, h_mat, grad_u, grad_theta

    def _force_descent(
        self, p_flat: tf.Tensor, grad_theta_flat: tf.Tensor, _: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        dot_gp = self._dot(grad_theta_flat, p_flat)
        return tf.cond(dot_gp >= 0.0, lambda: -grad_theta_flat, lambda: p_flat), None

    def _apply_step(
        self, theta_flat: tf.Tensor, alpha: tf.Tensor, p_flat: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        return theta_flat + alpha * p_flat, None

    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        dtype = self.precision
        return tf.tensordot(tf.cast(a, dtype), tf.cast(b, dtype), axes=1)

    def _line_search(
        self, theta_flat: tf.Tensor, p_flat: tf.Tensor, input: tf.Tensor
    ) -> tf.Tensor:
        def eval_fn(alpha: tf.Tensor) -> ValueAndGradient:
            theta_backup = self.map.copy_theta(self.map.get_theta())
            theta_alpha, _ = self._apply_step(theta_flat, alpha, p_flat)

            self.map.set_theta(self.map.unflatten_theta(theta_alpha))
            f, _, grad = self._get_grad(input)
            grad_flat = self.map.flatten_theta(grad)
            df = self._dot(grad_flat, p_flat)
            df = tf.cast(df, grad_flat.dtype)

            self.map.set_theta(theta_backup)
            return ValueAndGradient(x=alpha, f=f, df=df)

        return self.line_search.search(theta_flat, p_flat, eval_fn)

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        first_batch = self.sampler(inputs)
        n_batches = first_batch.shape[0]
        if n_batches != 1:
            raise NotImplementedError("❌ Hessian optimizer requires a single batch.")

        if getattr(self.sampler, "dynamic_augmentation", False):
            static_batches = None
            dynamic_augmentation = True
        else:
            static_batches = first_batch
            dynamic_augmentation = False

        input = first_batch[0, :, :, :, :]

        theta_flat = self.map.flatten_theta(self.map.get_theta())

        cost, _, _, _ = self._get_grads_and_hessian(input)
        U, V = self.map.get_UV(input)
        self._init_step_state(U, V, theta_flat)

        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = tf.constant(-1, dtype=tf.int32)
        costs = tf.TensorArray(dtype=cost.dtype, size=int(self.iter_max))

        for iter in tf.range(self.iter_max):

            if dynamic_augmentation:
                next_batch = self.sampler(inputs)
            else:
                next_batch = static_batches
            input = next_batch[0, :, :, :, :]

            cost, h_mat, grad_u, grad_theta = self._get_grads_and_hessian(input)

            grad_theta_flat = tf.concat(
                [tf.reshape(g, (-1,)) for g in grad_theta], axis=0
            )
            grad_theta_flat = tf.reshape(grad_theta_flat, (-1, 1))

            # Damping to avoid singularity and add convexity
            h_mat += tf.eye(tf.shape(h_mat)[0], dtype=h_mat.dtype) * tf.cast(
                self.damping, h_mat.dtype
            )

            # Solve for newton step
            p_flat = tf.linalg.solve(h_mat, -grad_theta_flat)

            p_flat = tf.squeeze(p_flat, axis=-1)
            grad_theta_flat = tf.squeeze(grad_theta_flat, axis=-1)

            p_flat, _ = self._force_descent(p_flat, grad_theta_flat, theta_flat)

            alpha = self._line_search(theta_flat, p_flat, input)
            alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))

            theta_flat, _ = self._apply_step(theta_flat, alpha, p_flat)
            self.map.set_theta(self.map.unflatten_theta(theta_flat))

            self.map.on_step_end(iter)
            costs = costs.write(iter, cost)

            U, V = self.map.get_UV(input)
            grad_u_norm, step_norm = self._get_grad_norm(grad_u, grad_theta)
            self._update_step_state(
                iter, U, V, theta_flat, cost, grad_u_norm, step_norm
            )

            halt_status = self._check_stopping()
            self._update_display()

            if self.debug_mode and iter % self.debug_freq == 0:
                self._update_debug_state(iter, cost, grad_u, grad_theta_flat)
                self._debug_display()

            iter_last = iter
            if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
                break

        self._finalize_display(halt_status)
        return costs.stack()[: iter_last + 1]
