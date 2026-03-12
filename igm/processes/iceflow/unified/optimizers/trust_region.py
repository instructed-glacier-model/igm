# # #!/usr/bin/env python3

# # # Copyright (C) 2021-2025 IGM authors
# # # Published under the GNU GPL (Version 3), check at the LICENSE file

# # import tensorflow as tf
# # from typing import Callable, Optional, Tuple

# # from .optimizer import Optimizer
# # from ..mappings import Mapping
# # from ..halt import Halt, HaltStatus


# # class OptimizerTrustRegion(Optimizer):
# #     """
# #     Matrix-free Trust Region optimizer.

# #     Uses truncated CG to solve the trust region subproblem:
# #         min_p  m(p) = f + g^T p + 0.5 p^T H p
# #         s.t.   ||p|| <= delta
# #     """

# #     def __init__(
# #         self,
# #         cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
# #         map: Mapping,
# #         halt: Optional[Halt] = None,
# #         print_cost: bool = True,
# #         print_cost_freq: int = 1,
# #         precision: str = "float32",
# #         ord_grad_u: str = "l2_weighted",
# #         ord_grad_theta: str = "l2_weighted",
# #         iter_max: int = 100,
# #         damping: float = 0.0,
# #         cg_max_iter: int = 100,
# #         cg_tol: float = 1e-12,
# #         delta_init: float = 1e5,
# #         delta_max: float = 1e10,
# #         eta: float = 0.15,
# #         diagnostics: bool = True,
# #         diagnostics_freq: int = 10,
# #         **kwargs,
# #     ):
# #         super().__init__(
# #             cost_fn,
# #             map,
# #             halt,
# #             print_cost,
# #             print_cost_freq,
# #             precision,
# #             ord_grad_u,
# #             ord_grad_theta,
# #             **kwargs,
# #         )

# #         self.name = "trust_region"

# #         self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
# #         self.damping = tf.constant(damping, dtype=self.precision)
# #         self.cg_max_iter = cg_max_iter
# #         self.cg_tol = tf.constant(cg_tol, dtype=self.precision)
        
# #         # Trust region parameters
# #         self.delta = tf.Variable(delta_init, dtype=self.precision)
# #         self.delta_max = tf.constant(delta_max, dtype=self.precision)
# #         self.eta = tf.constant(eta, dtype=self.precision)
        
# #         # Diagnostic settings
# #         self.diagnostics = diagnostics
# #         self.diagnostics_freq = diagnostics_freq
        
# #         # Preconditioner settings
# #         self.use_preconditioner = True
# #         self.precond_floor = tf.constant(1e-8, dtype=self.precision)
# #         self.precond_needs_update = tf.Variable(True, dtype=tf.bool)
# #         self.precond_diag = None

# #     def update_parameters(self, iter_max: int, damping: float) -> None:
# #         self.iter_max.assign(iter_max)
# #         self.damping = tf.cast(damping, self.precision)

# #     # ============================================================
# #     # Cost + gradient
# #     # ============================================================

# #     def _cost_and_grad(
# #         self, 
# #         inputs: tf.Tensor,
# #         cost_fn: Callable,
# #     ):
# #         with tf.GradientTape() as tape:
# #             U, V = self.map.get_UV(inputs)
# #             tape.watch((U, V))
# #             cost = cost_fn(U, V, inputs)

# #         grad_U, grad_V = tape.gradient(cost, (U, V))
# #         return cost, U, V, grad_U, grad_V

# #     # ============================================================
# #     # Hessian–vector product
# #     # ============================================================

# #     def _hvp(
# #         self,
# #         inputs: tf.Tensor,
# #         U: tf.Tensor,
# #         V: tf.Tensor,
# #         v_flat: tf.Tensor,
# #         cost_fn: Callable,
# #         damping: tf.Tensor,
# #     ) -> tf.Tensor:
# #         """
# #         Compute (H + damping I) v using nested GradientTapes.
# #         """
        
# #         nU = tf.size(U)
# #         vU = tf.reshape(v_flat[:nU], tf.shape(U))
# #         vV = tf.reshape(v_flat[nU:], tf.shape(V))

# #         with tf.GradientTape() as outer_tape:
# #             outer_tape.watch((U, V))
# #             with tf.GradientTape() as inner_tape:
# #                 inner_tape.watch((U, V))
# #                 c = cost_fn(U, V, inputs)
# #             gU, gV = inner_tape.gradient(c, (U, V))
            
# #             gv = tf.reduce_sum(gU * vU) + tf.reduce_sum(gV * vV)
        
# #         HvU, HvV = outer_tape.gradient(gv, (U, V))

# #         Hv_flat = tf.concat(
# #             [tf.reshape(HvU, [-1]), tf.reshape(HvV, [-1])],
# #             axis=0,
# #         )

# #         return Hv_flat + damping * v_flat

# #     # ============================================================
# #     # DIAGNOSTICS: Lanczos algorithm for eigenvalue estimates
# #     # ============================================================

# #     def _lanczos_eigenvalues(
# #         self,
# #         inputs: tf.Tensor,
# #         U: tf.Tensor,
# #         V: tf.Tensor,
# #         cost_fn: Callable,
# #         damping: tf.Tensor,
# #         n_iter: int = 20,
# #     ) -> Tuple[tf.Tensor, tf.Tensor]:
# #         """
# #         Use Lanczos iteration to estimate extreme eigenvalues of H.
# #         """
# #         cost, _, _, grad_U, grad_V = self._cost_and_grad(inputs, cost_fn)
# #         g_flat = tf.concat([tf.reshape(grad_U, [-1]), tf.reshape(grad_V, [-1])], axis=0)
        
# #         v = tf.random.normal(tf.shape(g_flat), dtype=self.precision)
# #         v = v / tf.norm(v)
        
# #         alpha = tf.TensorArray(dtype=self.precision, size=n_iter, dynamic_size=False)
# #         beta = tf.TensorArray(dtype=self.precision, size=n_iter, dynamic_size=False)
        
# #         v_old = tf.zeros_like(v)
# #         beta_old = tf.constant(0.0, dtype=self.precision)
        
# #         for j in tf.range(n_iter):
# #             w = self._hvp(inputs, U, V, v, cost_fn, damping)
            
# #             alpha_j = tf.reduce_sum(w * v)
# #             w = w - alpha_j * v - beta_old * v_old
            
# #             beta_j = tf.norm(w)
            
# #             alpha = alpha.write(j, alpha_j)
# #             beta = beta.write(j, beta_j)
            
# #             v_old = v
# #             v = w / (beta_j + 1e-16)
# #             beta_old = beta_j
        
# #         alpha_vals = alpha.stack()
# #         beta_vals = beta.stack()
        
# #         lambda_max = tf.reduce_max(alpha_vals + 2.0 * beta_vals)
# #         lambda_min = tf.reduce_min(alpha_vals - 2.0 * beta_vals)
        
# #         return lambda_min, lambda_max

# #     # ============================================================
# #     # DIAGNOSTICS: Rayleigh quotient on gradient direction
# #     # ============================================================

# #     def _gradient_curvature(
# #         self,
# #         inputs: tf.Tensor,
# #         U: tf.Tensor,
# #         V: tf.Tensor,
# #         g_flat: tf.Tensor,
# #         cost_fn: Callable,
# #         damping: tf.Tensor,
# #     ) -> tf.Tensor:
# #         """
# #         Compute g^T H g / g^T g (curvature along gradient direction).
# #         """
# #         Hg = self._hvp(inputs, U, V, g_flat, cost_fn, damping)
# #         gHg = tf.reduce_sum(g_flat * Hg)
# #         gg = tf.reduce_sum(g_flat * g_flat)
        
# #         return gHg / (gg + 1e-16)

# #     # ============================================================
# #     # Trust Region CG Subproblem Solver (Steihaug-Tong)
# #     # ============================================================

# #     def _cg_trust_region(
# #         self,
# #         inputs: tf.Tensor,
# #         U: tf.Tensor,
# #         V: tf.Tensor,
# #         g_flat: tf.Tensor,
# #         cost_fn: Callable,
# #         damping: tf.Tensor,
# #         delta: tf.Tensor,
# #         cg_max_iter: int,
# #         cg_tol: tf.Tensor,
# #         verbose: bool = False,
# #     ) -> tf.Tensor:
# #         """
# #         Solve trust region subproblem using preconditioned Steihaug-Tong CG.
# #         """
        
# #         # Initialize preconditioner on first call
# #         if self.precond_diag is None:
# #             n_total = tf.size(g_flat)
# #             self.precond_diag = tf.ones(n_total, dtype=self.precision)
        
# #         # Compute diagonal preconditioner if needed
# #         def compute_precond():
# #             n_samples = 10
# #             n_total = tf.size(g_flat)
# #             diag_est = tf.zeros(n_total, dtype=self.precision)
            
# #             for _ in tf.range(n_samples):
# #                 z = tf.random.uniform((n_total,), dtype=self.precision)
# #                 z = tf.where(z < 0.5, -tf.ones_like(z), tf.ones_like(z))
# #                 Hz = self._hvp(inputs, U, V, z, cost_fn, tf.constant(0.0, dtype=self.precision))
# #                 diag_est += z * Hz
            
# #             diag_est = diag_est / tf.cast(n_samples, self.precision)
# #             diag_abs = tf.maximum(tf.abs(diag_est), self.precond_floor)
# #             self.precond_needs_update.assign(False)
# #             return 1.0 / diag_abs

# #         def keep_precond():
# #             return self.precond_diag

# #         precond = tf.cond(
# #             self.precond_needs_update,
# #             compute_precond,
# #             keep_precond
# #         )
        
# #         # Update the stored preconditioner
# #         self.precond_diag = precond

# #         x = tf.zeros_like(g_flat)
# #         r = g_flat
# #         z = precond * r  # Apply preconditioner
# #         p = -z
# #         rz = tf.reduce_sum(r * z)
        
# #         initial_residual = tf.sqrt(rz)
# #         converged = tf.constant(False)
# #         hit_boundary = tf.constant(False)
# #         neg_curvature = tf.constant(False)

# #         for i in tf.range(cg_max_iter):
# #             should_continue = tf.logical_not(
# #                 tf.logical_or(converged, tf.logical_or(hit_boundary, neg_curvature))
# #             )
            
# #             if should_continue:
# #                 converged = rz <= cg_tol
                
# #                 if tf.logical_not(converged):
# #                     Ap = self._hvp(inputs, U, V, p, cost_fn, damping)
# #                     pAp = tf.reduce_sum(p * Ap)
                    
# #                     if verbose and i % 5 == 0:
# #                         rel_residual = tf.sqrt(rz) / (initial_residual + 1e-16)
# #                         x_norm = tf.norm(x)
# #                         tf.print("  PCG iter:", i, "| rel_res:", rel_residual, 
# #                                 "| ||x||:", x_norm, "| δ:", delta, "| p^T Ap:", pAp)
                    
# #                     if pAp <= 0.0:
# #                         if verbose:
# #                             tf.print("  NEGATIVE CURVATURE detected")
# #                         neg_curvature = tf.constant(True)
# #                         tau = self._find_boundary_step(x, p, delta)
# #                         x = x + tau * p
# #                     else:
# #                         alpha = rz / pAp
# #                         x_new = x + alpha * p
                        
# #                         if tf.norm(x_new) >= delta:
# #                             if verbose:
# #                                 tf.print("  TRUST REGION BOUNDARY hit")
# #                             hit_boundary = tf.constant(True)
# #                             tau = self._find_boundary_step(x, p, delta)
# #                             x = x + tau * p
# #                         else:
# #                             x = x_new
# #                             r = r + alpha * Ap
# #                             z = precond * r  # Apply preconditioner
# #                             rz_new = tf.reduce_sum(r * z)
# #                             beta = rz_new / rz
# #                             p = -z + beta * p
# #                             rz = rz_new

# #         if verbose:
# #             final_residual = tf.sqrt(rz) / (initial_residual + 1e-16)
# #             tf.print("  PCG finished | final rel_res =", final_residual, "| ||x|| =", tf.norm(x))

# #         return x

# #     def _find_boundary_step(
# #         self, 
# #         x: tf.Tensor, 
# #         p: tf.Tensor, 
# #         delta: tf.Tensor
# #     ) -> tf.Tensor:
# #         """
# #         Find tau >= 0 such that ||x + tau*p|| = delta.
        
# #         This solves: ||x + tau*p||^2 = delta^2
# #         Expanding: (x^T x) + 2*tau*(x^T p) + tau^2*(p^T p) = delta^2
# #         This is a quadratic in tau: a*tau^2 + b*tau + c = 0
# #         """
# #         a = tf.reduce_sum(p * p)
# #         b = 2.0 * tf.reduce_sum(x * p)
# #         c = tf.reduce_sum(x * x) - delta * delta
        
# #         # Solve quadratic, take positive root
# #         discriminant = b * b - 4.0 * a * c
# #         tau = (-b + tf.sqrt(tf.maximum(discriminant, 0.0))) / (2.0 * a)
        
# #         return tf.maximum(tau, 0.0)

# #     # ============================================================
# #     # Model reduction (predicted vs actual)
# #     # ============================================================

# #     def _compute_rho(
# #         self,
# #         f_old: tf.Tensor,
# #         f_new: tf.Tensor,
# #         g_flat: tf.Tensor,
# #         p_flat: tf.Tensor,
# #         inputs: tf.Tensor,
# #         U: tf.Tensor,
# #         V: tf.Tensor,
# #         cost_fn: Callable,
# #         damping: tf.Tensor,
# #     ) -> tf.Tensor:
# #         """
# #         Compute ratio of actual to predicted reduction:
# #             rho = (f_old - f_new) / (m_old - m_new)
        
# #         where m(p) = f + g^T p + 0.5 p^T H p is the quadratic model.
# #         """
# #         actual_reduction = f_old - f_new
        
# #         # Predicted reduction: -g^T p - 0.5 p^T H p
# #         Hp = self._hvp(inputs, U, V, p_flat, cost_fn, damping)
# #         predicted_reduction = -(tf.reduce_sum(g_flat * p_flat) + 
# #                                0.5 * tf.reduce_sum(p_flat * Hp))
        
# #         rho = actual_reduction / (predicted_reduction + 1e-16)
# #         return rho

# #     # ============================================================
# #     # Newton direction with trust region
# #     # ============================================================

# #     def _get_grad(
# #         self, 
# #         inputs: tf.Tensor,
# #         cost_fn: Callable,
# #         damping: tf.Tensor,
# #         delta: tf.Tensor,
# #         cg_max_iter: int,
# #         cg_tol: tf.Tensor,
# #         verbose: bool = False,
# #     ):
# #         """
# #         Compute cost, gradient, and trust region step.
# #         """

# #         cost, U, V, grad_U, grad_V = self._cost_and_grad(inputs, cost_fn)

# #         g_flat = tf.concat(
# #             [tf.reshape(grad_U, [-1]), tf.reshape(grad_V, [-1])],
# #             axis=0,
# #         )

# #         step_flat = self._cg_trust_region(
# #             inputs, U, V, g_flat, cost_fn, damping, delta, 
# #             cg_max_iter, cg_tol, verbose
# #         )

# #         nU = tf.size(U)
# #         step_U = tf.reshape(step_flat[:nU], tf.shape(U))
# #         step_V = tf.reshape(step_flat[nU:], tf.shape(V))

# #         return cost, U, V, g_flat, (grad_U, grad_V), (step_U, step_V)

# #     # ============================================================
# #     # Utilities
# #     # ============================================================

# #     def _apply_step(
# #         self, theta_flat: tf.Tensor, p_flat: tf.Tensor
# #     ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
# #         return theta_flat + p_flat, None

# #     @tf.function(reduce_retracing=True)
# #     def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
# #         dtype = self.precision
# #         return tf.tensordot(tf.cast(a, dtype), tf.cast(b, dtype), axes=1)

# #     # ============================================================
# #     # Main optimization loop
# #     # ============================================================

# #     @tf.function(jit_compile=False)
# #     def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
# #         first_batch = self.sampler(inputs)
# #         if first_batch.shape[0] != 1:
# #             raise NotImplementedError("Trust Region requires a single batch.")

# #         input = first_batch[0, :, :, :, :]

# #         # Extract values
# #         cost_fn = self.cost_fn
# #         damping = self.damping
# #         cg_max_iter = self.cg_max_iter
# #         cg_tol = self.cg_tol
# #         diagnostics = self.diagnostics
# #         diag_freq = self.diagnostics_freq

# #         theta_flat = self.map.flatten_theta(self.map.get_theta())

# #         cost, U, V, g_flat, grad_u, step_theta = self._get_grad(
# #             input, cost_fn, damping, self.delta, cg_max_iter, cg_tol, False
# #         )
# #         self._init_step_state(U, V, theta_flat)

# #         halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
# #         iter_last = tf.constant(-1, dtype=tf.int32)
# #         costs = tf.TensorArray(dtype=cost.dtype, size=int(self.iter_max))

# #         for iter in tf.range(self.iter_max):

# #             # Recompute preconditioner every 10 iterations
# #             if iter % 1 == 0:
# #                 self.precond_needs_update.assign(True)

# #             # Run diagnostics
# #             run_diagnostics = tf.logical_and(
# #                 diagnostics,
# #                 tf.equal(tf.math.floormod(iter, diag_freq), 0)
# #             )

# #             if run_diagnostics:
# #                 tf.print("\n========== DIAGNOSTICS AT ITERATION", iter, "==========")
                
# #                 lambda_min, lambda_max = self._lanczos_eigenvalues(
# #                     input, U, V, cost_fn, damping, n_iter=20
# #                 )
# #                 tf.print("Estimated eigenvalues: λ_min =", lambda_min, ", λ_max =", lambda_max)
                
# #                 cond_number = lambda_max / (tf.abs(lambda_min) + 1e-16)
# #                 tf.print("Estimated condition number: κ(H) =", cond_number)
                
# #                 if lambda_min > 0.0:
# #                     tf.print("Status: STRONGLY CONVEX (λ_min > 0)")
# #                 elif lambda_min > -1e-6:
# #                     tf.print("Status: NEARLY SINGULAR or FLAT (λ_min ≈ 0)")
# #                 else:
# #                     tf.print("Status: INDEFINITE (λ_min < 0) - possible saddle point")
                
# #                 grad_curv = self._gradient_curvature(input, U, V, g_flat, cost_fn, damping)
# #                 tf.print("Curvature along gradient: g^T H g / ||g||^2 =", grad_curv)
                
# #                 grad_norm = tf.norm(g_flat)
# #                 tf.print("Gradient norm: ||g|| =", grad_norm)
# #                 tf.print("Trust region radius: δ =", self.delta)
                
# #                 tf.print("=" * 60, "\n")

# #             # Solve trust region subproblem
# #             cost_old = cost
# #             cost, U, V, g_flat, grad_u, step_theta = self._get_grad(
# #                 input, cost_fn, damping, self.delta, cg_max_iter, cg_tol, run_diagnostics
# #             )

# #             p_flat = self.map.flatten_theta(step_theta)
            
# #             # Try the step
# #             theta_new, _ = self._apply_step(theta_flat, p_flat)
# #             self.map.set_theta(self.map.unflatten_theta(theta_new))
            
# #             # Evaluate cost at new point
# #             cost_new, _, _, _, _ = self._cost_and_grad(input, cost_fn)
            
# #             # Compute ratio of actual to predicted reduction
# #             rho = self._compute_rho(
# #                 cost_old, cost_new, g_flat, p_flat, 
# #                 input, U, V, cost_fn, damping
# #             )
            
# #             if run_diagnostics:
# #                 tf.print("Trust region ratio ρ =", rho)
            
# #             # Update trust region radius based on rho
# #             p_norm = tf.norm(p_flat)
            
# #             # Shrink if poor agreement
# #             new_delta = tf.cond(
# #                 rho < 0.25,
# #                 lambda: 0.25 * self.delta,
# #                 lambda: self.delta
# #             )
            
# #             # Expand if good agreement and hit boundary
# #             new_delta = tf.cond(
# #                 tf.logical_and(rho > 0.75, p_norm >= 0.9 * self.delta),
# #                 lambda: tf.minimum(2.0 * new_delta, self.delta_max),
# #                 lambda: new_delta
# #             )
            
# #             self.delta.assign(new_delta)
            
# #             if run_diagnostics:
# #                 if rho < 0.25:
# #                     tf.print("  Shrinking δ -> ", self.delta)
# #                 elif rho > 0.75 and p_norm >= 0.9 * self.delta:
# #                     tf.print("  Expanding δ -> ", self.delta)
            
# #             # Accept or reject step
# #             accept_step = rho > self.eta
            
# #             theta_flat = tf.cond(
# #                 accept_step,
# #                 lambda: theta_new,
# #                 lambda: theta_flat
# #             )
            
# #             cost = tf.cond(
# #                 accept_step,
# #                 lambda: cost_new,
# #                 lambda: cost_old
# #             )
            
# #             # Restore parameters if rejected
# #             if tf.logical_not(accept_step):
# #                 self.map.set_theta(self.map.unflatten_theta(theta_flat))
            
# #             if run_diagnostics:
# #                 if accept_step:
# #                     tf.print("  Step ACCEPTED")
# #                 else:
# #                     tf.print("  Step REJECTED")

# #             costs = costs.write(iter, cost)

# #             U, V = self.map.get_UV(input)
# #             grad_u_norm, step_norm = self._get_grad_norm(grad_u, step_theta)
# #             self._update_step_state(
# #                 iter, U, V, theta_flat, cost, grad_u_norm, step_norm
# #             )

# #             halt_status = self._check_stopping()
# #             self._update_display()

# #             iter_last = iter
            
# #             if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
# #                 break

# #         tf.print("Final values of U and V", U, V)
# #         self._finalize_display(halt_status)
# #         return costs.stack()[: iter_last + 1]

# #!/usr/bin/env python3

# # Copyright (C) 2021-2025 IGM authors
# # Published under the GNU GPL (Version 3), check at the LICENSE file

# import tensorflow as tf
# from typing import Callable, Optional, Tuple, List

# from .optimizer import Optimizer
# from ..mappings import Mapping
# from ..halt import Halt, HaltStatus

# from dataclasses import dataclass


# @dataclass
# class CGContext:
#     """Context for CG solver to avoid passing many arguments."""

#     inputs: tf.Tensor
#     cost_fn: Callable
#     damping: tf.Tensor


# class OptimizerTrustRegion(Optimizer):
#     """
#     Matrix-free Trust Region optimizer.

#     Uses truncated CG (Steihaug-Tong) to solve the trust region subproblem:
#         min_p  m(p) = f + g^T p + 0.5 p^T H p
#         s.t.   ||p|| <= delta
#     """

#     def __init__(
#         self,
#         cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
#         map: Mapping,
#         halt: Optional[Halt] = None,
#         print_cost: bool = True,
#         print_cost_freq: int = 1,
#         precision: str = "float32",
#         ord_grad_u: str = "l2_weighted",
#         ord_grad_theta: str = "l2_weighted",
#         iter_max: int = 100,
#         damping: float = 0.0,
#         cg_max_iter: int = 100,
#         cg_tol: float = 1e-12,
#         delta_init: float = 1e2,
#         delta_max: float = 1e10,
#         eta: float = 0.15,
#         **kwargs,
#     ):
#         super().__init__(
#             cost_fn,
#             map,
#             halt,
#             print_cost,
#             print_cost_freq,
#             precision,
#             ord_grad_u,
#             ord_grad_theta,
#             **kwargs,
#         )

#         self.name = "trust_region"

#         self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
#         self.damping = tf.constant(damping, dtype=self.precision)
#         self.cg_max_iter = cg_max_iter
#         self.cg_tol = tf.constant(cg_tol, dtype=self.precision)

#         self.delta = tf.Variable(delta_init, dtype=self.precision)
#         self.delta_max = tf.constant(delta_max, dtype=self.precision)
#         self.eta = tf.constant(eta, dtype=self.precision)

#     def update_parameters(self, iter_max: int, damping: float) -> None:
#         self.iter_max.assign(iter_max)
#         self.damping = tf.cast(damping, self.precision)

#     def _cost_and_grad(
#         self,
#         inputs: tf.Tensor,
#         cost_fn: Callable,
#     ):
#         """Compute cost and gradient w.r.t. theta."""
#         theta = self.map.get_theta()
#         with tf.GradientTape(persistent=True) as tape:
#             U, V = self.map.get_UV(inputs)
#             cost = cost_fn(U, V, inputs)

#         grad_u = tape.gradient(cost, (U, V))
#         grad_theta = tape.gradient(cost, theta)

#         return cost, grad_u, grad_theta

#     def _cost_only(
#         self,
#         inputs: tf.Tensor,
#         cost_fn: Callable,
#     ) -> tf.Tensor:
#         """Compute cost only (no gradients needed)."""
#         U, V = self.map.get_UV(inputs)
#         return cost_fn(U, V, inputs)

#     def _hvp(
#         self,
#         inputs: tf.Tensor,
#         v_flat: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#     ) -> tf.Tensor:
#         """Compute (H + damping I) v using reverse-over-reverse Hessian-vector product."""

#         theta = self.map.get_theta()

#         with tf.GradientTape() as outer_tape:
#             with tf.GradientTape() as inner_tape:
#                 U, V = self.map.get_UV(inputs)
#                 cost = cost_fn(U, V, inputs)

#             grad_theta = inner_tape.gradient(cost, theta)

#         v = self.map.unflatten_theta(v_flat)

#         Hv_theta = outer_tape.gradient(
#             grad_theta,
#             theta,
#             output_gradients=v,
#         )

#         Hv_theta_flat = tf.concat(
#             [tf.reshape(h, (-1,)) for h in Hv_theta],
#             axis=0,
#         )

#         return Hv_theta_flat + damping * v_flat

#     def _cg_trust_region(
#         self,
#         inputs: tf.Tensor,
#         g_flat: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#         delta: tf.Tensor,
#         cg_max_iter: int,
#         cg_tol: tf.Tensor,
#     ) -> tf.Tensor:
#         """Solve trust region subproblem using Steihaug-Tong truncated CG.

#         Solves: min_p  g^T p + 0.5 p^T H p   s.t.  ||p|| <= delta

#         On negative curvature or boundary hit, moves to the trust region boundary.
#         """

#         ctx = CGContext(inputs=inputs, cost_fn=cost_fn, damping=damping)

#         x = tf.zeros_like(g_flat)
#         r = -g_flat
#         p = r
#         rs = tf.tensordot(r, r, axes=1)

#         for i in tf.range(cg_max_iter):
#             if rs <= cg_tol:
#                 break

#             Ap = self._hvp(ctx.inputs, p, ctx.cost_fn, ctx.damping)
#             pAp = tf.tensordot(p, Ap, axes=1)

#             if pAp <= 0.0:
#                 if i == 0:
#                     tau = delta / (tf.norm(r) + 1e-16)
#                     x = tau * r
#                 else:
#                     tau = self._find_boundary_step(x, p, delta)
#                     x = x + tau * p
#                 break

#             alpha = rs / pAp
#             x_new = x + alpha * p

#             if tf.norm(x_new) >= delta:
#                 tau = self._find_boundary_step(x, p, delta)
#                 x = x + tau * p
#                 break

#             x = x_new
#             r_new = r - alpha * Ap
#             rs_new = tf.tensordot(r_new, r_new, axes=1)
#             beta = rs_new / rs
#             p = r_new + beta * p

#             r = r_new
#             rs = rs_new

#         return x

#     def _find_boundary_step(
#         self,
#         x: tf.Tensor,
#         p: tf.Tensor,
#         delta: tf.Tensor,
#     ) -> tf.Tensor:
#         """Find tau >= 0 such that ||x + tau*p|| = delta."""
#         a = tf.reduce_sum(p * p)
#         b = 2.0 * tf.reduce_sum(x * p)
#         c = tf.reduce_sum(x * x) - delta * delta

#         discriminant = b * b - 4.0 * a * c
#         tau = (-b + tf.sqrt(tf.maximum(discriminant, 0.0))) / (2.0 * a)

#         return tf.maximum(tau, 0.0)

#     def _predicted_reduction(
#         self,
#         g_flat: tf.Tensor,
#         p_flat: tf.Tensor,
#         inputs: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#     ) -> tf.Tensor:
#         """Compute predicted reduction: -g^T p - 0.5 p^T H p.

#         Must be called while theta is still at the old point.
#         """
#         Hp = self._hvp(inputs, p_flat, cost_fn, damping)
#         return -(tf.reduce_sum(g_flat * p_flat) + 0.5 * tf.reduce_sum(p_flat * Hp))

#     def _get_grad(
#         self,
#         inputs: tf.Tensor,
#         cost_fn: Callable,
#     ) -> Tuple[tf.Tensor, List[tf.Tensor | tf.Variable], List[tf.Tensor | tf.Variable]]:
#         """Compute cost and gradients (function and parameter space)."""

#         cost, grad_u, grad_theta = self._cost_and_grad(inputs, cost_fn)

#         return cost, grad_u, grad_theta

#     def _apply_step(
#         self, theta_flat: tf.Tensor, p_flat: tf.Tensor
#     ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
#         return theta_flat + p_flat, None

#     @tf.function(reduce_retracing=True)
#     def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
#         dtype = self.precision
#         return tf.tensordot(tf.cast(a, dtype), tf.cast(b, dtype), axes=1)

#     @tf.function(jit_compile=False)
#     def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
#         first_batch = self.sampler(inputs)
#         if first_batch.shape[0] != 1:
#             raise NotImplementedError("Trust Region requires a single batch.")

#         input = first_batch[0, :, :, :, :]

#         cost_fn = self.cost_fn
#         damping = self.damping
#         cg_max_iter = self.cg_max_iter
#         cg_tol = self.cg_tol

#         theta_flat = self.map.flatten_theta(self.map.get_theta())

#         cost, grad_u, grad_theta = self._get_grad(input, cost_fn)

#         U, V = self.map.get_UV(input)
#         self._init_step_state(U, V, theta_flat)

#         halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
#         iter_last = tf.constant(-1, dtype=tf.int32)
#         costs = tf.TensorArray(dtype=cost.dtype, size=int(self.iter_max))

#         for iter in tf.range(self.iter_max):

#             cost, grad_u, grad_theta = self._get_grad(input, cost_fn)
#             grad_theta_flat = self.map.flatten_theta(grad_theta)

#             # Backup current theta
#             theta_backup = self.map.copy_theta(self.map.get_theta())

#             # Solve trust region subproblem (HVP at current theta)
#             p_flat = self._cg_trust_region(
#                 inputs=input,
#                 g_flat=grad_theta_flat,
#                 cost_fn=cost_fn,
#                 damping=damping,
#                 delta=self.delta,
#                 cg_max_iter=cg_max_iter,
#                 cg_tol=cg_tol,
#             )

#             # Compute predicted reduction BEFORE changing theta
#             pred_red = self._predicted_reduction(
#                 grad_theta_flat, p_flat, input, cost_fn, damping
#             )

#             # Trial step: move to new point and evaluate cost
#             theta_new, _ = self._apply_step(theta_flat, p_flat)
#             self.map.set_theta(self.map.unflatten_theta(theta_new))
#             cost_new = self._cost_only(input, cost_fn)

#             # Compute rho = actual_reduction / predicted_reduction
#             actual_red = cost - cost_new
#             rho = actual_red / (pred_red + 1e-16)

#             # Handle NaN/Inf: treat as a very bad step
#             rho = tf.where(tf.math.is_finite(rho), rho, tf.constant(-1.0, dtype=self.precision))

#             # Update trust region radius
#             p_norm = tf.norm(p_flat)

#             new_delta = tf.cond(
#                 rho < 0.25,
#                 lambda: 0.25 * self.delta,
#                 lambda: self.delta,
#             )

#             new_delta = tf.cond(
#                 tf.logical_and(rho > 0.75, p_norm >= 0.9 * self.delta),
#                 lambda: tf.minimum(2.0 * new_delta, self.delta_max),
#                 lambda: new_delta,
#             )

#             self.delta.assign(new_delta)

#             # Accept or reject step
#             accept_step = rho > self.eta

#             if accept_step:
#                 theta_flat = theta_new
#                 cost = cost_new
#             else:
#                 self.map.set_theta(theta_backup)

#             costs = costs.write(iter, cost)

#             U, V = self.map.get_UV(input)
#             grad_u_norm, step_norm = self._get_grad_norm(grad_u, grad_theta)
#             self._update_step_state(
#                 iter, U, V, theta_flat, cost, grad_u_norm, step_norm
#             )

#             halt_status = self._check_stopping()
#             self._update_display()

#             iter_last = iter
#             if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
#                 break

#         self._finalize_display(halt_status)
#         return costs.stack()[: iter_last + 1]

# #!/usr/bin/env python3

# # Copyright (C) 2021-2025 IGM authors
# # Published under the GNU GPL (Version 3), check at the LICENSE file

# import tensorflow as tf
# from typing import Callable, Optional, Tuple

# from .optimizer import Optimizer
# from ..mappings import Mapping
# from ..halt import Halt, HaltStatus


# class OptimizerTrustRegion(Optimizer):
#     """
#     Matrix-free Trust Region optimizer.

#     Uses truncated CG to solve the trust region subproblem:
#         min_p  m(p) = f + g^T p + 0.5 p^T H p
#         s.t.   ||p|| <= delta
#     """

#     def __init__(
#         self,
#         cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
#         map: Mapping,
#         halt: Optional[Halt] = None,
#         print_cost: bool = True,
#         print_cost_freq: int = 1,
#         precision: str = "float32",
#         ord_grad_u: str = "l2_weighted",
#         ord_grad_theta: str = "l2_weighted",
#         iter_max: int = 100,
#         damping: float = 0.0,
#         cg_max_iter: int = 100,
#         cg_tol: float = 1e-12,
#         delta_init: float = 1e5,
#         delta_max: float = 1e10,
#         eta: float = 0.15,
#         diagnostics: bool = True,
#         diagnostics_freq: int = 10,
#         **kwargs,
#     ):
#         super().__init__(
#             cost_fn,
#             map,
#             halt,
#             print_cost,
#             print_cost_freq,
#             precision,
#             ord_grad_u,
#             ord_grad_theta,
#             **kwargs,
#         )

#         self.name = "trust_region"

#         self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
#         self.damping = tf.constant(damping, dtype=self.precision)
#         self.cg_max_iter = cg_max_iter
#         self.cg_tol = tf.constant(cg_tol, dtype=self.precision)
        
#         # Trust region parameters
#         self.delta = tf.Variable(delta_init, dtype=self.precision)
#         self.delta_max = tf.constant(delta_max, dtype=self.precision)
#         self.eta = tf.constant(eta, dtype=self.precision)
        
#         # Diagnostic settings
#         self.diagnostics = diagnostics
#         self.diagnostics_freq = diagnostics_freq
        
#         # Preconditioner settings
#         self.use_preconditioner = True
#         self.precond_floor = tf.constant(1e-8, dtype=self.precision)
#         self.precond_needs_update = tf.Variable(True, dtype=tf.bool)
#         self.precond_diag = None

#     def update_parameters(self, iter_max: int, damping: float) -> None:
#         self.iter_max.assign(iter_max)
#         self.damping = tf.cast(damping, self.precision)

#     # ============================================================
#     # Cost + gradient
#     # ============================================================

#     def _cost_and_grad(
#         self, 
#         inputs: tf.Tensor,
#         cost_fn: Callable,
#     ):
#         with tf.GradientTape() as tape:
#             U, V = self.map.get_UV(inputs)
#             tape.watch((U, V))
#             cost = cost_fn(U, V, inputs)

#         grad_U, grad_V = tape.gradient(cost, (U, V))
#         return cost, U, V, grad_U, grad_V

#     # ============================================================
#     # Hessian–vector product
#     # ============================================================

#     def _hvp(
#         self,
#         inputs: tf.Tensor,
#         U: tf.Tensor,
#         V: tf.Tensor,
#         v_flat: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#     ) -> tf.Tensor:
#         """
#         Compute (H + damping I) v using nested GradientTapes.
#         """
        
#         nU = tf.size(U)
#         vU = tf.reshape(v_flat[:nU], tf.shape(U))
#         vV = tf.reshape(v_flat[nU:], tf.shape(V))

#         with tf.GradientTape() as outer_tape:
#             outer_tape.watch((U, V))
#             with tf.GradientTape() as inner_tape:
#                 inner_tape.watch((U, V))
#                 c = cost_fn(U, V, inputs)
#             gU, gV = inner_tape.gradient(c, (U, V))
            
#             gv = tf.reduce_sum(gU * vU) + tf.reduce_sum(gV * vV)
        
#         HvU, HvV = outer_tape.gradient(gv, (U, V))

#         Hv_flat = tf.concat(
#             [tf.reshape(HvU, [-1]), tf.reshape(HvV, [-1])],
#             axis=0,
#         )

#         return Hv_flat + damping * v_flat

#     # ============================================================
#     # DIAGNOSTICS: Lanczos algorithm for eigenvalue estimates
#     # ============================================================

#     def _lanczos_eigenvalues(
#         self,
#         inputs: tf.Tensor,
#         U: tf.Tensor,
#         V: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#         n_iter: int = 20,
#     ) -> Tuple[tf.Tensor, tf.Tensor]:
#         """
#         Use Lanczos iteration to estimate extreme eigenvalues of H.
#         """
#         cost, _, _, grad_U, grad_V = self._cost_and_grad(inputs, cost_fn)
#         g_flat = tf.concat([tf.reshape(grad_U, [-1]), tf.reshape(grad_V, [-1])], axis=0)
        
#         v = tf.random.normal(tf.shape(g_flat), dtype=self.precision)
#         v = v / tf.norm(v)
        
#         alpha = tf.TensorArray(dtype=self.precision, size=n_iter, dynamic_size=False)
#         beta = tf.TensorArray(dtype=self.precision, size=n_iter, dynamic_size=False)
        
#         v_old = tf.zeros_like(v)
#         beta_old = tf.constant(0.0, dtype=self.precision)
        
#         for j in tf.range(n_iter):
#             w = self._hvp(inputs, U, V, v, cost_fn, damping)
            
#             alpha_j = tf.reduce_sum(w * v)
#             w = w - alpha_j * v - beta_old * v_old
            
#             beta_j = tf.norm(w)
            
#             alpha = alpha.write(j, alpha_j)
#             beta = beta.write(j, beta_j)
            
#             v_old = v
#             v = w / (beta_j + 1e-16)
#             beta_old = beta_j
        
#         alpha_vals = alpha.stack()
#         beta_vals = beta.stack()
        
#         lambda_max = tf.reduce_max(alpha_vals + 2.0 * beta_vals)
#         lambda_min = tf.reduce_min(alpha_vals - 2.0 * beta_vals)
        
#         return lambda_min, lambda_max

#     # ============================================================
#     # DIAGNOSTICS: Rayleigh quotient on gradient direction
#     # ============================================================

#     def _gradient_curvature(
#         self,
#         inputs: tf.Tensor,
#         U: tf.Tensor,
#         V: tf.Tensor,
#         g_flat: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#     ) -> tf.Tensor:
#         """
#         Compute g^T H g / g^T g (curvature along gradient direction).
#         """
#         Hg = self._hvp(inputs, U, V, g_flat, cost_fn, damping)
#         gHg = tf.reduce_sum(g_flat * Hg)
#         gg = tf.reduce_sum(g_flat * g_flat)
        
#         return gHg / (gg + 1e-16)

#     # ============================================================
#     # Trust Region CG Subproblem Solver (Steihaug-Tong)
#     # ============================================================

#     def _cg_trust_region(
#         self,
#         inputs: tf.Tensor,
#         U: tf.Tensor,
#         V: tf.Tensor,
#         g_flat: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#         delta: tf.Tensor,
#         cg_max_iter: int,
#         cg_tol: tf.Tensor,
#         verbose: bool = False,
#     ) -> tf.Tensor:
#         """
#         Solve trust region subproblem using preconditioned Steihaug-Tong CG.
#         """
        
#         # Initialize preconditioner on first call
#         if self.precond_diag is None:
#             n_total = tf.size(g_flat)
#             self.precond_diag = tf.ones(n_total, dtype=self.precision)
        
#         # Compute diagonal preconditioner if needed
#         def compute_precond():
#             n_samples = 10
#             n_total = tf.size(g_flat)
#             diag_est = tf.zeros(n_total, dtype=self.precision)
            
#             for _ in tf.range(n_samples):
#                 z = tf.random.uniform((n_total,), dtype=self.precision)
#                 z = tf.where(z < 0.5, -tf.ones_like(z), tf.ones_like(z))
#                 Hz = self._hvp(inputs, U, V, z, cost_fn, tf.constant(0.0, dtype=self.precision))
#                 diag_est += z * Hz
            
#             diag_est = diag_est / tf.cast(n_samples, self.precision)
#             diag_abs = tf.maximum(tf.abs(diag_est), self.precond_floor)
#             self.precond_needs_update.assign(False)
#             return 1.0 / diag_abs

#         def keep_precond():
#             return self.precond_diag

#         precond = tf.cond(
#             self.precond_needs_update,
#             compute_precond,
#             keep_precond
#         )
        
#         # Update the stored preconditioner
#         self.precond_diag = precond

#         x = tf.zeros_like(g_flat)
#         r = g_flat
#         z = precond * r  # Apply preconditioner
#         p = -z
#         rz = tf.reduce_sum(r * z)
        
#         initial_residual = tf.sqrt(rz)
#         converged = tf.constant(False)
#         hit_boundary = tf.constant(False)
#         neg_curvature = tf.constant(False)

#         for i in tf.range(cg_max_iter):
#             should_continue = tf.logical_not(
#                 tf.logical_or(converged, tf.logical_or(hit_boundary, neg_curvature))
#             )
            
#             if should_continue:
#                 converged = rz <= cg_tol
                
#                 if tf.logical_not(converged):
#                     Ap = self._hvp(inputs, U, V, p, cost_fn, damping)
#                     pAp = tf.reduce_sum(p * Ap)
                    
#                     if verbose and i % 5 == 0:
#                         rel_residual = tf.sqrt(rz) / (initial_residual + 1e-16)
#                         x_norm = tf.norm(x)
#                         tf.print("  PCG iter:", i, "| rel_res:", rel_residual, 
#                                 "| ||x||:", x_norm, "| δ:", delta, "| p^T Ap:", pAp)
                    
#                     if pAp <= 0.0:
#                         if verbose:
#                             tf.print("  NEGATIVE CURVATURE detected")
#                         neg_curvature = tf.constant(True)
#                         tau = self._find_boundary_step(x, p, delta)
#                         x = x + tau * p
#                     else:
#                         alpha = rz / pAp
#                         x_new = x + alpha * p
                        
#                         if tf.norm(x_new) >= delta:
#                             if verbose:
#                                 tf.print("  TRUST REGION BOUNDARY hit")
#                             hit_boundary = tf.constant(True)
#                             tau = self._find_boundary_step(x, p, delta)
#                             x = x + tau * p
#                         else:
#                             x = x_new
#                             r = r + alpha * Ap
#                             z = precond * r  # Apply preconditioner
#                             rz_new = tf.reduce_sum(r * z)
#                             beta = rz_new / rz
#                             p = -z + beta * p
#                             rz = rz_new

#         if verbose:
#             final_residual = tf.sqrt(rz) / (initial_residual + 1e-16)
#             tf.print("  PCG finished | final rel_res =", final_residual, "| ||x|| =", tf.norm(x))

#         return x

#     def _find_boundary_step(
#         self, 
#         x: tf.Tensor, 
#         p: tf.Tensor, 
#         delta: tf.Tensor
#     ) -> tf.Tensor:
#         """
#         Find tau >= 0 such that ||x + tau*p|| = delta.
        
#         This solves: ||x + tau*p||^2 = delta^2
#         Expanding: (x^T x) + 2*tau*(x^T p) + tau^2*(p^T p) = delta^2
#         This is a quadratic in tau: a*tau^2 + b*tau + c = 0
#         """
#         a = tf.reduce_sum(p * p)
#         b = 2.0 * tf.reduce_sum(x * p)
#         c = tf.reduce_sum(x * x) - delta * delta
        
#         # Solve quadratic, take positive root
#         discriminant = b * b - 4.0 * a * c
#         tau = (-b + tf.sqrt(tf.maximum(discriminant, 0.0))) / (2.0 * a)
        
#         return tf.maximum(tau, 0.0)

#     # ============================================================
#     # Model reduction (predicted vs actual)
#     # ============================================================

#     def _compute_rho(
#         self,
#         f_old: tf.Tensor,
#         f_new: tf.Tensor,
#         g_flat: tf.Tensor,
#         p_flat: tf.Tensor,
#         inputs: tf.Tensor,
#         U: tf.Tensor,
#         V: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#     ) -> tf.Tensor:
#         """
#         Compute ratio of actual to predicted reduction:
#             rho = (f_old - f_new) / (m_old - m_new)
        
#         where m(p) = f + g^T p + 0.5 p^T H p is the quadratic model.
#         """
#         actual_reduction = f_old - f_new
        
#         # Predicted reduction: -g^T p - 0.5 p^T H p
#         Hp = self._hvp(inputs, U, V, p_flat, cost_fn, damping)
#         predicted_reduction = -(tf.reduce_sum(g_flat * p_flat) + 
#                                0.5 * tf.reduce_sum(p_flat * Hp))
        
#         rho = actual_reduction / (predicted_reduction + 1e-16)
#         return rho

#     # ============================================================
#     # Newton direction with trust region
#     # ============================================================

#     def _get_grad(
#         self, 
#         inputs: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#         delta: tf.Tensor,
#         cg_max_iter: int,
#         cg_tol: tf.Tensor,
#         verbose: bool = False,
#     ):
#         """
#         Compute cost, gradient, and trust region step.
#         """

#         cost, U, V, grad_U, grad_V = self._cost_and_grad(inputs, cost_fn)

#         g_flat = tf.concat(
#             [tf.reshape(grad_U, [-1]), tf.reshape(grad_V, [-1])],
#             axis=0,
#         )

#         step_flat = self._cg_trust_region(
#             inputs, U, V, g_flat, cost_fn, damping, delta, 
#             cg_max_iter, cg_tol, verbose
#         )

#         nU = tf.size(U)
#         step_U = tf.reshape(step_flat[:nU], tf.shape(U))
#         step_V = tf.reshape(step_flat[nU:], tf.shape(V))

#         return cost, U, V, g_flat, (grad_U, grad_V), (step_U, step_V)

#     # ============================================================
#     # Utilities
#     # ============================================================

#     def _apply_step(
#         self, theta_flat: tf.Tensor, p_flat: tf.Tensor
#     ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
#         return theta_flat + p_flat, None

#     @tf.function(reduce_retracing=True)
#     def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
#         dtype = self.precision
#         return tf.tensordot(tf.cast(a, dtype), tf.cast(b, dtype), axes=1)

#     # ============================================================
#     # Main optimization loop
#     # ============================================================

#     @tf.function(jit_compile=False)
#     def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
#         first_batch = self.sampler(inputs)
#         if first_batch.shape[0] != 1:
#             raise NotImplementedError("Trust Region requires a single batch.")

#         input = first_batch[0, :, :, :, :]

#         # Extract values
#         cost_fn = self.cost_fn
#         damping = self.damping
#         cg_max_iter = self.cg_max_iter
#         cg_tol = self.cg_tol
#         diagnostics = self.diagnostics
#         diag_freq = self.diagnostics_freq

#         theta_flat = self.map.flatten_theta(self.map.get_theta())

#         cost, U, V, g_flat, grad_u, step_theta = self._get_grad(
#             input, cost_fn, damping, self.delta, cg_max_iter, cg_tol, False
#         )
#         self._init_step_state(U, V, theta_flat)

#         halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
#         iter_last = tf.constant(-1, dtype=tf.int32)
#         costs = tf.TensorArray(dtype=cost.dtype, size=int(self.iter_max))

#         for iter in tf.range(self.iter_max):

#             # Recompute preconditioner every 10 iterations
#             if iter % 1 == 0:
#                 self.precond_needs_update.assign(True)

#             # Run diagnostics
#             run_diagnostics = tf.logical_and(
#                 diagnostics,
#                 tf.equal(tf.math.floormod(iter, diag_freq), 0)
#             )

#             if run_diagnostics:
#                 tf.print("\n========== DIAGNOSTICS AT ITERATION", iter, "==========")
                
#                 lambda_min, lambda_max = self._lanczos_eigenvalues(
#                     input, U, V, cost_fn, damping, n_iter=20
#                 )
#                 tf.print("Estimated eigenvalues: λ_min =", lambda_min, ", λ_max =", lambda_max)
                
#                 cond_number = lambda_max / (tf.abs(lambda_min) + 1e-16)
#                 tf.print("Estimated condition number: κ(H) =", cond_number)
                
#                 if lambda_min > 0.0:
#                     tf.print("Status: STRONGLY CONVEX (λ_min > 0)")
#                 elif lambda_min > -1e-6:
#                     tf.print("Status: NEARLY SINGULAR or FLAT (λ_min ≈ 0)")
#                 else:
#                     tf.print("Status: INDEFINITE (λ_min < 0) - possible saddle point")
                
#                 grad_curv = self._gradient_curvature(input, U, V, g_flat, cost_fn, damping)
#                 tf.print("Curvature along gradient: g^T H g / ||g||^2 =", grad_curv)
                
#                 grad_norm = tf.norm(g_flat)
#                 tf.print("Gradient norm: ||g|| =", grad_norm)
#                 tf.print("Trust region radius: δ =", self.delta)
                
#                 tf.print("=" * 60, "\n")

#             # Solve trust region subproblem
#             cost_old = cost
#             cost, U, V, g_flat, grad_u, step_theta = self._get_grad(
#                 input, cost_fn, damping, self.delta, cg_max_iter, cg_tol, run_diagnostics
#             )

#             p_flat = self.map.flatten_theta(step_theta)
            
#             # Try the step
#             theta_new, _ = self._apply_step(theta_flat, p_flat)
#             self.map.set_theta(self.map.unflatten_theta(theta_new))
            
#             # Evaluate cost at new point
#             cost_new, _, _, _, _ = self._cost_and_grad(input, cost_fn)
            
#             # Compute ratio of actual to predicted reduction
#             rho = self._compute_rho(
#                 cost_old, cost_new, g_flat, p_flat, 
#                 input, U, V, cost_fn, damping
#             )
            
#             if run_diagnostics:
#                 tf.print("Trust region ratio ρ =", rho)
            
#             # Update trust region radius based on rho
#             p_norm = tf.norm(p_flat)
            
#             # Shrink if poor agreement
#             new_delta = tf.cond(
#                 rho < 0.25,
#                 lambda: 0.25 * self.delta,
#                 lambda: self.delta
#             )
            
#             # Expand if good agreement and hit boundary
#             new_delta = tf.cond(
#                 tf.logical_and(rho > 0.75, p_norm >= 0.9 * self.delta),
#                 lambda: tf.minimum(2.0 * new_delta, self.delta_max),
#                 lambda: new_delta
#             )
            
#             self.delta.assign(new_delta)
            
#             if run_diagnostics:
#                 if rho < 0.25:
#                     tf.print("  Shrinking δ -> ", self.delta)
#                 elif rho > 0.75 and p_norm >= 0.9 * self.delta:
#                     tf.print("  Expanding δ -> ", self.delta)
            
#             # Accept or reject step
#             accept_step = rho > self.eta
            
#             theta_flat = tf.cond(
#                 accept_step,
#                 lambda: theta_new,
#                 lambda: theta_flat
#             )
            
#             cost = tf.cond(
#                 accept_step,
#                 lambda: cost_new,
#                 lambda: cost_old
#             )
            
#             # Restore parameters if rejected
#             if tf.logical_not(accept_step):
#                 self.map.set_theta(self.map.unflatten_theta(theta_flat))
            
#             if run_diagnostics:
#                 if accept_step:
#                     tf.print("  Step ACCEPTED")
#                 else:
#                     tf.print("  Step REJECTED")

#             costs = costs.write(iter, cost)

#             U, V = self.map.get_UV(input)
#             grad_u_norm, step_norm = self._get_grad_norm(grad_u, step_theta)
#             self._update_step_state(
#                 iter, U, V, theta_flat, cost, grad_u_norm, step_norm
#             )

#             halt_status = self._check_stopping()
#             self._update_display()

#             iter_last = iter
            
#             if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
#                 break

#         tf.print("Final values of U and V", U, V)
#         self._finalize_display(halt_status)
#         return costs.stack()[: iter_last + 1]

#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, Optional, Tuple, List

from .optimizer import Optimizer
from ..mappings import Mapping
from ..halt import Halt, HaltStatus

from dataclasses import dataclass


@dataclass
class CGContext:
    """Context for CG solver to avoid passing many arguments."""

    inputs: tf.Tensor
    cost_fn: Callable
    damping: tf.Tensor


class OptimizerTrustRegion(Optimizer):
    """
    Matrix-free Trust Region optimizer.

    Uses truncated CG (Steihaug-Tong) to solve the trust region subproblem:
        min_p  m(p) = f + g^T p + 0.5 p^T H p
        s.t.   ||p|| <= delta
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
        iter_max: int = 100,
        damping: float = 0.0,
        cg_max_iter: int = 100,
        cg_tol: float = 1e-12,
        delta_init: float = 1e2,
        delta_max: float = 1e10,
        eta: float = 0.15,
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

        self.name = "trust_region"

        self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
        self.damping = tf.constant(damping, dtype=self.precision)
        self.cg_max_iter = cg_max_iter
        self.cg_tol = tf.constant(cg_tol, dtype=self.precision)

        self.delta = tf.Variable(delta_init, dtype=self.precision)
        self.delta_max = tf.constant(delta_max, dtype=self.precision)
        self.eta = tf.constant(eta, dtype=self.precision)

    def update_parameters(self, iter_max: int, damping: float) -> None:
        self.iter_max.assign(iter_max)
        self.damping = tf.cast(damping, self.precision)

    def _cost_and_grad(
        self,
        inputs: tf.Tensor,
        cost_fn: Callable,
    ):
        """Compute cost and gradient w.r.t. theta."""
        theta = self.map.get_theta()
        with tf.GradientTape(persistent=True) as tape:
            U, V = self.map.get_UV(inputs)
            cost = cost_fn(U, V, inputs)

        grad_u = tape.gradient(cost, (U, V))
        grad_theta = tape.gradient(cost, theta)

        return cost, grad_u, grad_theta

    def _cost_only(
        self,
        inputs: tf.Tensor,
        cost_fn: Callable,
    ) -> tf.Tensor:
        """Compute cost only (no gradients needed)."""
        U, V = self.map.get_UV(inputs)
        return cost_fn(U, V, inputs)

    def _hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        cost_fn: Callable,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        """Compute (H + damping I) v using reverse-over-reverse Hessian-vector product."""

        print("TRACING", "_hvp")
        
        theta = self.map.get_theta()

        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                U, V = self.map.get_UV(inputs)
                cost = cost_fn(U, V, inputs)

            grad_theta = inner_tape.gradient(cost, theta)

        v = self.map.unflatten_theta(v_flat)

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

    def _cg_trust_region(
        self,
        inputs: tf.Tensor,
        g_flat: tf.Tensor,
        cost_fn: Callable,
        damping: tf.Tensor,
        delta: tf.Tensor,
        cg_max_iter: int,
        cg_tol: tf.Tensor,
    ) -> tf.Tensor:
        """Solve trust region subproblem using Steihaug-Tong truncated CG.

        Solves: min_p  g^T p + 0.5 p^T H p   s.t.  ||p|| <= delta

        On negative curvature or boundary hit, moves to the trust region boundary.
        """

        print("TRACING", "_cg_trust_region")
        ctx = CGContext(inputs=inputs, cost_fn=cost_fn, damping=damping)

        x = tf.zeros_like(g_flat)
        r = -g_flat
        p = r
        rs = tf.tensordot(r, r, axes=1)

        for i in tf.range(cg_max_iter):
            if rs <= cg_tol:
                break

            Ap = self._hvp(ctx.inputs, p, ctx.cost_fn, ctx.damping)
            pAp = tf.tensordot(p, Ap, axes=1)

            if pAp <= 0.0:
                if i == 0:
                    tau = delta / (tf.norm(r) + 1e-16)
                    x = tau * r
                else:
                    tau = self._find_boundary_step(x, p, delta)
                    x = x + tau * p
                break

            alpha = rs / pAp
            x_new = x + alpha * p

            if tf.norm(x_new) >= delta:
                tau = self._find_boundary_step(x, p, delta)
                x = x + tau * p
                break

            x = x_new
            r_new = r - alpha * Ap
            rs_new = tf.tensordot(r_new, r_new, axes=1)
            beta = rs_new / rs
            p = r_new + beta * p

            r = r_new
            rs = rs_new

        return x

    def _find_boundary_step(
        self,
        x: tf.Tensor,
        p: tf.Tensor,
        delta: tf.Tensor,
    ) -> tf.Tensor:
        """Find tau >= 0 such that ||x + tau*p|| = delta."""
        a = tf.reduce_sum(p * p)
        b = 2.0 * tf.reduce_sum(x * p)
        c = tf.reduce_sum(x * x) - delta * delta

        discriminant = b * b - 4.0 * a * c
        tau = (-b + tf.sqrt(tf.maximum(discriminant, 0.0))) / (2.0 * a)

        return tf.maximum(tau, 0.0)

    def _predicted_reduction(
        self,
        g_flat: tf.Tensor,
        p_flat: tf.Tensor,
        inputs: tf.Tensor,
        cost_fn: Callable,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        """Compute predicted reduction: -g^T p - 0.5 p^T H p.

        Must be called while theta is still at the old point.
        """
        Hp = self._hvp(inputs, p_flat, cost_fn, damping)
        return -(tf.reduce_sum(g_flat * p_flat) + 0.5 * tf.reduce_sum(p_flat * Hp))

    def _get_grad(
        self,
        inputs: tf.Tensor,
        cost_fn: Callable,
    ) -> Tuple[tf.Tensor, List[tf.Tensor | tf.Variable], List[tf.Tensor | tf.Variable]]:
        """Compute cost and gradients (function and parameter space)."""

        print("TRACING", "_get_grad")
        cost, grad_u, grad_theta = self._cost_and_grad(inputs, cost_fn)

        return cost, grad_u, grad_theta

    def _apply_step(
        self, theta_flat: tf.Tensor, p_flat: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        return theta_flat + p_flat, None

    @tf.function(reduce_retracing=True)
    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        dtype = self.precision
        return tf.tensordot(tf.cast(a, dtype), tf.cast(b, dtype), axes=1)

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        first_batch = self.sampler(inputs)
        if first_batch.shape[0] != 1:
            raise NotImplementedError("Trust Region requires a single batch.")

        input = first_batch[0, :, :, :, :]

        cost_fn = self.cost_fn
        damping = self.damping
        cg_max_iter = self.cg_max_iter
        cg_tol = self.cg_tol

        theta_flat = self.map.flatten_theta(self.map.get_theta())

        cost, grad_u, grad_theta = self._get_grad(input, cost_fn)

        U, V = self.map.get_UV(input)
        self._init_step_state(U, V, theta_flat)

        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = tf.constant(-1, dtype=tf.int32)
        costs = tf.TensorArray(dtype=cost.dtype, size=int(self.iter_max))

        for iter in tf.range(self.iter_max):

            cost, grad_u, grad_theta = self._get_grad(input, cost_fn)
            grad_theta_flat = self.map.flatten_theta(grad_theta)

            # Backup current theta
            theta_backup = self.map.copy_theta(self.map.get_theta())

            # Solve trust region subproblem (HVP at current theta)
            p_flat = self._cg_trust_region(
                inputs=input,
                g_flat=grad_theta_flat,
                cost_fn=cost_fn,
                damping=damping,
                delta=self.delta,
                cg_max_iter=cg_max_iter,
                cg_tol=cg_tol,
            )

            # Compute predicted reduction BEFORE changing theta
            pred_red = self._predicted_reduction(
                grad_theta_flat, p_flat, input, cost_fn, damping
            )

            # Trial step: move to new point and evaluate cost
            theta_new, _ = self._apply_step(theta_flat, p_flat)
            self.map.set_theta(self.map.unflatten_theta(theta_new))
            cost_new = self._cost_only(input, cost_fn)

            # Compute rho = actual_reduction / predicted_reduction
            actual_red = cost - cost_new
            rho = actual_red / (pred_red + 1e-16)

            # Handle NaN/Inf: treat as a very bad step
            rho = tf.where(tf.math.is_finite(rho), rho, tf.constant(-1.0, dtype=self.precision))

            # Update trust region radius
            p_norm = tf.norm(p_flat)

            new_delta = tf.cond(
                rho < 0.25,
                lambda: 0.25 * self.delta,
                lambda: self.delta,
            )

            new_delta = tf.cond(
                tf.logical_and(rho > 0.75, p_norm >= 0.9 * self.delta),
                lambda: tf.minimum(2.0 * new_delta, self.delta_max),
                lambda: new_delta,
            )

            self.delta.assign(new_delta)

            # Accept or reject step
            accept_step = rho > self.eta

            if accept_step:
                theta_flat = theta_new
                cost = cost_new
            else:
                self.map.set_theta(theta_backup)

            costs = costs.write(iter, cost)

            U, V = self.map.get_UV(input)
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