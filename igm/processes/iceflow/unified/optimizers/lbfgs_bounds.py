#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Optional, Tuple

from .lbfgs import OptimizerLBFGS
from .line_searches import LineSearches, ValueAndGradient
from igm.utils.math.norms import compute_norm

class OptimizerLBFGSBounds(OptimizerLBFGS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "lbfgs_bounded"

        if not hasattr(self.map, "get_box_bounds_flat"):
            raise ValueError(
                "❌ Mapping must provide get_box_bounds_flat() for bounded optimization."
            )

    @tf.function(reduce_retracing=True)
    def _project(self, theta: tf.Tensor, L: tf.Tensor, U: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(theta, L, U)

    @tf.function(reduce_retracing=True)
    def _get_mask(
        self, w: tf.Tensor, g: tf.Tensor, L: tf.Tensor, U: tf.Tensor
    ) -> tf.Tensor:
        tol = tf.constant(1e-7, w.dtype) if w.dtype == tf.float32 else tf.constant(1e-12, w.dtype)

        interior = tf.logical_and(w > L + tol, w < U - tol)
        at_lower = tf.logical_and(w <= L + tol, g < 0.0)
        at_upper = tf.logical_and(w >= U - tol, g > 0.0)
        return tf.logical_or(interior, tf.logical_or(at_lower, at_upper))

    def _force_descent(
        self, p_flat: tf.Tensor, grad_theta_flat: tf.Tensor, theta_flat: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        L, U = self.map.get_box_bounds_flat()

        # Free set at the point the step actually starts from. (Using the free
        # set of a further projected-gradient probe here is inconsistent with
        # the mask used to build the direction and can zero valid components.)
        mask = self._get_mask(theta_flat, grad_theta_flat, L, U)
        p_flat = tf.where(mask, p_flat, tf.zeros_like(p_flat))

        # Zero components pointing out of the box at active bounds; otherwise a
        # single such component makes alpha_max = 0 and kills the whole step.
        tol = tf.constant(1e-7, theta_flat.dtype) if theta_flat.dtype == tf.float32 else tf.constant(1e-12, theta_flat.dtype)
        outward = ((theta_flat <= L + tol) & (p_flat < 0.0)) | (
            (theta_flat >= U - tol) & (p_flat > 0.0)
        )
        p_flat = tf.where(outward, tf.zeros_like(p_flat), p_flat)

        dot_gp = self._dot(grad_theta_flat, p_flat)

        # Masked steepest descent fallback (inward-pointing by construction of
        # _get_mask: components kept at a bound have a gradient pushing inward).
        p_sd = tf.where(mask, -grad_theta_flat, tf.zeros_like(grad_theta_flat))
        return tf.cond(dot_gp >= 0.0, lambda: p_sd, lambda: p_flat), mask

    def _apply_step(
        self, theta_flat: tf.Tensor, alpha: tf.Tensor, p_flat: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        L, U = self.map.get_box_bounds_flat()
        theta_trial = theta_flat + alpha * p_flat
        theta_proj = self._project(theta_trial, L, U)
        return theta_proj, theta_trial

    @tf.function
    def _line_search(self, theta_flat: tf.Tensor, p_flat: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        L, U = self.map.get_box_bounds_flat()
        amax = self._alpha_max(theta_flat, p_flat, L, U)

        def eval_fn(alpha: tf.Tensor) -> ValueAndGradient:
            alpha_eff = tf.minimum(alpha, amax)

            theta_backup = self.map.copy_theta(self.map.get_theta())
            theta_alpha, _ = self._apply_step(theta_flat, alpha_eff, p_flat)

            self.map.set_theta(self.map.unflatten_theta(theta_alpha))
            f, _, grad = self._get_grad(input)
            grad_flat = self.map.flatten_theta(grad)

            mask = self._get_mask(theta_alpha, grad_flat, L, U)
            p_masked = tf.where(mask, p_flat, tf.zeros_like(p_flat))
            df = self._dot(grad_flat, p_masked)

            self.map.set_theta(theta_backup)
            return ValueAndGradient(x=alpha_eff, f=f, df=tf.cast(df, grad_flat.dtype))

        return self.line_search.search(theta_flat, p_flat, eval_fn)


    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:

        L, U = self.map.get_box_bounds_flat()
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        theta_proj = self._project(theta_flat, L, U)
        self.map.set_theta(self.map.unflatten_theta(theta_proj))

        return super().minimize_impl(inputs)

    @tf.function(reduce_retracing=True)
    def _alpha_max(self, w: tf.Tensor, p: tf.Tensor, L: tf.Tensor, U: tf.Tensor) -> tf.Tensor:
        # Maximum alpha such that w + alpha p stays within [L, U] componentwise
        inf = tf.constant(float("inf"), dtype=w.dtype)

        p_pos = p > 0.0
        p_neg = p < 0.0

        a_pos = tf.where(p_pos, (U - w) / p, inf)
        a_neg = tf.where(p_neg, (L - w) / p, inf)

        a = tf.minimum(a_pos, a_neg)
        amax = tf.reduce_min(a)

        return tf.maximum(amax, tf.constant(0.0, dtype=w.dtype))
    
    def _clip_alpha(self, alpha: tf.Tensor, theta_flat: tf.Tensor, p_flat: tf.Tensor) -> tf.Tensor:
        L, U = self.map.get_box_bounds_flat()
        amax = self._alpha_max(theta_flat, p_flat, L, U)
        return tf.minimum(alpha, amax)
    
    def _constrain_pair(
        self,
        s: tf.Tensor,
        y: tf.Tensor,
        w_prev: tf.Tensor,
        theta_trial: Optional[tf.Tensor],
        mask: Optional[tf.Tensor],
        w_new: Optional[tf.Tensor] = None,
        g_new: Optional[tf.Tensor] = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        # w_new is the projected (actual) new iterate; theta_trial is pre-projection
        if w_new is None or g_new is None or theta_trial is None or mask is None:
            return tf.zeros_like(s), tf.zeros_like(y)

        L, U = self.map.get_box_bounds_flat()

        # Did projection change any component?
        tol = tf.constant(1e-7, w_new.dtype) if w_new.dtype == tf.float32 else tf.constant(1e-12, w_new.dtype)
        proj_changed = tf.abs(w_new - theta_trial) > tol

        # Free-set at the NEW point (based on descent feasibility)
        mask_new = self._get_mask(w_new, g_new, L, U)

        # Only keep curvature info where variable stayed free AND no projection happened
        keep = tf.logical_and(tf.logical_and(mask, mask_new), tf.logical_not(proj_changed))

        s = tf.where(keep, s, tf.zeros_like(s))
        y = tf.where(keep, y, tf.zeros_like(y))
        return s, y

    def _get_grad_norm(
        self, grad_u: list[tf.Tensor], grad_theta: list[tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        grad_u_norm, _ = super()._get_grad_norm(grad_u, grad_theta)

        theta_flat = self.map.flatten_theta(self.map.get_theta())
        grad_theta_flat = self.map.flatten_theta(grad_theta)
        L, U = self.map.get_box_bounds_flat()

        mask = self._get_mask(theta_flat, grad_theta_flat, L, U)
        proj_grad = tf.where(mask, grad_theta_flat, tf.zeros_like(grad_theta_flat))

        grad_theta_norm = compute_norm(proj_grad, ord=self.ord_grad_theta)

        return grad_u_norm, grad_theta_norm
        
    @tf.function(reduce_retracing=True)
    def _projected_grad_dir(
        self, w: tf.Tensor, g: tf.Tensor, L: tf.Tensor, U: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Projected steepest-descent direction and mask."""
        mask = self._get_mask(w, g, L, U)
        d = tf.where(mask, -g, tf.zeros_like(g))
        return d, mask


    @tf.function(reduce_retracing=True)
    def _cauchy_point_approx(
        self,
        w: tf.Tensor,
        g: tf.Tensor,
        L: tf.Tensor,
        U: tf.Tensor,
        step_scale: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Lightweight Cauchy point approximation:
        d = projected steepest descent
        t = min(step_scale, alpha_max(w, d, L, U))
        w_c = clip(w + t d)
        Returns (w_c, mask_c, t)
        """
        d, _ = self._projected_grad_dir(w, g, L, U)
        d_norm_inf = tf.reduce_max(tf.abs(d))

        def no_move():
            mask_c = self._get_mask(w, g, L, U)
            return w, mask_c, tf.cast(0.0, w.dtype)

        def do_move():
            amax = self._alpha_max(w, d, L, U)
            t = tf.minimum(tf.cast(step_scale, w.dtype), tf.cast(amax, w.dtype))
            w_c = self._project(w + t * d, L, U)
            # Still using g at w (cheap approximation); if you want, you can recompute g at w_c (not lightweight).
            mask_c = self._get_mask(w_c, g, L, U)
            return w_c, mask_c, t

        return tf.cond(d_norm_inf <= 0.0, no_move, do_move)
        
    @tf.function(reduce_retracing=True)
    def _step_base_point(
        self,
        theta_flat: tf.Tensor,
        grad_theta_flat: tf.Tensor,
        input: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Bounded subspace step (L-BFGS-B-ish active-set prediction).

        The Cauchy probe is used ONLY to predict the free set: the step itself
        starts from the current iterate. Moving the base point to an
        unvalidated, unscaled projected-gradient point breaks monotonicity
        (no decrease check exists there), breaks the secant pairing of the
        curvature pairs, and costs an extra gradient evaluation per iteration.
        """
        del input  # no extra evaluation needed for the prediction-only probe

        L, U = self.map.get_box_bounds_flat()

        # Free set predicted at the Cauchy probe point (gradient at the current
        # iterate is reused — the cheap approximation already used inside
        # _cauchy_point_approx).
        _, mask_c, _ = self._cauchy_point_approx(
            theta_flat, grad_theta_flat, L, U, step_scale=tf.cast(1.0, theta_flat.dtype)
        )

        # Restrict the model gradient to the predicted free subspace.
        grad_masked = tf.where(mask_c, grad_theta_flat, tf.zeros_like(grad_theta_flat))

        return theta_flat, grad_masked, mask_c


    def _mask_memory_for_subspace(
        self,
        s_list: tf.Tensor,
        y_list: tf.Tensor,
        mask: Optional[tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # Mask is expected to be a boolean tensor (free set at Cauchy)
        if mask is None:
            return s_list, y_list
        m = tf.cast(mask, s_list.dtype)[None, :]  # [1, w_dim] broadcast over memory rows
        return s_list * m, y_list * m
