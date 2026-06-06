#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Mohr-Coulomb sliding law.

Same generalised power-law kinematic form as Weertman/Budd (the bed
behaves elastically beneath a velocity-dependent shear stress), but the
reference shear stress is derived inside the law from a Mohr-Coulomb
yield criterion

    tau_ref = N * tan(phi)

so the law reads the effective pressure `N` from
`fieldin["effective_pressure"]` (in MPa) and computes the friction
angle `phi` from its own config block (uniform, or linearly
interpolated against bed elevation `topg = usurf - thk`). It does NOT
read `fieldin["slidingco"]`.

The resulting basal shear stress is

    tau_b = N * tan(phi) * (u_b / u_ref)^(1/m)

which is the same equation as the legacy `enthalpy/till/friction` +
Weertman pipeline produced once the `1e-6` Pa->MPa conversion and the
`u_ref^(-1/m)` baked into `state.slidingco` are folded into a single
clean form.
"""

import numpy as np
import tensorflow as tf
from typing import Dict

from ..sliding import SlidingComponent, mask_gr
from igm.processes.iceflow.horizontal import HorizontalDiscr
from igm.processes.iceflow.vertical import VerticalDiscr


class MohrCoulombParams(tf.experimental.ExtensionType):
    """Parameters for the Mohr-Coulomb sliding law."""

    regularization: float
    exponent: float
    u_ref: float  # (m/yr)
    phi: float  # uniform friction angle (deg); used unless bed_min/bed_max are both set
    phi_min: float  # friction angle at bed = bed_min (deg)
    phi_max: float  # friction angle at bed = bed_max (deg)
    bed_min: float  # m a.s.l.; set NaN to disable interpolation -> uniform phi
    bed_max: float  # m a.s.l.; set NaN to disable interpolation -> uniform phi
    tauc_min: float  # MPa, lower clamp on N*tan(phi) (after ice-free assignment)
    tauc_max: float  # MPa, upper clamp on N*tan(phi)
    tauc_ice_free: float  # MPa, HARD assignment in ice-free cells (h_ice = 0).
    # Matches the legacy enthalpy/till/friction behaviour:
    # without it, ice-free cells get the global tauc_min
    # floor, which is typically 10x too weak and gives the
    # unified solver a near-zero-friction region that
    # pulls the optimization toward unphysical fast flow
    # in those cells -> NaN at cold-start.
    rho_ratio: float
    use_mask_gr: bool


class MohrCoulomb(SlidingComponent):
    """Sliding component implementing the Mohr-Coulomb friction-angle law."""

    def __init__(self, params: MohrCoulombParams) -> None:
        self.name = "mohr_coulomb"
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        discr_h: HorizontalDiscr,
        discr_v: VerticalDiscr,
    ) -> tf.Tensor:
        return cost_mohr_coulomb(U, V, fieldin, discr_h, discr_v, self.params)


def cost_mohr_coulomb(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    params: MohrCoulombParams,
) -> tf.Tensor:
    """Compute the Mohr-Coulomb sliding cost.

    Reads `N` from `fieldin["effective_pressure"]` (MPa). The presence of
    this field is enforced by the iceflow energy validator
    (utils.get_energy_components).
    """

    h = fieldin["thk"]
    s = fieldin["usurf"]
    N = fieldin["effective_pressure"]
    dx = fieldin["dX"]

    topg = s - h

    V_b = discr_v.V_b
    dtype = U.dtype

    m = tf.cast(params.exponent, dtype)
    u_regu = tf.cast(params.regularization, dtype)
    u_ref = tf.cast(params.u_ref, dtype)
    rho_ratio = tf.cast(params.rho_ratio, dtype)
    use_mask_gr = tf.cast(params.use_mask_gr, tf.bool)
    tauc_min = tf.cast(params.tauc_min, dtype)
    tauc_max = tf.cast(params.tauc_max, dtype)
    tauc_ice_free = tf.cast(params.tauc_ice_free, dtype)

    # Compute friction angle phi (uniform or bed-elevation interpolated)
    phi = _compute_phi(topg, params)
    tan_phi = tf.math.tan(phi * tf.cast(np.pi / 180.0, dtype))

    # Mohr-Coulomb yield stress (MPa). Match the legacy ordering exactly:
    #   1. tau_c = N * tan(phi)
    #   2. in ice-free cells (h=0), HARD assign tauc_ice_free
    #   3. clip to [tauc_min, tauc_max]
    # Step (2) is critical at cold-start: without it, ice-free cells get only
    # the global tauc_min floor, which is too weak and causes the unified
    # solver to predict runaway velocities there.
    tauc = N * tan_phi
    tauc = tf.where(h > 0.0, tauc, tauc_ice_free)
    tauc = tf.clip_by_value(tauc, tauc_min, tauc_max)

    # Apply grounding mask on the reference stress
    if use_mask_gr:
        tauc = tauc * mask_gr(h, topg, rho_ratio)

    # Interpolate to horizontal quad points
    U_h = discr_h.interp_h(U)  # (batch, Nq_h, Nz, Ny-1, Nx-1)
    V_h = discr_h.interp_h(V)
    tauc_h = discr_h.interp_h(tauc)  # (batch, Nq_h, Ny-1, Nx-1)

    # Basal velocity -> (batch, Nq_h, Ny-1, Nx-1)
    ux_b = tf.einsum("z,bhzyx->bhyx", V_b, U_h)
    uy_b = tf.einsum("z,bhzyx->bhyx", V_b, V_h)

    # Bed-slope correction term
    b = s - h
    dbdx_h, dbdy_h = discr_h.grad_h(b, dx)

    u_corr_b = ux_b * dbdx_h + uy_b * dbdy_h
    u_b = tf.sqrt(ux_b**2 + uy_b**2 + u_regu**2 + u_corr_b**2)

    # cost = tauc * (u_b/u_ref)^p / p,  p = 1 + 1/m   -> derivative gives
    # tau_b = tauc * (u_b/u_ref)^(1/m), which is the Mohr-Coulomb form.
    p = 1.0 + 1.0 / m
    C_h = tauc_h / tf.pow(u_ref, 1.0 / m)
    cost_h = C_h * tf.pow(u_b, p) / p

    w_h = discr_h.w_h[tf.newaxis, :, tf.newaxis, tf.newaxis]
    return tf.reduce_sum(cost_h * w_h, axis=1)


def _compute_phi(topg: tf.Tensor, params: MohrCoulombParams) -> tf.Tensor:
    """Return phi in degrees: uniform, or linear interp in bed elevation.

    The legacy `compute_phi_tf` in `enthalpy/till/friction/utils.py` uses
    `bed_min/bed_max = None` to signal "uniform". Hydra dataclasses can't
    carry None inside `tf.experimental.ExtensionType`, so we encode the
    same intent with NaN sentinels in the YAML (or equivalently a hard
    test that both are positive-finite).
    """
    dtype = topg.dtype
    phi_uniform = tf.cast(params.phi, dtype)
    bed_min = tf.cast(params.bed_min, dtype)
    bed_max = tf.cast(params.bed_max, dtype)
    use_interp = tf.math.is_finite(bed_min) & tf.math.is_finite(bed_max)

    phi_lo = tf.cast(params.phi_min, dtype)
    phi_hi = tf.cast(params.phi_max, dtype)
    phi_interp = phi_lo + (phi_hi - phi_lo) * (topg - bed_min) / (bed_max - bed_min)
    phi_interp = tf.where(
        topg <= bed_min, phi_lo, tf.where(topg >= bed_max, phi_hi, phi_interp)
    )

    return tf.where(use_interp, phi_interp, phi_uniform * tf.ones_like(topg))
