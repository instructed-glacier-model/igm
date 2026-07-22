#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Shared cost kernel for power-law sliding (Weertman & Budd).

Both Weertman and Budd are special cases of the general power law

    cost = (tau_ref / (N_ref^q * u_ref^(1/m))) * N^q * |u_b|^p / p,    p = 1 + 1/m

with N=1, N_ref=1, q=1 recovering Weertman exactly. The exponent q
controls the sensitivity to effective pressure:
  q = 1.0  standard linear Budd (strong thickness dependence)
  q = 0.5  sublinear Budd (Tsai 2015; weaker, recommended for tidewater)
  q = 0.0  Weertman (no N dependence)

Keeping the math in one place avoids the two laws drifting out of sync.

Coulomb (Shapero 2021 regularised) has a different functional form and
stays in its own file.
"""

import tensorflow as tf

from ..sliding import mask_gr
from igm.processes.iceflow.horizontal import HorizontalDiscr


@tf.function()
def power_law_cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    tau_ref: tf.Tensor,
    N: tf.Tensor,
    dx: tf.Tensor,
    m: tf.Tensor,
    u_regu: tf.Tensor,
    u_ref: tf.Tensor,
    N_ref: tf.Tensor,
    q: tf.Tensor,
    rho_ratio: tf.Tensor,
    use_mask_gr: bool,
    discr_h: HorizontalDiscr,
    V_b: tf.Tensor,
) -> tf.Tensor:
    """Generalised Weertman/Budd power-law sliding cost.

    Parameters
    ----------
    U, V : (batch, Nz, Ny, Nx)  horizontal velocities (m/yr)
    h, s : (batch, Ny, Nx)      thickness & upper-surface elevation (m)
    tau_ref : (batch, Ny, Nx)   reference basal shear stress (MPa)
    N       : (batch, Ny, Nx)   effective pressure (any consistent unit;
                                pass ones for Weertman)
    dx                          grid spacing (m)
    m                           power-law exponent
    u_regu                      velocity regularisation (m/yr)
    u_ref                       reference basal velocity (m/yr)
    N_ref                       reference effective pressure (same unit
                                as N; pass 1.0 for Weertman)
    q                           effective-pressure exponent (1.0 = linear
                                Budd, 0.5 = sublinear Tsai, 0.0 = Weertman)
    rho_ratio                   rho_water / rho_ice (for grounding mask)
    use_mask_gr                 if True, zero tau_ref on floating ice
    discr_h                     horizontal discretisation
    V_b                         basal extraction vector (dofs -> basal)

    Returns
    -------
    (batch, Ny-1, Nx-1) energy cost in MPa m/yr.
    """

    # Apply grounding mask to basal shear stress
    if use_mask_gr:
        topg = s - h
        tau_ref = tau_ref * mask_gr(h, topg, rho_ratio)

    # Interpolate to horizontal quad points
    U_h = discr_h.interp_h(U)  # (batch, Nq_h, Nz, Ny-1, Nx-1)
    V_h = discr_h.interp_h(V)
    tau_ref_h = discr_h.interp_h(tau_ref)  # (batch, Nq_h, Ny-1, Nx-1)
    N_h = discr_h.interp_h(N)  # (batch, Nq_h, Ny-1, Nx-1)

    # Extract basal velocity -> (batch, Nq_h, Ny-1, Nx-1)
    ux_b = tf.einsum("z,bhzyx->bhyx", V_b, U_h)
    uy_b = tf.einsum("z,bhzyx->bhyx", V_b, V_h)

    # Bed gradient ∇b -> (batch, Nq_h, Ny-1, Nx-1)
    b = s - h
    dbdx_h, dbdy_h = discr_h.grad_h(b, dx)

    # Basal speed with bed-slope correction and regularisation
    u_corr_b = ux_b * dbdx_h + uy_b * dbdy_h
    u_b = tf.sqrt(ux_b**2 + uy_b**2 + u_regu**2 + u_corr_b**2)

    # Power-law cost with effective-pressure exponent q
    p = 1.0 + 1.0 / m
    C_h = tau_ref_h / (tf.pow(N_ref, q) * tf.pow(u_ref, 1.0 / m))
    cost_h = C_h * tf.pow(N_h, q) * tf.pow(u_b, p) / p

    # Integrate over horizontal quad points
    w_h = discr_h.w_h[tf.newaxis, :, tf.newaxis, tf.newaxis]
    return tf.reduce_sum(cost_h * w_h, axis=1)
