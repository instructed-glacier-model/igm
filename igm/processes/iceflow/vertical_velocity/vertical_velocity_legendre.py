#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from igm.utils.grad.grad import grad_xy


def compute_vertical_velocity_legendre(cfg, state):
    """
    Compute vertical velocity in Legendre basis from incompressibility.

    From ∂w/∂z = -(∂u/∂x + ∂v/∂y), integrating from the bed:
        w(ζ) = ub_z - H ∫₀^ζ (∂u/∂x + ∂v/∂y) dζ'
    """

    discr_v = state.iceflow.discr_v

    # Basal vertical velocity from kinematic boundary condition: w_b = u_b · ∇b
    dbdx, dbdy = grad_xy(state.topg, state.dX, state.dX, False, "extrapolate")
    w_b = state.uvelbase * dbdx + state.vvelbase * dbdy

    # Horizontal divergence in Legendre basis: ∂u/∂x + ∂v/∂y
    dUdx, _ = grad_xy(state.U, state.dX, state.dX, False)
    _, dVdy = grad_xy(state.V, state.dX, state.dX, False)
    div_zeta = dUdx + dVdy

    # Basal velocity term: w_b only contributes to mode 0 (since P₀ = 1)
    W = w_b[None, ...] * discr_v.V_const[:, None, None]

    # Divergence integral term: -H · V_int · div
    W = W - state.thk[None, ...] * tf.einsum("mn,nkl->mkl", discr_v.V_int, div_zeta)

    return W
