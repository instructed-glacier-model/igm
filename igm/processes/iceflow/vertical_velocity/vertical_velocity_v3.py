#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Vertical velocity computation from incompressibility constraint.

This module computes the vertical velocity field w(x,y,ζ) from the horizontal
velocity fields u(x,y,ζ) and v(x,y,ζ) using the incompressibility condition
in terrain-following coordinates.

Physical Background
-------------------
Ice is treated as incompressible, so the velocity field is divergence-free:

    ∂u/∂x + ∂v/∂y + ∂w/∂z = 0

where derivatives are taken at constant z (physical height). Rearranging:

    ∂w/∂z = -(∂u/∂x + ∂v/∂y)  =  -div_z

Integrating from the bed (z = b) to height z:

    w(z) = w_b - ∫_b^z div_z dz'

where w_b = u_b · ∇b is the basal vertical velocity from the kinematic
boundary condition (ice velocity parallel to bed).

Terrain-Following Coordinates
-----------------------------
We use a terrain-following vertical coordinate ζ ∈ [0,1]:

    z = b(x,y) + ζ · H(x,y)

where b is bed elevation and H = s - b is ice thickness.

The physical derivatives relate to ζ-coordinate derivatives via:

    ∂u/∂x|_z = ∂u/∂x|_ζ + (∂u/∂ζ)(∂ζ/∂x)|_z

The coordinate transformation gives:

    (∂ζ/∂x)|_z · H = -[(1-ζ) ∂b/∂x + ζ ∂s/∂x]

So the physical divergence becomes:

    div_z = div_ζ + (1/H)(∂u/∂ζ) · [-(1-ζ)∂b/∂x - ζ∂s/∂x]
                  + (1/H)(∂v/∂ζ) · [-(1-ζ)∂b/∂y - ζ∂s/∂y]

Changing variables in the integral (dz = H dζ):

    w(ζ) = w_b - H ∫₀^ζ div_ζ dζ'
               + ∂b/∂x ∫₀^ζ (∂u/∂ζ')(1-ζ') dζ'
               + ∂s/∂x ∫₀^ζ (∂u/∂ζ') ζ' dζ'
               + ∂b/∂y ∫₀^ζ (∂v/∂ζ')(1-ζ') dζ'
               + ∂s/∂y ∫₀^ζ (∂v/∂ζ') ζ' dζ'

Exact Integration via Integration by Parts
------------------------------------------
For u = Σₙ Uₙ φₙ(ζ), the terrain correction integrals are:

    ∫₀^ζ φ'ₙ(ζ')(1-ζ') dζ'  and  ∫₀^ζ φ'ₙ(ζ') ζ' dζ'

Using integration by parts, these evaluate exactly to:

    ψᵇₙ(ζ) = (1-ζ) φₙ(ζ) - φₙ(0) + Φₙ(ζ)
    ψˢₙ(ζ) = ζ φₙ(ζ) - Φₙ(ζ)

where Φₙ(ζ) = ∫₀^ζ φₙ(ζ') dζ' is the antiderivative.

These satisfy ψᵇₙ(0) = ψˢₙ(0) = 0, ensuring w(0) = w_b.

Final Formula in Coefficient Form
---------------------------------
The vertical velocity DOFs are computed as:

    W = w_b · V_const
      - H · V_int · div_ζ
      + ∂b/∂x · V_corr_b · U + ∂s/∂x · V_corr_s · U
      + ∂b/∂y · V_corr_b · V + ∂s/∂y · V_corr_s · V

where:
    - V_const: coefficients for constant function (handles w_b contribution)
    - V_int: integration matrix (handles -H ∫ div_ζ dζ' term)
    - V_corr_b, V_corr_s: terrain correction matrices (handle coordinate transformation)

Output Interpretation
---------------------
The output W has shape (Ndof, Ny, Nx):

- For nodal bases (Lagrange, MOLHO, SSA): W[n,:,:] = w(ζₙ) at node n
- For spectral bases (Legendre): W[n,:,:] = coefficient of mode n

To evaluate w at any point:
    w(ζ) = Σₙ Wₙ φₙ(ζ)  or equivalently  w = V_q @ W  at quadrature points
"""

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.grad.grad import grad_xy


def compute_vertical_velocity_v3(cfg: DictConfig, state: State) -> tf.Tensor:

    # Retrieve vertical discretization
    discr_v = state.iceflow.discr_v

    # Compute basal vertical velocity
    dbdx, dbdy = grad_xy(state.topg, state.dX, state.dX, False, "extrapolate")
    w_b = state.uvelbase * dbdx + state.vvelbase * dbdy

    # Compute divergence flux
    dudx, _ = grad_xy(state.U, state.dX, state.dX, False)
    _, dvdy = grad_xy(state.V, state.dX, state.dX, False)

    # Constant term due to basal velocity
    W = w_b[None, ...] * discr_v.V_const[:, None, None]

    # Variable term due to divergence flux
    W = W - state.thk[None, ...] * tf.einsum("lk,kji->lji", discr_v.V_int, dudx + dvdy)

    # Add correction terms for terrain-following coordinates
    dsdx, dsdy = grad_xy(state.usurf, state.dX, state.dX, False, "extrapolate")

    corr_b_U = tf.einsum("mn,nkl->mkl", discr_v.V_corr_b, state.U)
    corr_s_U = tf.einsum("mn,nkl->mkl", discr_v.V_corr_s, state.U)
    corr_b_V = tf.einsum("mn,nkl->mkl", discr_v.V_corr_b, state.V)
    corr_s_V = tf.einsum("mn,nkl->mkl", discr_v.V_corr_s, state.V)

    W = W + dbdx[None, ...] * corr_b_U + dsdx[None, ...] * corr_s_U
    W = W + dbdy[None, ...] * corr_b_V + dsdy[None, ...] * corr_s_V

    return W
