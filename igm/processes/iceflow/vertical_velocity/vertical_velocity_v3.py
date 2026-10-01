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
from igm.utils.grad.grad import grad_xy, grad_stag
from igm.utils.grad.compute_divflux_slope_limiter import compute_divflux_slope_limiter


def compute_vertical_velocity_v3(cfg: DictConfig, state: State) -> tf.Tensor:
    method = str(cfg.processes.iceflow.vertical_velocity.get("method", "incompressibility")).lower()
    if method == "fluxform":
        return compute_vertical_velocity_fluxform_v3(cfg, state)
    return compute_vertical_velocity_incompressibility_v3(cfg, state)


def _fluxform_nodes(discr_v):
    """Nodes ζ_q where the flux form is evaluated, the evaluation matrix E (u(ζ_q) = E U),
    the integration matrix I (∫_0^{ζ_q} u dζ = I U) and E^{-1} (None for nodal bases).

    Nodal bases (Lagrange, MOLHO): the basis' own nodes, recovered as ζ_l = Σ_n V_int[l, n]
    (V_int applied to the constant function); E = identity, I = V_int.
    Spectral bases (Legendre): Chebyshev-Lobatto nodes on [0, 1] (they include the bed and the
    surface and keep the interpolation well conditioned); E from the basis functions,
    I = E V_int, and W is projected back to coefficients with E^{-1}."""
    V_int = discr_v.V_int
    Nz = int(V_int.shape[0])
    zeta = tf.reduce_sum(V_int * discr_v.V_const[None, :], axis=1)
    is_nodal = bool(tf.reduce_all(tf.abs(tf.sort(zeta) - zeta) < 1e-6)) and float(zeta[0]) < 1e-6
    if is_nodal:
        return zeta, None, V_int, None
    k = tf.range(Nz, dtype=V_int.dtype)
    nodes = 0.5 * (1.0 - tf.cos(3.141592653589793 * k / float(Nz - 1)))
    E = tf.stack([tf.cast(f(nodes), V_int.dtype) for f in discr_v.basis_fct], axis=1)  # (Nq, Ndof)
    return nodes, E, tf.matmul(E, V_int), tf.linalg.inv(E)


def compute_vertical_velocity_fluxform_v3(cfg: DictConfig, state: State) -> tf.Tensor:
    """
    Kinematic FLUX FORM of the incompressibility integral, for any vertical basis:

        w(ζ_q) = u(ζ_q)·∇z_q − ∇·Q_q,   z_q = b + ζ_q H,   Q_q = ∫_b^{z_q} u dz = H ∫_0^{ζ_q} u dζ

    This is the same quantity as the matrix form (the two are equal analytically), but the
    horizontal divergence of the layer flux Q_q is taken with the SAME scheme as the thickness
    update (`thk`: upwind, slope-limited `compute_divflux_slope_limiter`, same `slope_type`). At the surface Q = H ū is the ice flux, so wvelsurf is exactly
    consistent with the mass conservation actually computed by the model
    (w_s = u_s·∇s − ∇·q = u_s·∇s + ∂H/∂t − smb).

    The slope ∇z_q is taken, by default, on the STAGGERED grid (corner gradient of the ice-flow
    energy, averaged back to the cell centres: `slope_stencil: staggered`) — the slope the velocity
    field was solved against; `slope_stencil: central` uses the unstaggered stencil of the matrix
    form, which amplifies the grid-scale roughness of the surface.

    Nodal bases (Lagrange, MOLHO) are evaluated at their own nodes; spectral bases (Legendre) at
    Chebyshev-Lobatto nodes, then projected back to coefficients (see `_fluxform_nodes`).
    Aletsch, 2026-10-01: grid-scale noise ratio of wvelsurf 0.46 (matrix form) → 0.21.
    """
    discr_v = state.iceflow.discr_v
    nodes, E, I, E_inv = _fluxform_nodes(discr_v)
    Nq = int(nodes.shape[0])

    stencil = str(cfg.processes.iceflow.vertical_velocity.get("slope_stencil", "staggered")).lower()

    def grad_centres(X):
        if stencil == "central":
            return grad_xy(X, state.dX, state.dX, False, "extrapolate")
        sx, sy = grad_stag(X, state.dX, state.dX)  # (ny-1, nx-1) at the cell corners

        def to_centres(a):
            a = tf.pad(a, [[1, 1], [1, 1]], "SYMMETRIC")
            return 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])

        return to_centres(sx), to_centres(sy)

    # thk-consistent divergence settings
    cfg_thk = cfg.processes.get("thk", None)
    slope_type = str(cfg_thk.get("slope_type", "superbee")) if cfg_thk is not None else "superbee"
    dt = tf.cast(getattr(state, "dt", tf.constant(1.0)), state.thk.dtype)

    if E is None:
        Uq, Vq = state.U, state.V
    else:
        Uq = tf.einsum("qn,nji->qji", E, state.U)
        Vq = tf.einsum("qn,nji->qji", E, state.V)
    IU = tf.einsum("qn,nji->qji", I, state.U)  # ∫_0^{ζ_q} u dζ
    IV = tf.einsum("qn,nji->qji", I, state.V)

    base = state.topg
    dbdx, dbdy = grad_centres(base)
    W = [Uq[0] * dbdx + Vq[0] * dbdy]  # ζ_0 = 0: kinematic condition at the bed

    for q in range(1, Nq):
        z_q = nodes[q]
        h_q = z_q * state.thk
        divQ = compute_divflux_slope_limiter(
            IU[q] / z_q, IV[q] / z_q, h_q, state.dx, state.dx, dt,
            slope_type=slope_type,
        )  # scalar dx, as in the thk module
        szx, szy = grad_centres(base + h_q)
        W.append(Uq[q] * szx + Vq[q] * szy - divQ)

    W = tf.stack(W, axis=0)
    if E_inv is not None:
        W = tf.einsum("nq,qji->nji", E_inv, W)  # back to basis coefficients
    return W


def compute_vertical_velocity_incompressibility_v3(cfg: DictConfig, state: State) -> tf.Tensor:

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
