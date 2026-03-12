#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Dict

from ..sliding import SlidingComponent
from igm.processes.iceflow.horizontal import HorizontalDiscr
from igm.processes.iceflow.vertical import VerticalDiscr
from igm.processes.iceflow.emulate.utils.misc import get_effective_pressure_precentage


class CoulombParams(tf.experimental.ExtensionType):
    """Parameters for Coulomb sliding law."""

    regu: float
    exponent: float
    mu: float
    u_ref: float  # (m/yr)


class Coulomb(SlidingComponent):
    """Sliding component implementing Coulomb's sliding law."""

    def __init__(self, params: CoulombParams):
        """Initialize Coulomb sliding component with parameters."""
        self.name = "coulomb"
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        discr_h: HorizontalDiscr,
        discr_v: VerticalDiscr,
    ) -> tf.Tensor:
        """Compute Coulomb sliding cost."""
        return cost_coulomb(U, V, fieldin, discr_h, discr_v, self.params)


def cost_coulomb(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    coulomb_params: CoulombParams,
) -> tf.Tensor:
    """Compute Coulomb sliding cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    tau_ref = fieldin["slidingco"]
    dx = fieldin["dX"]
    N = fieldin["effective_pressure"]
    
    V_b = discr_v.V_b

    dtype = U.dtype
    m = tf.cast(coulomb_params.exponent, dtype)
    u_regu = tf.cast(coulomb_params.regu, dtype)
    μ = tf.cast(coulomb_params.mu, dtype)
    u_ref = tf.cast(coulomb_params.u_ref, dtype)

    return _cost(U, V, h, s, tau_ref, dx, m, μ, u_regu, u_ref, discr_h, V_b)


@tf.function()
def _cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    N: tf.Tensor,
    s: tf.Tensor,
    tau_ref: tf.Tensor,
    dx: tf.Tensor,
    m: tf.Tensor,
    μ: tf.Tensor,
    u_regu: tf.Tensor,
    u_ref: tf.Tensor,
    discr_h: HorizontalDiscr,
    V_b: tf.Tensor,
) -> tf.Tensor:
    """
    Compute the Coulomb sliding law cost term.

    Calculates the sliding energy dissipation using a regularized
    Coulomb power law, following Shapero et al. (2021).

    Parameters
    ----------
    U : tf.Tensor
        Horizontal velocity along x axis (m/year)
    V : tf.Tensor
        Horizontal velocity along y axis (m/year)
    h : tf.Tensor
        Ice thickness (m)
    N : tf.Tensor
        Effective pressure (Mpa)
    s : tf.Tensor
        Upper-surface elevation (m)
    tau_ref : tf.Tensor
        Reference basal shear stress (MPa)
    u_ref : tf.Tensor
        Reference velocity (m/year)
    dx : tf.Tensor
        Grid spacing (m)
    m : tf.Tensor
        Coulomb exponent (-)
    μ: tf.Tensor
        Till coefficient (-)
    u_regu : tf.Tensor
        Regularization parameter for velocity magnitude (m/year)
    discr_h: HorizontalDiscr
        Horizontal discretization class (-)
    V_b : tf.Tensor
        Basal extraction vector: dofs -> basal (-)

    Returns
    -------
    tf.Tensor
        Coulomb sliding cost in MPa m/year
    """

    # Interpolate to horizontal quad points
    U_h = discr_h.interp_h(U)  # -> (batch, Nq_h, Nz, Ny-1, Nx-1)
    V_h = discr_h.interp_h(V)  # -> (batch, Nq_h, Nz, Ny-1, Nx-1)
    tau_ref_h = discr_h.interp_h(tau_ref)  # -> (batch, Nq_h, Ny-1, Nx-1)
    N_h = discr_h.interp_h(N)  # -> (batch, Nq_h, Ny-1, Nx-1)

    # Extract basal velocity -> (batch, Nq_h, Ny-1, Nx-1)
    ux_b = tf.einsum("z,bhzyx->bhyx", V_b, U_h)
    uy_b = tf.einsum("z,bhzyx->bhyx", V_b, V_h)

    # Compute bed gradient ∇b -> (batch, Nq_h, Ny-1, Nx-1)
    b = s - h
    dbdx_h, dbdy_h = discr_h.grad_h(b, dx)

    # Basal velocity magnitude with bed slope correction and regu
    u_corr_b = ux_b * dbdx_h + uy_b * dbdy_h
    u_b = tf.sqrt(ux_b**2 + uy_b**2 + u_regu**2 + u_corr_b**2)

    # Effective exponent
    p = 1.0 + 1.0 / m

    C_h = tau_ref_h / tf.pow(u_ref, 1.0 / m)

    # Compute smooth transition between Weertman and Coulomb (Shapero et al. 2021)
    τ_c = μ * N_h
    u_c = tf.pow(τ_c / C_h, m)
    
    # τ_c * [ (|u_b|^p + |u_c|^p)^(1/p) - u_c ]
    cost_h = τ_c * (tf.pow(tf.pow(u_b, p) + tf.pow(u_c, p), 1.0 / p) - u_c)

    # Integrate over horizontal quad points
    w_h = discr_h.w_h[tf.newaxis, :, tf.newaxis, tf.newaxis]
    return tf.reduce_sum(cost_h * w_h, axis=1)
