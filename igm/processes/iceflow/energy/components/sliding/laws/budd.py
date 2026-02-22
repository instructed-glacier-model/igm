#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Dict

from ..sliding import SlidingComponent, mask_gr
from igm.processes.iceflow.horizontal import HorizontalDiscr
from igm.processes.iceflow.vertical import VerticalDiscr
from igm.processes.iceflow.emulate.utils.misc import get_effective_pressure_precentage


class BuddParams(tf.experimental.ExtensionType):
    """Parameters for Budd sliding law."""

    regu: float
    exponent: float
    rho_ratio: float


class Budd(SlidingComponent):
    """Sliding component implementing Budd's sliding law."""

    def __init__(self, params: BuddParams) -> None:
        """Initialize Budd sliding component with parameters."""
        self.name = "budd"
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        discr_h: HorizontalDiscr,
        discr_v: VerticalDiscr,
    ) -> tf.Tensor:
        """Compute Budd sliding cost."""
        return cost_budd(U, V, fieldin, discr_h, discr_v, self.params)


def cost_budd(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    budd_params: BuddParams,
) -> tf.Tensor:
    """Compute Budd sliding cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    C = fieldin["slidingco"]
    dx = fieldin["dX"]

    V_b = discr_v.V_b

    dtype = U.dtype
    m = tf.cast(budd_params.exponent, dtype)
    u_regu = tf.cast(budd_params.regu, dtype)
    rho_ratio = tf.cast(budd_params.rho_ratio, dtype)

    return _cost(U, V, h, s, C, dx, m, u_regu, rho_ratio, discr_h, V_b)


@tf.function()
def _cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    C: tf.Tensor,
    dx: tf.Tensor,
    m: tf.Tensor,
    u_regu: tf.Tensor,
    rho_ratio: tf.Tensor,
    discr_h: HorizontalDiscr,
    V_b: tf.Tensor,
) -> tf.Tensor:
    """
    Compute the Budd sliding law cost term.

    Calculates the sliding energy dissipation using Budd's power law:
    C * N * |u_b|^(1+1/m) / (1+1/m), where u_b is the basal velocity magnitude
    corrected for bed topography.

    Parameters
    ----------
    U : tf.Tensor
        Horizontal velocity along x axis (m/year)
    V : tf.Tensor
        Horizontal velocity along y axis (m/year)
    h : tf.Tensor
        Ice thickness (m)
    s : tf.Tensor
        Upper-surface elevation (m)
    C : tf.Tensor
        Friction coefficient ((m/year)^(-1/m))
    dx : tf.Tensor
        Grid spacing (m)
    m : tf.Tensor
        Budd exponent (-)
    u_regu : tf.Tensor
        Regularization parameter for velocity magnitude (m/year)
    rho_ratio : tf.Tensor
        Water density / ice density ratio (-)
    discr_h: HorizontalDiscr
        Horizontal discretization class (-)
    V_b : tf.Tensor
        Basal extraction vector: dofs -> basal (-)

    Returns
    -------
    tf.Tensor
        Budd sliding cost in MPa m/year
    """

    # Apply grounding mask to sliding coefficient
    topg = s - h
    gr = mask_gr(h, topg, rho_ratio)
    C = C * gr

    # Interpolate to horizontal quad points
    U_h = discr_h.interp_h(U)  # -> (batch, Nq_h, Nz, Ny-1, Nx-1)
    V_h = discr_h.interp_h(V)  # -> (batch, Nq_h, Nz, Ny-1, Nx-1)
    C_h = discr_h.interp_h(C)  # -> (batch, Nq_h, Ny-1, Nx-1)
    h_h = discr_h.interp_h(h)  # -> (batch, Nq_h, Ny-1, Nx-1)

    # Extract basal velocity -> (batch, Nq_h, Ny-1, Nx-1)
    ux_b = tf.einsum("z,bhzyx->bhyx", V_b, U_h)
    uy_b = tf.einsum("z,bhzyx->bhyx", V_b, V_h)

    # Compute bed gradient ∇b -> (batch, Nq_h, Ny-1, Nx-1)
    b = s - h
    dbdx_h, dbdy_h = discr_h.grad_h(b, dx)

    # Basal velocity magnitude with bed slope correction and regu
    u_corr_b = ux_b * dbdx_h + uy_b * dbdy_h
    u_b = tf.sqrt(ux_b**2 + uy_b**2 + u_regu**2 + u_corr_b**2)

    # Effective pressure at quad points
    # TODO: should be within the inputs
    N_h = get_effective_pressure_precentage(h_h, percentage=0.0)
    N_h = tf.where(N_h < 1e-3, 1e-3, N_h)

    # Effective exponent
    p = 1.0 + 1.0 / m

    # C * N * |u_b|^p / p at each quad point
    cost_h = C_h * N_h * tf.pow(u_b, p) / p

    # Integrate over horizontal quad points
    w_h = discr_h.w_h[tf.newaxis, :, tf.newaxis, tf.newaxis]
    return tf.reduce_sum(cost_h * w_h, axis=1)
