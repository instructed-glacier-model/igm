#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Dict

from ..sliding import SlidingComponent
from igm.processes.iceflow.horizontal import HorizontalDiscr
from igm.processes.iceflow.vertical import VerticalDiscr


class WeertmanParams(tf.experimental.ExtensionType):
    """Parameters for Weertman sliding law."""

    regu: float
    exponent: float
    u_ref: float  # (m/yr)


class Weertman(SlidingComponent):
    """Sliding component implementing Weertman's sliding law."""

    def __init__(self, params: WeertmanParams) -> None:
        """Initialize Weertman sliding component with parameters."""
        self.name = "weertman"
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        discr_h: HorizontalDiscr,
        discr_v: VerticalDiscr,
    ) -> tf.Tensor:
        """Compute Weertman sliding cost."""
        return cost_weertman(U, V, fieldin, discr_h, discr_v, self.params)


def cost_weertman(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    weertman_params: WeertmanParams,
) -> tf.Tensor:
    """Compute Weertman sliding cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    tau_ref = fieldin["slidingco"]
    dx = fieldin["dX"]

    V_b = discr_v.V_b

    dtype = U.dtype
    m = tf.cast(weertman_params.exponent, dtype)
    u_regu = tf.cast(weertman_params.regu, dtype)
    u_ref = tf.cast(weertman_params.u_ref, dtype)

    return _cost(U, V, h, s, tau_ref, dx, m, u_regu, u_ref, discr_h, V_b)


@tf.function()
def _cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    tau_ref: tf.Tensor,
    dx: tf.Tensor,
    m: tf.Tensor,
    u_regu: tf.Tensor,
    u_ref: tf.Tensor,
    discr_h: HorizontalDiscr,
    V_b: tf.Tensor,
) -> tf.Tensor:
    """
    Compute the Weertman sliding law cost term.

    Calculates the sliding energy dissipation using Weertman's power law:
    C * |u_b|^(1+1/m) / (1+1/m), where u_b is the basal velocity magnitude
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
    tau_ref : tf.Tensor
        Reference basal shear stress τ_ref (MPa), defined at speed u_ref
    u_ref : tf.Tensor
        Reference basal velocity magnitude (m/year)
    dx : tf.Tensor
        Grid spacing (m)
    m : tf.Tensor
        Weertman exponent (-)
    u_regu : tf.Tensor
        Regularization parameter for velocity magnitude (m/year)
    discr_h: HorizontalDiscr
        Horizontal discretization class (-)
    V_b : tf.Tensor
        Basal extraction vector: dofs -> basal (-)

    Returns
    -------
    tf.Tensor
        Weertman sliding cost in MPa m/year
    """

    # Interpolate to horizontal quad points
    U_h = discr_h.interp_h(U)  # -> (batch, Nq_h, Nz, Ny-1, Nx-1)
    V_h = discr_h.interp_h(V)  # -> (batch, Nq_h, Nz, Ny-1, Nx-1)
    tau_ref_h = discr_h.interp_h(tau_ref)  # -> (batch, Nq_h, Ny-1, Nx-1)

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

    # C * |u_b|^p / p at each quad point
    cost_h = C_h * tf.pow(u_b, p) / p

    # cost_h = (tau_ref_h * u_ref) * tf.pow(u_b / u_ref, p) / p

    # Integrate over horizontal quad points
    w_h = discr_h.w_h[tf.newaxis, :, tf.newaxis, tf.newaxis]
    return tf.reduce_sum(cost_h * w_h, axis=1)
