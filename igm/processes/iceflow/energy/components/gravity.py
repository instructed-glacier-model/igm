#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict

import tensorflow as tf
from omegaconf import DictConfig

from igm.processes.iceflow.discretization import HorizontalDiscr, VerticalDiscr

from .energy import EnergyComponent


class GravityParams(tf.experimental.ExtensionType):
    """Parameters for gravity energy component."""

    rho: float
    g: float
    fnge: bool


class GravityComponent(EnergyComponent):
    """Energy component representing gravitational potential energy."""

    name = "gravity"

    def __init__(self, params: GravityParams) -> None:
        """Initialize gravity component with parameters."""
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        discr_h: HorizontalDiscr,
        discr_v: VerticalDiscr,
    ) -> tf.Tensor:
        """Compute gravitational energy cost."""
        return cost_gravity(U, V, fieldin, discr_h, discr_v, self.params)


def get_gravity_params_args(cfg: DictConfig) -> Dict[str, Any]:
    """Extract gravity parameters from configuration."""

    cfg_physics = cfg.processes.iceflow.physics

    return {
        "rho": cfg_physics.ice_density,
        "g": cfg_physics.gravity_cst,
        "fnge": cfg_physics.force_negative_gravitational_energy,
    }


def cost_gravity(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    gravity_params: GravityParams,
) -> tf.Tensor:
    """Compute gravitational energy cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    dx = fieldin["dX"]

    V_q = discr_v.V_q
    w_v = discr_v.w

    dtype = U.dtype
    rho = tf.cast(gravity_params.rho, dtype)
    g = tf.cast(gravity_params.g, dtype)
    fnge = gravity_params.fnge

    return _cost(U, V, h, s, dx, rho, g, fnge, discr_h, V_q, w_v)


@tf.function()
def _cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    dx: tf.Tensor,
    rho: tf.Tensor,
    g: tf.Tensor,
    fnge: bool,
    discr_h: HorizontalDiscr,
    V_q: tf.Tensor,
    w_v: tf.Tensor,
) -> tf.Tensor:
    """
    Compute the gravitational energy cost term.

    Calculates the work done by gravity: ρ g h ∫ (u,v)·∇s dz

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
    dx : tf.Tensor
        Grid spacing (m)
    rho : tf.Tensor
        Ice density (kg m^-3)
    g : tf.Tensor
        Gravity acceleration (m s^-2)
    fnge : bool
        Force negative gravitational energy flag
    discr_h : HorizontalDiscr
        Horizontal discretization class (-)
    V_q : tf.Tensor
        Quadrature matrix: dofs -> quads (-)
    w_v : tf.Tensor
        Weights for vertical integration (-)

    Returns
    -------
    tf.Tensor
        Gravitational energy cost in MPa m/year
    """

    # Interpolate to horizontal quad points
    u_h = discr_h.interp_h(U)  # -> (batch, Nq_h, Nz, Ny-1, Nx-1)
    v_h = discr_h.interp_h(V)  # -> (batch, Nq_h, Nz, Ny-1, Nx-1)
    h_h = discr_h.interp_h(h)  # -> (batch, Nq_h, Ny-1, Nx-1)

    # Evaluate at vertical quad points -> (batch, Nq_h, Nq_v, Ny-1, Nx-1)
    u_hv = tf.einsum("vz,bhzyx->bhvyx", V_q, u_h)
    v_hv = tf.einsum("vz,bhzyx->bhvyx", V_q, v_h)

    # Surface gradient at horizontal quad points -> (batch, Nq_h, Ny-1, Nx-1)
    dsdx_h, dsdy_h = discr_h.grad_h(s, dx)

    # Expand surface gradient for broadcasting -> (batch, Nq_h, Nq_v, Ny-1, Nx-1)
    dsdx_hv = dsdx_h[:, :, tf.newaxis, :, :]
    dsdy_hv = dsdy_h[:, :, tf.newaxis, :, :]

    # (u,v)·∇s
    u_dot_grad_s = u_hv * dsdx_hv + v_hv * dsdy_hv

    # Optionally enforce non-positive (flow not going uphill)
    if fnge:
        u_dot_grad_s = tf.minimum(u_dot_grad_s, 0.0)

    # Integrate: ρ g h ∫ (u,v)·∇s dz
    w_h = discr_h.w_h[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
    w_v = w_v[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]
    h_h = h_h[:, :, tf.newaxis, :, :]
    return 1e-6 * rho * g * tf.reduce_sum(h_h * u_dot_grad_s * w_h * w_v, axis=[1, 2])
