#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict

from .energy import EnergyComponent
from igm.processes.iceflow.horizontal import HorizontalDiscr
from igm.processes.iceflow.vertical import VerticalDiscr
from igm.utils.grad.strain_rate import (
    compute_eps_dot2,
    correct_grad_zeta_to_z,
    dampen_eps_dot_z_floating,
)


class ViscosityParams(tf.experimental.ExtensionType):
    """Parameters for viscous energy component."""

    n: float
    h_min: float
    eps_dot_regu: float
    eps_dot_min: float
    eps_dot_max: float


class ViscosityComponent(EnergyComponent):
    """Energy component representing viscous energy dissipation."""

    name = "viscosity"

    def __init__(self, params) -> None:
        """Initialize viscous component with parameters."""
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        discr_h: HorizontalDiscr,
        discr_v: VerticalDiscr,
    ) -> tf.Tensor:
        """Compute viscous energy cost."""
        return cost_viscosity(U, V, fieldin, discr_h, discr_v, self.params)


def get_viscosity_params_args(cfg) -> Dict[str, Any]:
    """Extract viscous parameters from configuration."""

    cfg_physics = cfg.processes.iceflow.physics

    return {
        "n": cfg_physics.viscosity.exponent,
        "h_min": cfg_physics.thr_ice_thk,
        "eps_dot_regu": cfg_physics.viscosity.regularization,
        "eps_dot_min": cfg_physics.min_sr,
        "eps_dot_max": cfg_physics.max_sr,
    }


def cost_viscosity(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    viscosity_params: ViscosityParams,
) -> tf.Tensor:
    """Compute viscous energy cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    A = fieldin["arrhenius"]
    dx = fieldin["dX"]

    V_q = discr_v.V_q
    V_q_grad = discr_v.V_q_grad
    w_v = discr_v.w
    zeta_v = discr_v.zeta

    dtype = U.dtype
    n = tf.cast(viscosity_params.n, dtype)
    h_min = tf.cast(viscosity_params.h_min, dtype)
    eps_dot_regu = tf.cast(viscosity_params.eps_dot_regu, dtype)
    eps_dot_min = tf.cast(viscosity_params.eps_dot_min, dtype)
    eps_dot_max = tf.cast(viscosity_params.eps_dot_max, dtype)

    return _cost(
        U,
        V,
        h,
        s,
        A,
        dx,
        n,
        h_min,
        eps_dot_regu,
        eps_dot_min,
        eps_dot_max,
        discr_h,
        w_v,
        zeta_v,
        V_q,
        V_q_grad,
    )


@tf.function()
def _cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    A: tf.Tensor,
    dx: tf.Tensor,
    n: tf.Tensor,
    h_min: tf.Tensor,
    eps_dot_regu: tf.Tensor,
    eps_dot_min: tf.Tensor,
    eps_dot_max: tf.Tensor,
    discr_h: HorizontalDiscr,
    w_v: tf.Tensor,
    zeta_v: tf.Tensor,
    V_q: tf.Tensor,
    V_q_grad: tf.Tensor,
) -> tf.Tensor:
    """
    Compute the viscous energy cost term.

    Calculates the viscous dissipation: h ∫ B ε̇^(1+1/n) / (1+1/n) dz

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
    A : tf.Tensor
        Arrhenius factor (MPa^-n year^-1)
    dx : tf.Tensor
        Grid spacing (m)
    n : tf.Tensor
        Glen's flow law exponent (-)
    h_min : tf.Tensor
        Minimum ice thickness threshold (m)
    eps_dot_regu : tf.Tensor
        Strain rate regularization (year^-1)
    eps_dot_min : tf.Tensor
        Minimum strain rate (year^-1)
    eps_dot_max : tf.Tensor
        Maximum strain rate (year^-1)
    discr_h : HorizontalDiscr
        Horizontal discretization class (-)
    w_v : tf.Tensor
        Weights for vertical integration (-)
    zeta_v : tf.Tensor
        Vertical quadrature points in [0,1] (-)
    V_q : tf.Tensor
        Vertical quadrature matrix: dofs -> quads (-)
    V_q_grad : tf.Tensor
        Vertical gradient matrix: dofs -> grad at quads (-)

    Returns
    -------
    tf.Tensor
        Viscous energy cost in MPa m/year
    """

    # Ice stiffness parameter
    B = 2.0 * tf.pow(A, -1.0 / n)

    # Effective exponent
    p = 1.0 + 1.0 / n

    # Fields at quadrature points
    h_h = discr_h.interp_h(h)  # -> (batch, Nq_h, Ny-1, Nx-1)
    B_h = discr_h.interp_h(B)  # -> (batch, Nq_h, Ny-1, Nx-1)
    u_h = discr_h.interp_h(U)  # -> (batch, Nq_h, Nz, Ny-1, Nx-1)
    v_h = discr_h.interp_h(V)  # -> (batch, Nq_h, Nz, Ny-1, Nx-1)

    # Evalue at horizontal quad points -> (batch, Nq_h, Nz, Ny-1, Nx-1)
    dudx_h, dudy_h = discr_h.grad_h(U, dx)
    dvdx_h, dvdy_h = discr_h.grad_h(V, dx)

    # Evaluate at vertical quad points -> (batch, Nq_h, Nq_v, Ny-1, Nx-1)
    dudx_hv = tf.einsum("vz,bhzyx->bhvyx", V_q, dudx_h)
    dudy_hv = tf.einsum("vz,bhzyx->bhvyx", V_q, dudy_h)
    dvdx_hv = tf.einsum("vz,bhzyx->bhvyx", V_q, dvdx_h)
    dvdy_hv = tf.einsum("vz,bhzyx->bhvyx", V_q, dvdy_h)
    dudz_hv = tf.einsum("vz,bhzyx->bhvyx", V_q_grad, u_h)
    dvdz_hv = tf.einsum("vz,bhzyx->bhvyx", V_q_grad, v_h)

    dudz_hv = dudz_hv / tf.maximum(h_h, h_min)[:, :, tf.newaxis, :]
    dvdz_hv = dvdz_hv / tf.maximum(h_h, h_min)[:, :, tf.newaxis, :]
    # dudz_q, dvdz_q = dampen_eps_dot_z_floating(dudz_q, dvdz_q, C)

    # Correct for terrain-following coordinates
    l = s - h
    dldx_h, dldy_h = discr_h.grad_h(l, dx)
    dsdx_h, dsdy_h = discr_h.grad_h(s, dx)

    dudx_hv, dudy_hv, dvdx_hv, dvdy_hv = correct_grad_zeta_to_z(
        dudx_hv,
        dudy_hv,
        dvdx_hv,
        dvdy_hv,
        dudz_hv,
        dvdz_hv,
        dldx_h[:, :, tf.newaxis, :, :],
        dldy_h[:, :, tf.newaxis, :, :],
        dsdx_h[:, :, tf.newaxis, :, :],
        dsdy_h[:, :, tf.newaxis, :, :],
        zeta_v[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis],
    )

    # Compute strain rate
    eps_dot2_hv = compute_eps_dot2(dudx_hv, dvdx_hv, dudy_hv, dvdy_hv, dudz_hv, dvdz_hv)

    eps_dot2_min = tf.pow(eps_dot_min, 2.0)
    eps_dot2_max = tf.pow(eps_dot_max, 2.0)
    eps_dot2_hv = tf.clip_by_value(eps_dot2_hv, eps_dot2_min, eps_dot2_max)

    # Compute viscous contribution
    eps_dot2_regu = tf.pow(eps_dot_regu, 2.0)
    exponent = (p - 2.0) / 2.0
    visc_term_hv = tf.pow(eps_dot2_hv + eps_dot2_regu, exponent) * eps_dot2_hv / p

    # h * ∫ [B * eps_dot^(1+1/n) / (1+1/n)] dz
    w_h = discr_h.w_h
    w_h = w_h[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis]
    w_v = w_v[tf.newaxis, tf.newaxis, :, tf.newaxis, tf.newaxis]
    B_h = B_h[:, :, tf.newaxis, :, :]
    h_h = h_h[:, :, tf.newaxis, :, :]
    return tf.reduce_sum(h_h * B_h * visc_term_hv * w_h * w_v, axis=[1, 2])
