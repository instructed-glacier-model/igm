#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict, Tuple
from omegaconf import DictConfig

from .energy import EnergyComponent
from igm.processes.iceflow.horizontal import HorizontalDiscr
from igm.processes.iceflow.vertical import VerticalDiscr


class FloatingParams(tf.experimental.ExtensionType):
    """Parameters for floating ice-shelf calving front."""

    rho: float
    rho_water: float
    g: float
    cf_eswn: Tuple[str, ...]


class FloatingComponent(EnergyComponent):
    """Energy component for floating ice-shelf calving front."""

    name = "floating"

    def __init__(self, params: FloatingParams) -> None:
        """Initialize floating component with parameters."""
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        discr_h: HorizontalDiscr,
        discr_v: VerticalDiscr,
    ) -> tf.Tensor:
        """Compute calving front energy cost."""
        return cost_floating(U, V, fieldin, discr_h, discr_v, self.params)


def get_floating_params_args(cfg: DictConfig) -> Dict[str, Any]:
    """Extract floating ice parameters from configuration."""

    cfg_physics = cfg.processes.iceflow.physics

    return {
        "cf_eswn": cfg_physics.cf_eswn,
        "rho": cfg_physics.ice_density,
        "rho_water": cfg_physics.water_density,
        "g": cfg_physics.gravity_cst,
    }


def cost_floating(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    floating_params: FloatingParams,
) -> tf.Tensor:
    """Compute calving front energy cost from field inputs."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    wl = fieldin["water_level"]
    dx = fieldin["dX"]

    V_q = discr_v.V_q
    w_v = discr_v.w

    dtype = U.dtype
    rho = tf.cast(floating_params.rho, dtype)
    rho_w = tf.cast(floating_params.rho_water, dtype)
    g = tf.cast(floating_params.g, dtype)
    cf_eswn = floating_params.cf_eswn

    return _cost(U, V, h, s, wl, dx, rho, rho_w, g, cf_eswn, discr_h, V_q, w_v)


@tf.function()
def _cost(
    U: tf.Tensor,
    V: tf.Tensor,
    h: tf.Tensor,
    s: tf.Tensor,
    wl: tf.Tensor,
    dx: tf.Tensor,
    rho: tf.Tensor,
    rho_w: tf.Tensor,
    g: tf.Tensor,
    cf_eswn: Tuple[str, ...],
    discr_h: HorizontalDiscr,
    V_q: tf.Tensor,
    w_v: tf.Tensor,
) -> tf.Tensor:
    """
    Compute the calving front energy cost term.

    Calculates the work done at the calving front: ∫_Γ P · u·n ds, expressed
    as an energy density per cell so that it can be summed with volume
    contributions. P = 0.5 * ρ * g * h² * (1 - ρ_w/ρ * r²) accounts for the
    stress balance at the calving front.

    Parameters
    ----------
    U : tf.Tensor
        Horizontal velocity along x axis (m/year), shape (batch, Nz, Ny, Nx)
    V : tf.Tensor
        Horizontal velocity along y axis (m/year), shape (batch, Nz, Ny, Nx)
    h : tf.Tensor
        Ice thickness (m), shape (batch, Ny, Nx)
    s : tf.Tensor
        Upper-surface elevation (m), shape (batch, Ny, Nx)
    dx : tf.Tensor
        Grid spacing (m)
    rho : tf.Tensor
        Ice density (kg m^-3)
    rho_w : tf.Tensor
        Water density (kg m^-3)
    g : tf.Tensor
        Gravity acceleration (m s^-2)
    cf_eswn : tuple
        Calving front boundaries: ("E", "S", "W", "N")
    discr_h : HorizontalDiscr
        Horizontal discretization class (-)
    V_q : tf.Tensor
        Quadrature matrix: dofs -> quads (-)
    w_v : tf.Tensor
        Weights for vertical integration (-)

    Returns
    -------
    tf.Tensor
        Calving front energy cost with shape (batch, Ny-1, Nx-1) in MPa m/year
    """

    dtype = U.dtype

    # Lower surface elevation
    l = s - h

    # Pad to detect calving front (pad with 1.0 if edge NOT in cf_eswn)
    pad_value = lambda edge: 1.0 if edge not in cf_eswn else 0.0

    # Pad h: [[batch], [y], [x]]
    h_ext = tf.pad(h, [[0, 0], [1, 0], [0, 0]], constant_values=pad_value("S"))
    h_ext = tf.pad(h_ext, [[0, 0], [0, 1], [0, 0]], constant_values=pad_value("N"))
    h_ext = tf.pad(h_ext, [[0, 0], [0, 0], [1, 0]], constant_values=pad_value("W"))
    h_ext = tf.pad(h_ext, [[0, 0], [0, 0], [0, 1]], constant_values=pad_value("E"))

    # Pad l
    l_ext = tf.pad(l, [[0, 0], [1, 0], [0, 0]], constant_values=pad_value("S"))
    l_ext = tf.pad(l_ext, [[0, 0], [0, 1], [0, 0]], constant_values=pad_value("N"))
    l_ext = tf.pad(l_ext, [[0, 0], [0, 0], [1, 0]], constant_values=pad_value("W"))
    l_ext = tf.pad(l_ext, [[0, 0], [0, 0], [0, 1]], constant_values=pad_value("E"))

    # Pad water_level (same convention as l: at non-CF boundaries the value
    # is irrelevant because h_ext kills the test; at CF boundaries we pad
    # with 0 so the legacy "ocean at sea level" behaviour is recovered if
    # the user's domain edge is in fact open ocean).
    wl_ext = tf.pad(wl, [[0, 0], [1, 0], [0, 0]], constant_values=pad_value("S"))
    wl_ext = tf.pad(wl_ext, [[0, 0], [0, 1], [0, 0]], constant_values=pad_value("N"))
    wl_ext = tf.pad(wl_ext, [[0, 0], [0, 0], [1, 0]], constant_values=pad_value("W"))
    wl_ext = tf.pad(wl_ext, [[0, 0], [0, 0], [0, 1]], constant_values=pad_value("E"))

    # Detect calving front: current cell has ice (h>0) and the neighbour
    # is a water cell (h_n=0 and its lower surface is at or below the
    # local water level w_n).
    is_ice = h > 0.0
    is_water = lambda h_n, l_n, w_n: (h_n == 0.0) & (l_n <= w_n)
    is_cf = lambda h_n, l_n, w_n: tf.cast(is_ice & is_water(h_n, l_n, w_n), dtype)

    CF_W = is_cf(h_ext[:, 1:-1, :-2], l_ext[:, 1:-1, :-2], wl_ext[:, 1:-1, :-2])
    CF_E = is_cf(h_ext[:, 1:-1, 2:],  l_ext[:, 1:-1, 2:],  wl_ext[:, 1:-1, 2:])
    CF_S = is_cf(h_ext[:, :-2, 1:-1], l_ext[:, :-2, 1:-1], wl_ext[:, :-2, 1:-1])
    CF_N = is_cf(h_ext[:, 2:, 1:-1],  l_ext[:, 2:, 1:-1],  wl_ext[:, 2:, 1:-1])

    # Depth-integrated velocity using vertical quadrature
    # U, V: (batch, Nz, Ny, Nx) -> (batch, Nq_v, Ny, Nx)
    u_q = tf.einsum("vz,bzyx->bvyx", V_q, U)
    v_q = tf.einsum("vz,bzyx->bvyx", V_q, V)
    w_q = w_v[tf.newaxis, :, tf.newaxis, tf.newaxis]
    U_int = tf.reduce_sum(u_q * w_q, axis=1)  # -> (batch, Ny, Nx)
    V_int = tf.reduce_sum(v_q * w_q, axis=1)  # -> (batch, Ny, Nx)

    # Pre-factor: P = 0.5 * ρ * g * h² * (1 - ρ_w/ρ * r²)
    # r = D/h, where D = max(w - l, 0) is the water depth at the cliff
    # (w = local water level, l = lower surface of the ice).
    r = tf.maximum((wl - l) / tf.maximum(h, 1.0), 0.0)
    P = 0.5 * g * rho * h * h * (1.0 - (rho_w / rho) * r * r)

    # Boundary energy density on nodal grid: (batch, Ny, Nx)
    # Line integral contribution P * u·n * dx divided by cell area dx²
    # gives energy density P * u·n / dx with units MPa m/year
    C_float = 1e-6 * P * (U_int * (CF_E - CF_W) + V_int * (CF_N - CF_S)) / dx

    # Map to staggered grid via horizontal quadrature -> (batch, Nq_h, Ny-1, Nx-1)
    C_h = discr_h.interp_h(C_float)

    # Integrate over horizontal quad points -> (batch, Ny-1, Nx-1)
    w_h = discr_h.w_h[tf.newaxis, :, tf.newaxis, tf.newaxis]
    return -tf.reduce_sum(C_h * w_h, axis=1)
