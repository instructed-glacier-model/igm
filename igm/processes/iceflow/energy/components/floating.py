#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict, Tuple

import tensorflow as tf
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

    return _cost(
        U,
        V,
        h,
        s,
        wl,
        dx,
        rho,
        rho_w,
        g,
        cf_eswn,
        discr_h,
        V_q,
        w_v,
    )


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
    Compute the calving-front energy cost term.

    The face integral of hydrostatic pressure against normal velocity is
    expressed as an energy density on each active structured-grid cell.

    Parameters
    ----------
    U : tf.Tensor
        Horizontal x velocity (m/year), shape (batch, Nz, Ny, Nx).
    V : tf.Tensor
        Horizontal y velocity (m/year), shape (batch, Nz, Ny, Nx).
    h : tf.Tensor
        Ice thickness (m), shape (batch, Ny, Nx).
    s : tf.Tensor
        Upper-surface elevation (m), shape (batch, Ny, Nx).
    wl : tf.Tensor
        Water-level elevation (m), shape (batch, Ny, Nx).
    dx : tf.Tensor
        Grid spacing (m).
    rho : tf.Tensor
        Ice density (kg m^-3).
    rho_w : tf.Tensor
        Water density (kg m^-3).
    g : tf.Tensor
        Gravitational acceleration (m s^-2).
    cf_eswn : tuple
        Domain edges open to calving fronts: ("E", "S", "W", "N").
    discr_h : HorizontalDiscr
        Horizontal discretization.
    V_q : tf.Tensor
        Vertical quadrature interpolation matrix.
    w_v : tf.Tensor
        Vertical quadrature weights.

    Returns
    -------
    tf.Tensor
        Calving-front energy density with shape (batch, Ny-1, Nx-1),
        in MPa m/year.
    """

    dtype = U.dtype

    l = s - h

    # Depth-integrated velocity using vertical quadrature
    u_q = tf.einsum("vz,bzyx->bvyx", V_q, U)
    v_q = tf.einsum("vz,bzyx->bvyx", V_q, V)
    w_q = w_v[tf.newaxis, :, tf.newaxis, tf.newaxis]
    u_integrated = tf.reduce_sum(u_q * w_q, axis=1)
    v_integrated = tf.reduce_sum(v_q * w_q, axis=1)

    # Wet neighbours are cells that contain neither ice nor exposed land.
    def corners(values):
        return (
            values[:, :-1, :-1],
            values[:, :-1, 1:],
            values[:, 1:, :-1],
            values[:, 1:, 1:],
        )

    is_ice = h > 0.0
    i_sw, i_se, i_nw, i_ne = corners(is_ice)
    cell_ice = i_sw & i_se & i_nw & i_ne

    is_land = (h <= 0.0) & (l > wl)
    l_sw, l_se, l_nw, l_ne = corners(is_land)
    cell_land = l_sw | l_se | l_nw | l_ne

    wet = tf.cast(~cell_ice & ~cell_land, dtype)

    # Declared calving-front domain edges are open ocean; other edges are closed.
    def wet_edge(edge):
        return 1.0 if edge in cf_eswn else 0.0

    wet_p = tf.pad(wet, [[0, 0], [1, 0], [0, 0]], constant_values=wet_edge("S"))
    wet_p = tf.pad(wet_p, [[0, 0], [0, 1], [0, 0]], constant_values=wet_edge("N"))
    wet_p = tf.pad(wet_p, [[0, 0], [0, 0], [1, 0]], constant_values=wet_edge("W"))
    wet_p = tf.pad(wet_p, [[0, 0], [0, 0], [0, 1]], constant_values=wet_edge("E"))

    wet_east = wet_p[:, 1:-1, 2:]
    wet_west = wet_p[:, 1:-1, :-2]
    wet_north = wet_p[:, 2:, 1:-1]
    wet_south = wet_p[:, :-2, 1:-1]

    # Evaluate hydrostatic pressure from the two nodes of each face.
    def face(h_a, h_b, lower_a, lower_b, water_a, water_b):
        face_thk = 0.5 * (h_a + h_b)
        water_depth = tf.maximum(
            0.5 * (water_a + water_b) - 0.5 * (lower_a + lower_b), 0.0
        )
        return 0.5 * g * (
            rho * face_thk * face_thk - rho_w * water_depth * water_depth
        )

    h_sw, h_se, h_nw, h_ne = corners(h)
    lower_sw, lower_se, lower_nw, lower_ne = corners(l)
    water_sw, water_se, water_nw, water_ne = corners(wl)
    u_sw, u_se, u_nw, u_ne = corners(u_integrated)
    v_sw, v_se, v_nw, v_ne = corners(v_integrated)

    pressure_east = face(
        h_se, h_ne, lower_se, lower_ne, water_se, water_ne
    )
    pressure_west = face(
        h_sw, h_nw, lower_sw, lower_nw, water_sw, water_nw
    )
    pressure_north = face(
        h_nw, h_ne, lower_nw, lower_ne, water_nw, water_ne
    )
    pressure_south = face(
        h_sw, h_se, lower_sw, lower_se, water_sw, water_se
    )

    velocity_east = 0.5 * (u_se + u_ne)
    velocity_west = 0.5 * (u_sw + u_nw)
    velocity_north = 0.5 * (v_nw + v_ne)
    velocity_south = 0.5 * (v_sw + v_se)

    # Convert the face integral to the cell energy-density convention.
    dx_c = 0.5 * (dx[:, :-1, :-1] + dx[:, 1:, 1:]) if dx.shape.rank == 3 else dx

    front_energy = (
        -pressure_east * velocity_east * wet_east
        + pressure_west * velocity_west * wet_west
        - pressure_north * velocity_north * wet_north
        + pressure_south * velocity_south * wet_south
    ) / dx_c

    return 1e-6 * front_energy * tf.cast(cell_ice, dtype)
