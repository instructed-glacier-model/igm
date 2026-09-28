#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""PICOP: PICO box properties feeding a buoyant-plume melt (Pelle et al., 2019).

The ambient temperature and salinity T_a, S_a are those of the node's PICO
box, with T_a at least the freezing point at sea level (Pelle et al., 2019);
the melt follows the plume parametrisation of Lazeroms et al. (2018),

    dT    = T_a - T_f(S_a, z_gl)
    Gamma = Cd_Gamma_T (gamma1 + gamma2 (dT / lambda3) E / (Cd_Gamma_TS0 + E))
    g     = sqrt(sin a / (Cd + E)) sqrt(Gamma / (Gamma + E)) E / (Gamma + E)
    l     = (dT / lambda3) (x0 Gamma + E) / (x0 (Gamma + E))
    m     = M0 P((z_d - z_gl) / l) g dT^2 (rho_w / rho_i)

with E = E0 sin a, a the basal slope angle, z_gl the depth of the grounding
line feeding the node and P the polynomial fit of the dimensionless melt
curve; P and the constants are those of Lazeroms et al. (2018). The last
factor converts the melt-water flux to ice, as in the implementation of
Burgard et al. (2022) (multimelt); Kori-ULB omits it. Melt is zero where the
plume has no thermal driving (dT <= 0), no source below sea level, or a flat
base.
"""

from typing import NamedTuple, Tuple

import tensorflow as tf
from numpy.polynomial import Polynomial
from omegaconf import DictConfig

from igm.common import State
from igm.processes.ocean.seawater import freezing_coefficients, freezing_point

from ...geometry import Geometry, basal_slope, grounding_line_depth
from ...utils import densities, require_process
from ..pico import pico

POLYNOMIAL = (
    0.1371330075095435,
    5.527656234709359e1,
    -8.951812433987858e2,
    8.927093637594877e3,
    -5.563863123811898e4,
    2.218596970948727e5,
    -5.820015295669482e5,
    1.015475347943186e6,
    -1.166290429178556e6,
    8.466870335320488e5,
    -3.520598035764990e5,
    6.387953795485420e4,
)

# The same polynomial in t = 2 x - 1, well conditioned on [-1, 1]: in x, the
# large alternating coefficients lose several percent in float32 near x = 1.
_SHIFTED = tuple(Polynomial(POLYNOMIAL)(Polynomial([0.5, 0.5])).coef)


def initialize(cfg: DictConfig, state: State) -> None:
    pico.initialize(cfg, state)
    require_process(cfg, "iceflow", "picop")


def dimensionless_melt(x: tf.Tensor) -> tf.Tensor:
    """Polynomial P(x) of Lazeroms et al. (2018), by Horner's rule in 2x - 1."""
    t = 2.0 * x - 1.0
    value = tf.zeros_like(x) + _SHIFTED[-1]
    for coefficient in reversed(_SHIFTED[:-1]):
        value = value * t + coefficient
    return value


def melt_rate(cfg: DictConfig, state: State, geom: Geometry) -> tf.Tensor:
    boxes = pico.solve_boxes(cfg, state, geom)
    pico.publish(state, boxes)
    z_gl = grounding_line_depth(cfg, state, geom)
    sin_a = basal_slope(geom.draft, geom.shelf, state.dx)
    melt = plume_melt(cfg, boxes.box_temp, boxes.box_salinity, geom.draft, z_gl, sin_a)
    return tf.where(geom.shelf, melt, 0.0)


class Constants(NamedTuple):
    """Plume parameters, as static values of the compiled melt formula."""

    M0: float
    gamma1: float
    gamma2: float
    Cd_Gamma_T: float
    Cd_Gamma_TS0: float
    E0: float
    Cd: float
    x0: float
    freezing: Tuple[float, float, float]  # (lambda1, lambda2, lambda3)
    to_ice: float  # rho_w / rho_i


def constants(cfg: DictConfig) -> Constants:
    p = cfg.processes.bmb.picop
    rho_i, rho_w = densities(cfg, "picop")
    return Constants(
        *(float(p[k]) for k in Constants._fields[:8]),
        freezing=freezing_coefficients(cfg),
        to_ice=rho_w / rho_i,
    )


def plume_melt(
    cfg: DictConfig,
    temp: tf.Tensor,
    salinity: tf.Tensor,
    draft: tf.Tensor,
    z_gl: tf.Tensor,
    sin_a: tf.Tensor,
) -> tf.Tensor:
    """Lazeroms et al. (2018) melt (m ice eq. yr-1) from ambient T and S."""
    return _plume_melt(temp, salinity, draft, z_gl, sin_a, constants(cfg))


@tf.function(autograph=False, jit_compile=True)
def _plume_melt(
    temp: tf.Tensor,
    salinity: tf.Tensor,
    draft: tf.Tensor,
    z_gl: tf.Tensor,
    sin_a: tf.Tensor,
    c: Constants,
) -> tf.Tensor:
    lambda3 = c.freezing[2]
    temp = tf.maximum(temp, freezing_point(salinity, 0.0, c.freezing))
    driving = temp - freezing_point(salinity, z_gl, c.freezing)
    active = (driving > 0.0) & (z_gl < 0.0) & (sin_a > 0.0)

    # Safe values off the active nodes keep every expression finite there.
    driving = tf.where(active, driving, 1.0)
    sin_a = tf.where(active, sin_a, 1.0)
    entrainment = c.E0 * sin_a
    exchange = c.Cd_Gamma_T * (
        c.gamma1
        + c.gamma2 * driving / lambda3 * entrainment / (c.Cd_Gamma_TS0 + entrainment)
    )
    velocity = (
        tf.sqrt(sin_a / (c.Cd + entrainment))
        * tf.sqrt(exchange / (exchange + entrainment))
        * entrainment
        / (exchange + entrainment)
    )
    length = (
        driving
        / lambda3
        * (c.x0 * exchange + entrainment)
        / (c.x0 * (exchange + entrainment))
    )
    x = tf.clip_by_value((draft - z_gl) / length, 0.0, 1.0)
    melt = c.M0 * dimensionless_melt(x) * velocity * driving**2 * c.to_ice
    return tf.where(active, melt, 0.0)
