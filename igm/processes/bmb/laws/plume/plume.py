#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Buoyant-plume melt of Lazeroms et al. (2019).

The ambient temperature and salinity T_a, S_a are the means of the ocean
fields at the base of each ice shelf, as in Kori-ULB and multimelt (Favier
et al. (2019) take the far-field temperature at the grounding-line depth
instead). With E = E0 sin a (a the basal slope angle) and z_gl the depth of
the grounding line feeding the node,

    dT    = T_a - T_f(S_a, z_gl)
    c1    = L alpha / (c_o Gamma beta S_a),   c_tau = -lambda1 alpha / (beta c1)
    X     = lambda3 (z_d - z_gl) / (dT (1 + C_eps (E / (Gamma + c_tau + E))^(3/4)))
    M_hat = (3 (1 - X)^(4/3) - 1) sqrt(1 - (1 - X)^(4/3)) / (2 sqrt(2))
    M     = sqrt(beta S_a g / (lambda3 (L / c_o)^3)) sqrt((1 - c1 Gamma) / (Cd + E))
            (Gamma E / (Gamma + c_tau + E))^(3/2) dT^2
    m     = M M_hat (rho_w / rho_i)

with X clipped to [0, 1] (beyond it, the refreezing of X = 1 is kept, as in
Kori-ULB and multimelt) and the constants of Lazeroms et al. (2019). The
last factor converts the melt-water flux to ice, as in the implementation of
Burgard et al. (2022) (multimelt); Kori-ULB omits it. Melt is zero where the
plume has no thermal driving (dT <= 0), no source below sea level, or a flat
base.
"""

import math
from typing import NamedTuple, Tuple

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.ocean.seawater import freezing_coefficients, freezing_point

from ...geometry import Geometry, basal_slope, grounding_line_depth, shelf_labels
from ...utils import SECONDS_PER_YEAR, densities, require_process, segment_means

GRAVITY = 9.81  # m s-2


def initialize(cfg: DictConfig, state: State) -> None:
    require_process(cfg, "ocean", "plume")
    require_process(cfg, "iceflow", "plume")


def melt_rate(cfg: DictConfig, state: State, geom: Geometry) -> tf.Tensor:
    labels = shelf_labels(geom)
    (temp, salinity), _ = segment_means(
        [state.ocean_temp, state.ocean_salinity],
        labels,
        geom.shelf,
        tf.size(labels) + 1,
    )
    z_gl = grounding_line_depth(cfg, state, geom)
    sin_a = basal_slope(geom.draft, geom.shelf, state.dx)
    melt = plume_melt(cfg, temp, salinity, geom.draft, z_gl, sin_a)
    return tf.where(geom.shelf, melt, 0.0)


class Constants(NamedTuple):
    """Plume parameters, as static values of the compiled melt formula."""

    gamma_T: float
    E0: float
    Cd: float
    C_eps: float
    alpha: float
    beta: float
    freezing: Tuple[float, float, float]  # (lambda1, lambda2, lambda3)
    latent: float  # L / c_o (K)
    to_ice: float  # rho_w / rho_i


def constants(cfg: DictConfig) -> Constants:
    p = cfg.processes.bmb.plume
    physics = cfg.processes.bmb.physics
    rho_i, rho_w = densities(cfg, "plume")
    return Constants(
        *(float(p[k]) for k in Constants._fields[:6]),
        freezing=freezing_coefficients(cfg),
        latent=float(physics.L_ice / physics.c_ocean),
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
    """Lazeroms et al. (2019) melt (m ice eq. yr-1) from ambient T and S."""
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
    lambda1, _, lambda3 = c.freezing
    driving = temp - freezing_point(salinity, z_gl, c.freezing)
    active = (driving > 0.0) & (z_gl < 0.0) & (sin_a > 0.0)

    # Safe values off the active nodes keep every expression finite there.
    driving = tf.where(active, driving, 1.0)
    salinity = tf.where(active, salinity, 35.0)
    sin_a = tf.where(active, sin_a, 1.0)
    entrainment = c.E0 * sin_a
    c_rho1 = c.latent * c.alpha / (c.gamma_T * c.beta * salinity)
    c_tau = -lambda1 * c.alpha / (c.beta * c_rho1)
    plume = c.gamma_T + c_tau + entrainment

    x = (
        lambda3
        * (draft - z_gl)
        / (driving * (1.0 + c.C_eps * (entrainment / plume) ** 0.75))
    )
    x = tf.clip_by_value(x, 0.0, 1.0)
    shape = (1.0 - x) ** (4.0 / 3.0)
    m_hat = (3.0 * shape - 1.0) * tf.sqrt(1.0 - shape) / (2.0 * math.sqrt(2.0))
    scale = (
        tf.sqrt(c.beta * salinity * GRAVITY / (lambda3 * c.latent**3))
        * tf.sqrt(tf.maximum(1.0 - c_rho1 * c.gamma_T, 0.0) / (c.Cd + entrainment))
        * (c.gamma_T * entrainment / plume) ** 1.5
        * driving**2
    )
    return tf.where(active, scale * m_hat * SECONDS_PER_YEAR * c.to_ice, 0.0)
