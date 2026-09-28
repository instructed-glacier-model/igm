#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
calving_rate
============

Lateral ablation rate at a marine ice front, consumed by the calving front
of the ``thk`` process (``cfg.processes.thk.front``). The front moves, normal
to itself, at

    u_cf = u . n - c - m_cf,

with ``c`` the calving rate and ``m_cf`` the frontal melt rate (both m/yr,
>= 0). The calving law ``cfg.processes.calving_rate.law`` is one of

    zero          c = 0
    constant      c = value
    water_depth   c = k max(z_wl - z_b, 0)                    Brown et al. (1982)
    eigen         c = K max(e1, 0) max(e2, 0), floating ice  Levermann et al. (2012)
    von_mises     c = |u| sigma / sigma_max                   Morlighem et al. (2016)
    ice_speed     c = max(f |u| - W, 0)                        CalvingMIP-style

and the frontal melt ``frontal_melt.method`` is ``none``, ``constant`` or
``field`` (a state variable). Laws that prescribe the front position rather
than a rate (a minimum front thickness, a fixed front) are options of the
front itself (``thk.front.min_thickness``, ``thk.front.fixed``).

The law is evaluated on the ice nodes that carry velocity, and carried to
the other cells of the front band (within ``band`` cells of the front, on
both sides) by the mean over their neighbours, as PISM does for its
Hayhurst law; the principal strain rates use one-sided differences that do
not cross the front (second order where two nodes lie behind it). The
``ice_speed`` law, which prescribes the front velocity relative to the ice
at the front, is instead evaluated on the whole band with the ice speed at
the front (``geometry.front_speed``). The rate is zero outside the band, on land, and at ice
fronts not facing the open ocean (``ocean_connected_only``); it is capped at
``max_rate``. Published fields, in m/yr:

    state.calving_rate        c
    state.frontal_melt_rate   m_cf

The time step includes ``c + m_cf`` in its CFL condition. Run this process
after ``iceflow`` and before ``thk``, e.g. ``[iceflow, calving_rate, time, thk]``.
"""

from types import ModuleType

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.math.neighbours import any_neighbour, neighbour_mean

from .geometry import FrontGeometry, front_geometry
from .laws import get_calving_law

FRONTAL_MELT_METHODS = ("constant", "field", "none")


def get_active_submodule(cfg: DictConfig) -> ModuleType:
    """The calving law, whose metadata lists the state variables it reads."""
    return get_calving_law(cfg)[1]


def initialize(cfg: DictConfig, state: State) -> None:
    p = cfg.processes.calving_rate
    _check_removed_keys(p)
    name, _ = get_calving_law(cfg)
    if "thk" not in cfg.processes:
        raise ValueError(
            "The calving_rate process feeds the front of the 'thk' process."
        )
    method = str(p.frontal_melt.method).strip().lower()
    if method not in FRONTAL_MELT_METHODS:
        raise ValueError(
            "cfg.processes.calving_rate.frontal_melt.method must be one of "
            f"{', '.join(FRONTAL_MELT_METHODS)}; got {method!r}."
        )
    if int(p.band) < 1:
        raise ValueError("cfg.processes.calving_rate.band must be >= 1.")
    if not float(p.max_rate) > 0.0:
        raise ValueError("cfg.processes.calving_rate.max_rate must be positive.")
    _check_law_parameters(name, p)
    _check_band(cfg, p)
    state.calving_rate = tf.zeros_like(state.thk)
    state.frontal_melt_rate = tf.zeros_like(state.thk)
    # The velocity-based laws need a first ice-flow solve.
    if hasattr(state, "ubar") and hasattr(state, "vbar") and hasattr(state, "usurf"):
        update(cfg, state)


def _check_removed_keys(p: DictConfig) -> None:
    """The former configuration keys fail loudly (thk/DESIGN.md)."""
    removed = {
        "Hcr": "cfg.processes.thk.front.min_thickness",
        "K2": "cfg.processes.calving_rate.eigen.K",
        "c_max": "cfg.processes.calving_rate.max_rate",
    }
    for key, target in removed.items():
        if key in p:
            raise ValueError(
                f"cfg.processes.calving_rate.{key} was removed; use {target}."
            )


def _check_law_parameters(name: str, p: DictConfig) -> None:
    """Reject parameter values the active law cannot use."""
    if name == "von_mises" and not (
        float(p.von_mises.sigma_max_floating) > 0.0
        and float(p.von_mises.sigma_max_grounded) > 0.0
    ):
        raise ValueError(
            "cfg.processes.calving_rate.von_mises.sigma_max_floating and "
            "sigma_max_grounded must be positive."
        )
    if name == "eigen" and float(p.eigen.K) < 0.0:
        raise ValueError("cfg.processes.calving_rate.eigen.K must be >= 0.")
    if name == "water_depth" and float(p.water_depth.k) < 0.0:
        raise ValueError("cfg.processes.calving_rate.water_depth.k must be >= 0.")
    if name == "constant" and float(p.constant.value) < 0.0:
        raise ValueError("cfg.processes.calving_rate.constant.value must be >= 0.")
    if name == "ice_speed" and float(p.ice_speed.factor) < 0.0:
        raise ValueError("cfg.processes.calving_rate.ice_speed.factor must be >= 0.")


def _check_band(cfg: DictConfig, p: DictConfig) -> None:
    """The level set reads the rate on its own band: ours must cover it."""
    front = cfg.processes.thk.get("front", None) or {}
    if str(front.get("method", "none") or "none").strip().lower() != "level_set":
        return
    level_set_band = int((front.get("level_set", None) or {}).get("band", 3))
    if int(p.band) < level_set_band:
        raise ValueError(
            "cfg.processes.calving_rate.band must be >= "
            f"cfg.processes.thk.front.level_set.band ({level_set_band})."
        )


def update(cfg: DictConfig, state: State) -> None:
    p = cfg.processes.calving_rate
    geom = front_geometry(cfg, state)
    _, law = get_calving_law(cfg)
    rate = law.calving_rate(cfg, state, geom)
    if law.ON_BAND:
        rate = tf.clip_by_value(rate, 0.0, float(p.max_rate))
        state.calving_rate = tf.where(geom.band, rate, tf.zeros_like(rate))
    else:
        state.calving_rate = spread(rate, geom, int(p.band), float(p.max_rate))
    state.frontal_melt_rate = frontal_melt(cfg, state, geom)


def finalize(cfg: DictConfig, state: State) -> None:
    pass


def spread(
    rate: tf.Tensor, geom: FrontGeometry, steps: int, max_rate: float
) -> tf.Tensor:
    """Carry a rate known on the supported nodes to the whole front band.

    Each step fills the cells next to a known cell with the mean over their
    known neighbours; the result is clipped to ``[0, max_rate]`` and zero
    outside the band.
    """
    known = geom.supported
    for _ in range(steps):
        new = ~known & any_neighbour(known)
        rate = tf.where(new, neighbour_mean(rate, known), rate)
        known = known | new
    rate = tf.clip_by_value(rate, 0.0, max_rate)
    return tf.where(geom.band & known, rate, tf.zeros_like(rate))


def frontal_melt(cfg: DictConfig, state: State, geom: FrontGeometry) -> tf.Tensor:
    """Frontal (submarine) melt rate m_cf (m/yr) in the front band."""
    p = cfg.processes.calving_rate.frontal_melt
    method = str(p.method).strip().lower()
    if method == "none":
        return tf.zeros_like(geom.thk)
    if method == "constant":
        melt = float(p.value) + tf.zeros_like(geom.thk)
    else:
        if not hasattr(state, p.field):
            raise ValueError(
                f"calving_rate.frontal_melt.field = {p.field!r} is not a state variable."
            )
        melt = tf.cast(getattr(state, p.field), geom.thk.dtype)
    return tf.where(geom.band, tf.maximum(melt, 0.0), tf.zeros_like(melt))
