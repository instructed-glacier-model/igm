#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
ocean
=====

Ocean properties in contact with the ice, consumed by the ``bmb`` process.
Each update evaluates the selected method at the depth

    z = min(lsurf - water_level, 0)    under ice (the ice base),
    z = min(topg  - water_level, 0)    in open water (the sea floor),

adds the uniform temperature anomaly delta_T(t) of ``anomaly_array`` and
publishes

    state.ocean_temp              T_o                 (°C)
    state.ocean_salinity          S_o                 (g kg-1)
    state.ocean_thermal_forcing   T_o - T_f(S_o, z)   (K)

where T_f is the linearised freezing point of :mod:`.seawater`. Methods
(``cfg.processes.ocean.method``):

    profile   horizontally uniform, piecewise-linear T(z) and S(z)
    fields    2-D fields provided by an input module

Without an ocean (the "no ocean" water level of ``thk.masks``) the depth is
clipped at 0 and the fields stay finite; no melt law uses them then.
"""

from types import ModuleType

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.math.interp1d_tf import interp1d_tf

from . import fields, profile
from .seawater import freezing_temperature

OceanMethods = {"fields": fields, "profile": profile}


def get_method(cfg: DictConfig) -> ModuleType:
    """Return the module of the configured ocean method."""
    name = cfg.processes.ocean.method.lower()
    if name not in OceanMethods:
        raise ValueError(
            f"cfg.processes.ocean.method = {name!r} is not available; "
            f"available methods: {', '.join(sorted(OceanMethods))}."
        )
    return OceanMethods[name]


def initialize(cfg: DictConfig, state: State) -> None:
    get_method(cfg).initialize(cfg, state)

    anomaly = np.array(cfg.processes.ocean.anomaly_array[1:], dtype=np.float32)
    if anomaly.ndim != 2 or anomaly.shape[0] < 1 or anomaly.shape[1] != 2:
        raise ValueError(
            "cfg.processes.ocean.anomaly_array needs a header row followed by "
            "at least one row [time, delta_temp]."
        )
    state.ocean_anomaly = anomaly[np.argsort(anomaly[:, 0])]

    # Filled by the first update, once time and the ice surfaces exist.
    for name in ("ocean_temp", "ocean_salinity", "ocean_thermal_forcing"):
        setattr(state, name, tf.zeros_like(state.thk))


def update(cfg: DictConfig, state: State) -> None:
    thk = state.thk
    z = ocean_depth(thk, state.lsurf, state.topg, state.water_level)
    temp, salinity = get_method(cfg).evaluate(cfg, state, z)
    anomaly = state.ocean_anomaly
    temp = temp + interp1d_tf(anomaly[:, 0], anomaly[:, 1], state.t)

    state.ocean_temp = temp
    state.ocean_salinity = salinity
    state.ocean_thermal_forcing = temp - freezing_temperature(cfg, salinity, z)


def finalize(cfg: DictConfig, state: State) -> None:
    pass


def ocean_depth(
    thk: tf.Tensor, lsurf: tf.Tensor, topg: tf.Tensor, water_level: tf.Tensor
) -> tf.Tensor:
    """Depth of the ice base, or of the sea floor where there is no ice (m)."""
    return tf.minimum(tf.where(thk > 0.0, lsurf, topg) - water_level, 0.0)
