#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Ocean temperature and salinity from 2-D input fields.

The fields are the state variables named in ``cfg.processes.ocean.fields``,
typically loaded from the input file by ``load_ncdf``, and are taken to be
already representative of the water in contact with the ice (or of the sea
floor in open water). Exactly one of ``temp`` and ``thermal_forcing`` is
used. With a thermal forcing only (as provided by ISMIP6), the temperature
is rebuilt as ``TF + T_f(S, z)``, so the thermal forcing published by the
``ocean`` process reproduces the input. A missing salinity field is replaced
by the uniform ``salinity_ref``.
"""

from typing import Tuple

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from .seawater import freezing_temperature


def initialize(cfg: DictConfig, state: State) -> None:
    p = cfg.processes.ocean.fields
    if bool(p.temp) == bool(p.thermal_forcing):
        raise ValueError(
            "cfg.processes.ocean.fields: set exactly one of 'temp' and "
            "'thermal_forcing' to the name of a state variable."
        )
    for name in (p.temp, p.thermal_forcing, p.salinity):
        if name and not hasattr(state, name):
            raise ValueError(
                f"The ocean method 'fields' reads the state variable {name!r}, "
                "which does not exist; provide it with an input module "
                "(e.g. as a variable of the load_ncdf input file)."
            )


def evaluate(
    cfg: DictConfig, state: State, z: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Temperature (°C) and salinity (g kg-1) at the depth ``z`` (m)."""
    p = cfg.processes.ocean.fields
    if p.salinity:
        salinity = tf.cast(getattr(state, p.salinity), z.dtype)
    else:
        salinity = tf.fill(tf.shape(z), tf.cast(p.salinity_ref, z.dtype))
    if p.thermal_forcing:
        thermal_forcing = tf.cast(getattr(state, p.thermal_forcing), z.dtype)
        temp = thermal_forcing + freezing_temperature(cfg, salinity, z)
    else:
        temp = tf.cast(getattr(state, p.temp), z.dtype)
    return temp, salinity
