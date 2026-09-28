#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Uniform calving rate: c = value (m/yr)."""

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ...geometry import FrontGeometry

#: The rate is evaluated on the supported ice nodes and carried to the
#: front band by the process (see ``laws/__init__.py``).
ON_BAND = False


def calving_rate(cfg: DictConfig, state: State, geom: FrontGeometry) -> tf.Tensor:
    value = float(cfg.processes.calving_rate.constant.value)
    return tf.where(
        geom.supported, value + tf.zeros_like(geom.thk), tf.zeros_like(geom.thk)
    )
