#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Calving rate tied to the ice speed: c = max(f |u| - W, 0).

With ``factor`` f = 1 the front moves at the prescribed velocity W relative
to the fixed grid wherever the ice flows normal to it (W = 0: a stationary
front), as in the CalvingMIP experiments; W > 0 is an advance. Mostly a
verification tool.

``|u|`` is the speed of the ice at the front (``FrontGeometry.front_speed``),
the speed with which the front moves, so the law is evaluated on the whole
front band (``ON_BAND``) instead of being carried from the last ice nodes:
on a spreading shelf the ice at the front is faster than at the last node
(by about ``du/dx dx``), which would otherwise bias the front velocity.
"""

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ...geometry import FrontGeometry

#: Evaluated on the front band, not carried from the supported nodes.
ON_BAND = True


def calving_rate(cfg: DictConfig, state: State, geom: FrontGeometry) -> tf.Tensor:
    p = cfg.processes.calving_rate.ice_speed
    rate = float(p.factor) * geom.front_speed - float(p.front_velocity)
    return tf.where(geom.band, tf.maximum(rate, 0.0), tf.zeros_like(geom.thk))
