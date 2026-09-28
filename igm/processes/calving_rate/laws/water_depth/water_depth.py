#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Water-depth calving law (Brown et al., 1982): c = k max(z_wl - z_b, 0).

Brown, C. S., Meier, M. F., and Post, A.: Calving speed of Alaska tidewater
glaciers, with application to Columbia Glacier, U.S. Geological Survey
Professional Paper 1258-C, 1982. ``k`` is in 1/yr.
"""

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ...geometry import FrontGeometry

#: The rate is evaluated on the supported ice nodes and carried to the
#: front band by the process (see ``laws/__init__.py``).
ON_BAND = False


def calving_rate(cfg: DictConfig, state: State, geom: FrontGeometry) -> tf.Tensor:
    k = float(cfg.processes.calving_rate.water_depth.k)
    return tf.where(geom.supported, k * geom.water_depth, tf.zeros_like(geom.thk))
