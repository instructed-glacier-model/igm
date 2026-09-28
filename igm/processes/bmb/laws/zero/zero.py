#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Zero sub-shelf melt.

Deliberately no ocean melt. With ``include_grounded_melt`` (the default) the
basal mass balance then reduces to minus the thermodynamic melt of the
``enthalpy`` process, ``bmb = -basal_melt_rate``: the method of a land-only
run (or of a marine run without ocean melt) whose basal melt should reach
the thickness equation.
"""

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ...geometry import Geometry


def melt_rate(cfg: DictConfig, state: State, geom: Geometry) -> tf.Tensor:
    return tf.zeros_like(geom.thk)
