#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Horizontally uniform, piecewise-linear ocean profiles T(z) and S(z).

``cfg.processes.ocean.profile.array`` lists rows ``[z, temp, salinity]``
after a header row, with ``z`` the depth relative to the water level (m,
negative below it). Values are interpolated linearly in depth and held
constant above the shallowest and below the deepest row, e.g. the ISOMIP+
COLD and WARM profiles of Asay-Davis et al. (2016).
"""

from typing import Tuple

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.utils.math.interp1d_tf import interp1d_tf


def initialize(cfg: DictConfig, state: State) -> None:
    rows = np.array(cfg.processes.ocean.profile.array[1:], dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] != 3:
        raise ValueError(
            "cfg.processes.ocean.profile.array needs a header row followed by "
            "at least one row [z, temp, salinity]."
        )
    state.ocean_profile = rows[np.argsort(rows[:, 0])]


def evaluate(
    cfg: DictConfig, state: State, z: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Temperature (°C) and salinity (g kg-1) at the depth ``z`` (m)."""
    profile = state.ocean_profile
    temp = interp1d_tf(profile[:, 0], profile[:, 1], z)
    salinity = interp1d_tf(profile[:, 0], profile[:, 2], z)
    return temp, salinity
