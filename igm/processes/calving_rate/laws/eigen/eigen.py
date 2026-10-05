#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Eigencalving (Levermann et al., 2012): c = K max(e1, 0) max(e2, 0).

Levermann, A., Albrecht, T., Winkelmann, R., Martin, M. A., Haseloff, M.,
and Joughin, I.: Kinematic first-order calving law implies potential for
abrupt ice-shelf retreat, The Cryosphere, 6, 273-286, 2012. ``e1 >= e2`` are
the principal horizontal strain rates (1/yr) and ``K`` is in m yr. Only
floating ice calves; ice spreading in both directions is required.
"""

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ...geometry import FrontGeometry
from ...strain_rates import principal_strain_rates

#: The rate is evaluated on the supported ice nodes and carried to the
#: front band by the process (see ``laws/__init__.py``).
ON_BAND = False


def calving_rate(cfg: DictConfig, state: State, geom: FrontGeometry) -> tf.Tensor:
    K = float(cfg.processes.calving_rate.eigen.K)
    e1, e2 = principal_strain_rates(
        tf.cast(state.ubar, geom.thk.dtype),
        tf.cast(state.vbar, geom.thk.dtype),
        geom.supported,
        geom.dx,
    )
    rate = K * tf.maximum(e1, 0.0) * tf.maximum(e2, 0.0)
    return tf.where(geom.supported & geom.floating, rate, tf.zeros_like(rate))
