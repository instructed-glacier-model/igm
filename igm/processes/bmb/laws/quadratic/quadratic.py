#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Quadratic thermal-forcing melt (Favier et al., 2019; Jourdain et al., 2020).

    m = gamma0 (rho_w c_o / (rho_i L))^2 (TF + dT) |<TF> + dT|

with TF the ocean thermal forcing at the ice draft (``ocean`` process) and
<TF> either TF itself (``averaging: local``), its mean over each ice shelf
(``shelf``), or its mean over the ice shelves of each basin of the integer
field ``state.basins`` (``basin``, the ISMIP6 non-local form). The sign of TF
is kept, so a negative thermal forcing gives refreezing unless the ``bmb``
process clips it; with ``allow_refreezing: false`` the local law is
gamma0 (...)^2 (TF)_+^2.

``gamma0`` is a heat-exchange velocity in m yr-1. Jourdain et al. (2020)
express the resulting melt in m w.e. yr-1: with ``water_equivalent`` it is
converted to ice with rho_fw / rho_i. The default gamma0, 14477 m yr-1, is
the value used by Kori-ULB for the ISMIP6 non-local MeanAnt calibration of
Jourdain et al. (2020).
"""

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ...geometry import Geometry, shelf_labels
from ...utils import densities, require_process, segment_means

AVERAGING = ("basin", "local", "shelf")
FRESH_WATER_DENSITY = 1000.0  # kg m-3


def initialize(cfg: DictConfig, state: State) -> None:
    p = cfg.processes.bmb.quadratic
    require_process(cfg, "ocean", "quadratic")
    densities(cfg, "quadratic")
    if p.averaging not in AVERAGING:
        raise ValueError(
            f"cfg.processes.bmb.quadratic.averaging = {p.averaging!r} is not "
            f"one of {', '.join(AVERAGING)}."
        )
    if p.averaging == "basin" and not hasattr(state, "basins"):
        raise ValueError(
            "The quadratic melt with averaging 'basin' reads the integer "
            "field 'basins'; provide it with an input module."
        )


def melt_rate(cfg: DictConfig, state: State, geom: Geometry) -> tf.Tensor:
    p = cfg.processes.bmb.quadratic
    rho_i, rho_w = densities(cfg, "quadratic")
    physics = cfg.processes.bmb.physics
    factor = p.gamma0 * (rho_w * physics.c_ocean / (rho_i * physics.L_ice)) ** 2
    if p.water_equivalent:
        factor *= FRESH_WATER_DENSITY / rho_i

    if p.averaging == "shelf":
        ids = shelf_labels(geom)
    elif p.averaging == "basin":
        ids = tf.cast(tf.round(state.basins), tf.int32)
    else:
        ids = tf.zeros_like(geom.shelf, tf.int32)
    return _quadratic(
        tf.cast(state.ocean_thermal_forcing, geom.draft.dtype),
        ids,
        geom.shelf,
        float(factor),
        float(p.delta_T),
        p.averaging == "local",
    )


@tf.function(autograph=False, jit_compile=True)
def _quadratic(
    thermal_forcing: tf.Tensor,
    ids: tf.Tensor,
    shelf: tf.Tensor,
    factor: float,
    delta_T: float,
    local: bool,
) -> tf.Tensor:
    thermal_forcing = thermal_forcing + delta_T
    if local:
        mean = thermal_forcing
    else:
        num_segments = tf.size(ids) + 1
        ids = tf.clip_by_value(ids, 0, num_segments - 1)
        (mean,), _ = segment_means([thermal_forcing], ids, shelf, num_segments)
    return factor * thermal_forcing * tf.abs(mean)
