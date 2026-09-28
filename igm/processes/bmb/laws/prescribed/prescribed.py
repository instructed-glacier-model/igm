#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Prescribed sub-shelf melt (m ice eq. yr-1, positive for melt).

``cfg.processes.bmb.prescribed.form`` selects

    constant      m = value
    field         m = the state variable named by ``field``
    mismip_plus   m = omega tanh(H_c / H0) (z0 - z_d)_+     (MISMIP+ Ice1)

with z_d the ice draft and H_c = z_d - z_b the water-column thickness, both
relative to the water level (Asay-Davis et al., 2016).
"""

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from ...geometry import Geometry

FORMS = ("constant", "field", "mismip_plus")


def initialize(cfg: DictConfig, state: State) -> None:
    p = cfg.processes.bmb.prescribed
    if p.form not in FORMS:
        raise ValueError(
            f"cfg.processes.bmb.prescribed.form = {p.form!r} is not one of "
            f"{', '.join(FORMS)}."
        )
    if p.form == "field" and not hasattr(state, p.field):
        raise ValueError(
            f"The prescribed melt reads the state variable {p.field!r}, which "
            "does not exist; provide it with an input module."
        )


def melt_rate(cfg: DictConfig, state: State, geom: Geometry) -> tf.Tensor:
    p = cfg.processes.bmb.prescribed
    dtype = geom.thk.dtype
    if p.form == "constant":
        return tf.fill(tf.shape(geom.thk), tf.cast(p.value, dtype))
    if p.form == "field":
        return tf.cast(getattr(state, p.field), dtype)

    q = p.mismip_plus
    column = geom.draft - (state.topg - state.water_level)
    return q.omega * tf.tanh(column / q.H0) * tf.maximum(q.z0 - geom.draft, 0.0)
