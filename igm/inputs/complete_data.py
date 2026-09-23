#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from igm.processes.thk.masks import WATER_LEVEL_NO_OCEAN


def complete_data(state, water_level=None):
    """
    This function adds a postriori import fields such as X, Y, x, dx, ....

    ``water_level`` is an optional sub-config with fields ``include`` and
    ``value``. ``state.water_level`` is always created unless it was loaded
    from the input file: a uniform 2D field equal to ``value`` when
    ``include`` is True, otherwise the "no ocean" level
    (see ``igm.processes.thk.masks``).
    """

    # define grids, i.e. state.X and state.Y has same shape as state.thk
    if not hasattr(state, "X"):
        state.X, state.Y = tf.meshgrid(state.x, state.y)

    # define cell spacing
    if not hasattr(state, "dx"):
        state.dx = state.x[1] - state.x[0]

    # define dX
    if not hasattr(state, "dX"):
        state.dX = tf.ones_like(state.X) * state.dx

    # if thickness is not defined in the netcdf, then it is set to zero
    if not hasattr(state, "thk"):
        state.thk = tf.Variable(
            tf.zeros((state.y.shape[0], state.x.shape[0])), trainable=False
        )
    else:
        # Clamp to non-negative: some input NetCDFs encode small negative thk
        # values near ice edges (interpolation/rounding artifacts). The legacy
        # emulated path masked those out via `tf.where(thk > 0, U, 0)` so it
        # silently absorbed the inconsistency, but the unified solver feeds
        # `thk` directly into the gravity + viscosity-column integration and
        # blows up to NaN on negative values. Clamping here is a global cure.
        state.thk = tf.Variable(tf.maximum(state.thk, 0.0), trainable=False)

    assert hasattr(state, "topg") | hasattr(state, "usurf")

    # case usurf defined, topg is not defined
    if not hasattr(state, "topg"):
        state.topg = tf.Variable(state.usurf - state.thk, trainable=False)

    # case usurf not defined, topg is defined
    if not hasattr(state, "usurf"):
        state.usurf = tf.Variable(state.topg + state.thk, trainable=False)

    # water_level: a uniform sea/lake level when requested, else "no ocean";
    # a field loaded from the input file is kept.
    if not hasattr(state, "water_level"):
        include = water_level is not None and getattr(water_level, "include", False)
        level = water_level.value if include else WATER_LEVEL_NO_OCEAN
        state.water_level = tf.Variable(
            tf.ones_like(state.topg) * tf.cast(level, state.topg.dtype),
            trainable=False,
        )
