#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

def complete_data(state, water_level=None):
    """
    This function adds a postriori import fields such as X, Y, x, dx, ....

    ``water_level`` is an optional sub-config with fields ``include`` and
    ``value``. When ``include`` is True and ``state.water_level`` is not
    already present (not loaded from the input NetCDF), a uniform 2D
    field equal to ``value`` is created. Downstream modules that need a
    water level -- e.g. the ``floating`` energy component of iceflow --
    then see it on state without the ``thk`` / ``thk_ls`` module having
    to populate it, which would force an awkward process ordering.
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
        state.thk = tf.Variable(tf.zeros((state.y.shape[0], state.x.shape[0])), trainable=False)
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

    # water_level: populate a uniform field only when explicitly requested
    # and not already loaded from the NetCDF.
    if (water_level is not None
            and getattr(water_level, "include", False)
            and not hasattr(state, "water_level")):
        state.water_level = tf.Variable(
            tf.ones_like(state.topg)
            * tf.cast(water_level.value, state.topg.dtype),
            trainable=False,
        )

