#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


def regu_fastfric(cfg, state):
    """Zero-order pull of the friction control toward 0, weighted by observed speed.

    COST = regularization.fastfric * mean( w(u_obs) * field^2 )

        w(u) = x^2 / (1 + x^2),   x = max(u - fastfric_uthr, 0) / fastfric_uscale

    Where the ice is fast, basal drag is nearly exhausted and the surface
    velocity carries almost no information on the friction: du/dtau -> 0 and
    the friction is unidentifiable there. This term selects the minimum-norm
    solution inside that null space instead of leaving it to the optimiser,
    which otherwise leaves grid-scale structure that the tau >= 0 clip
    rectifies into a positive bias.

    w is exactly 0 below fastfric_uthr, so slow ice is untouched. u_obs is
    data, not a control: the weight is constant through the optimisation.
    A wide fastfric_uscale spreads the transition over a whole trunk rather
    than a contour line of u_obs.
    """
    da = cfg.assimilations.data_assimilation
    weight = float(da.regularization.get("fastfric", 0.0))
    if weight <= 0.0:
        return tf.constant(0.0)

    u_thr = float(da.regularization.get("fastfric_uthr", 300.0))
    u_scale = float(da.regularization.get("fastfric_uscale", 300.0))

    field = getattr(state, state.da_friction)
    uobs = tf.math.sqrt(state.uvelsurfobs ** 2 + state.vvelsurfobs ** 2)
    uobs = tf.where(tf.math.is_nan(uobs), 0.0, uobs)

    x = tf.maximum(uobs - u_thr, 0.0) / u_scale
    w = x ** 2 / (1.0 + x ** 2)

    if da.optimization.sole_mask:
        w = tf.where(state.icemaskobs > 0.5, w, 0.0)

    return weight * tf.math.reduce_mean(w * field ** 2)
