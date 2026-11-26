#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Turn a velocity-error field into the scalars the SS-eSOAP studies report.

The formulas mirror ``igm-ais/scripts/score_snapshot.py`` and
``benchmark_network_architectures.py``: a pointwise relative error
``100 * |du| / |u_ref|`` on ice nodes with a positive reference speed, its
median over grounded and floating ice, and RMSE values in m/yr.
"""

from typing import Dict, Optional

import tensorflow as tf


def masked_median(values: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Median of ``values`` where ``mask`` is true; NaN when the mask is empty.

    Static-shape implementation: masked entries are pushed to ``+inf`` and the
    sorted array is indexed at ``(n_valid - 1) // 2`` (lower median).
    """
    flat = tf.reshape(values, [-1])
    mask_flat = tf.reshape(mask, [-1])
    inf = tf.constant(float("inf"), flat.dtype)
    nan = tf.constant(float("nan"), flat.dtype)
    pushed = tf.where(mask_flat, flat, inf)
    ordered = tf.sort(pushed)
    n_valid = tf.reduce_sum(tf.cast(mask_flat, tf.int32))
    index = tf.maximum(n_valid - 1, 0) // 2
    return tf.where(n_valid > 0, ordered[index], nan)


def masked_rmse(values: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """Root-mean-square of ``values`` over ``mask``; NaN when the mask is empty."""
    weights = tf.cast(mask, values.dtype)
    n_valid = tf.reduce_sum(weights)
    nan = tf.constant(float("nan"), values.dtype)
    mean_sq = tf.reduce_sum(weights * tf.square(values)) / tf.maximum(n_valid, 1.0)
    return tf.where(n_valid > 0, tf.sqrt(mean_sq), nan)


def relative_error_percent(
    du: tf.Tensor,
    dv: tf.Tensor,
    u_ref: tf.Tensor,
    v_ref: tf.Tensor,
) -> tf.Tensor:
    """Pointwise ``100 |du| / |u_ref|``; zero where the reference speed is zero."""
    speed_ref = tf.sqrt(tf.square(u_ref) + tf.square(v_ref))
    error = tf.sqrt(tf.square(du) + tf.square(dv))
    tiny = tf.cast(1e-30, du.dtype)
    return tf.where(
        speed_ref > 0.0,
        100.0 * error / tf.maximum(speed_ref, tiny),
        tf.zeros_like(error),
    )


def summarize_velocity_error(
    du: tf.Tensor,
    dv: tf.Tensor,
    u_ref: tf.Tensor,
    v_ref: tf.Tensor,
    ice_mask: tf.Tensor,
    grounded: Optional[tf.Tensor],
    prefix: str,
) -> Dict[str, tf.Tensor]:
    """Medians of the relative error and RMSEs over grounded/floating/all ice.

    ``du, dv`` is the (estimated or true) surface-velocity error, ``u_ref,
    v_ref`` the (estimated or true) reference velocity used as denominator.
    All fields are ``[Ny, Nx]``. Keys are ``{prefix}_median_rel_{region}``,
    ``{prefix}_rmse_{region}`` and ``{prefix}_cells_{region}``.
    """
    speed_ref = tf.sqrt(tf.square(u_ref) + tf.square(v_ref))
    error_mag = tf.sqrt(tf.square(du) + tf.square(dv))
    finite = tf.math.is_finite(speed_ref) & tf.math.is_finite(error_mag)
    valid = ice_mask & finite & (speed_ref > 0.0)
    relative = relative_error_percent(du, dv, u_ref, v_ref)

    if grounded is None:
        regions = {"all": valid}
    else:
        regions = {
            "grounded": valid & grounded,
            "floating": valid & tf.logical_not(grounded),
            "all": valid,
        }

    out: Dict[str, tf.Tensor] = {}
    for name, mask in regions.items():
        out[f"{prefix}_median_rel_{name}"] = masked_median(relative, mask)
        out[f"{prefix}_rmse_{name}"] = masked_rmse(error_mag, mask)
        out[f"{prefix}_cells_{name}"] = tf.cast(
            tf.reduce_sum(tf.cast(mask, tf.int32)), du.dtype
        )
    return out
