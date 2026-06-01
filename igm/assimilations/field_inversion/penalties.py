#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import tensorflow as tf

from .utils import masked_integral


def lap_sq(
    field: tf.Tensor,
    dx: tf.Tensor,
    lam: tf.Tensor,
    mask: tf.Tensor,
    area: tf.Tensor,
    eps: float = 1e-12,
    ref: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    0.5 * lam * ( ∫_mask lap_mask(field - ref)^2 dA ) / area

    If ref is not provided, this regularizes field directly. lap_mask uses
    only neighbours inside mask, so outside-mask values do not influence the
    penalty.
    """
    dtype = field.dtype
    m = tf.cast(mask, tf.bool)
    field_active = tf.where(m, tf.cast(field, dtype), tf.zeros_like(field))
    f = field_active

    if ref is not None:
        ref_active = tf.where(m, tf.cast(ref, dtype), tf.zeros_like(f))
        f = field_active - ref_active

    # Inactive NaNs have been suppressed before stencil arithmetic. Active NaNs
    # remain visible because they indicate a real model/evaluation failure.
    mf = tf.cast(m, dtype)

    fpad = tf.pad(f, [[1, 1], [1, 1]], mode="SYMMETRIC")
    mpad = tf.pad(mf, [[1, 1], [1, 1]], mode="CONSTANT", constant_values=0.0)

    c = fpad[1:-1, 1:-1]
    fu = fpad[0:-2, 1:-1]
    fd = fpad[2:, 1:-1]
    fl = fpad[1:-1, 0:-2]
    fr = fpad[1:-1, 2:]

    mu = mpad[0:-2, 1:-1]
    md = mpad[2:, 1:-1]
    ml = mpad[1:-1, 0:-2]
    mr = mpad[1:-1, 2:]

    dx = tf.cast(dx, dtype)
    lap = (mu * (fu - c) + md * (fd - c) + ml * (fl - c) + mr * (fr - c)) / (dx * dx)
    integral = masked_integral(tf.square(lap), m, dx)
    denom = tf.cast(area, dtype) + tf.cast(eps, dtype)

    return tf.cast(0.5, dtype) * lam * integral / denom


def penalty_l2(
    field: tf.Tensor,
    dx: tf.Tensor,
    lam: tf.Tensor,
    mask: tf.Tensor,
    area: tf.Tensor,
    eps: float = 1e-12,
    ref: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    0.5 * lam * ( ∫_mask (field - ref)^2 dA ) / area   if ref is provided
    0.5 * lam * ( ∫_mask field^2 dA ) / area           otherwise
    """
    dtype = field.dtype
    m = tf.cast(mask, tf.bool)
    field_active = tf.where(m, tf.cast(field, dtype), tf.zeros_like(field))
    if ref is None:
        diff = field_active
    else:
        ref_active = tf.where(m, tf.cast(ref, dtype), tf.zeros_like(field_active))
        diff = field_active - ref_active

    integral = masked_integral(tf.square(diff), m, dx)
    denom = tf.cast(area, dtype) + tf.cast(eps, dtype)
    return tf.cast(0.5, dtype) * lam * integral / denom


PenaltyRegistry = {
    "squared_laplacian": lap_sq,
    "l2": penalty_l2,
}
