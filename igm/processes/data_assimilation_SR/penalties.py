#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional
import tensorflow as tf

from .utils import masked_integral


def lap_sq(
    field: tf.Tensor,
    dx: tf.Tensor,
    lam: tf.Tensor,
    mask: Optional[tf.Tensor],
    A_domain: tf.Tensor,
    eps: float = 1e-12,
    ref: Optional[tf.Tensor] = None,  # unused here
) -> tf.Tensor:
    """
    0.5 * lam * ( ∫_mask lap_mask(field)^2 dA ) / A_domain

    lap_mask uses only neighbours that are inside mask (Neumann-like at mask boundary),
    so outside-mask frozen values do NOT influence the penalty.
    """
    dtype = field.dtype
    f = tf.cast(field, dtype)
    if ref is not None:
        f = f - tf.cast(ref, dtype)

    m = tf.ones_like(f, tf.bool) if mask is None else tf.cast(mask, tf.bool)
    mf = tf.cast(m, dtype)

    # Pad domain boundary (array edges). Mask outside array is False.
    fpad = tf.pad(f, [[1, 1], [1, 1]], mode="SYMMETRIC")
    mpad = tf.pad(mf, [[1, 1], [1, 1]], mode="CONSTANT", constant_values=0.0)

    c  = fpad[1:-1, 1:-1]
    mu = mpad[0:-2, 1:-1]
    md = mpad[2:  , 1:-1]
    ml = mpad[1:-1, 0:-2]
    mr = mpad[1:-1, 2:  ]

    fu = fpad[0:-2, 1:-1]
    fd = fpad[2:  , 1:-1]
    fl = fpad[1:-1, 0:-2]
    fr = fpad[1:-1, 2:  ]

    # Only count neighbour differences when neighbour is inside mask.
    lap_num = mu * (fu - c) + md * (fd - c) + ml * (fl - c) + mr * (fr - c)

    dx = tf.cast(dx, dtype)
    dx2 = dx * dx
    lap = lap_num / dx2

    lap2 = tf.square(lap)
    integral = masked_integral(lap2, tf.cast(m, tf.bool), dx)
    denom = tf.cast(A_domain, dtype) + tf.cast(eps, dtype)

    return tf.cast(0.5, dtype) * lam * integral / denom


def penalty_l2(
    field: tf.Tensor,
    dx: tf.Tensor,
    lam: tf.Tensor,
    mask: Optional[tf.Tensor],
    A_domain: tf.Tensor,
    eps: float = 1e-12,
    ref: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    0.5 * lam * ( ∫ (field - ref)^2 dA ) / A_domain   if ref is provided
    0.5 * lam * ( ∫ field^2 dA ) / A_domain           otherwise
    """
    dtype = field.dtype
    diff = field if ref is None else (field - tf.cast(ref, dtype))
    sq = tf.square(diff)

    m = tf.ones_like(sq, dtype=tf.bool) if mask is None else tf.cast(mask, tf.bool)
    integral = masked_integral(sq, m, dx)

    denom = tf.cast(A_domain, dtype) + tf.cast(eps, dtype)
    return tf.cast(0.5, dtype) * lam * integral / denom


PenaltyRegistry = {
    "squared_laplacian": lap_sq,
    "l2": penalty_l2,
}
