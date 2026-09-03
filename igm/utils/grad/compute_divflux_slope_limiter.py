"""Slope-limited finite-volume flux divergence for thickness transport."""

import tensorflow as tf

from igm.utils.math.gaussian_filter_tf import gaussian_filter_tf


def minmod(a, b):
    same_sign = a * b > 0.0
    return tf.where(
        (tf.abs(a) < tf.abs(b)) & same_sign,
        a,
        tf.where(
            (tf.abs(a) > tf.abs(b)) & same_sign,
            b,
            tf.zeros_like(a),
        ),
    )


def maxmod(a, b):
    same_sign = a * b > 0.0
    return tf.where(
        (tf.abs(a) < tf.abs(b)) & same_sign,
        b,
        tf.where(
            (tf.abs(a) > tf.abs(b)) & same_sign,
            a,
            tf.zeros_like(a),
        ),
    )


def _pad_x_two(field, left_symmetric, right_symmetric):
    zeros = tf.zeros_like(field[:, :1])
    left = (
        tf.concat([field[:, :1], field[:, :1]], axis=1)
        if left_symmetric
        else tf.concat([zeros, zeros], axis=1)
    )
    right = (
        tf.concat([field[:, -1:], field[:, -1:]], axis=1)
        if right_symmetric
        else tf.concat([zeros, zeros], axis=1)
    )
    return tf.concat([left, field, right], axis=1)


def _pad_y_two(field, top_symmetric, bottom_symmetric):
    zeros = tf.zeros_like(field[:1, :])
    top = (
        tf.concat([field[:1, :], field[:1, :]], axis=0)
        if top_symmetric
        else tf.concat([zeros, zeros], axis=0)
    )
    bottom = (
        tf.concat([field[-1:, :], field[-1:, :]], axis=0)
        if bottom_symmetric
        else tf.concat([zeros, zeros], axis=0)
    )
    return tf.concat([top, field, bottom], axis=0)


def _zero_x_boundary_flux(flux, left_symmetric, right_symmetric):
    if left_symmetric:
        flux = tf.concat([tf.zeros_like(flux[:, :1]), flux[:, 1:]], axis=1)
    if right_symmetric:
        flux = tf.concat([flux[:, :-1], tf.zeros_like(flux[:, -1:])], axis=1)
    return flux


def _zero_y_boundary_flux(flux, top_symmetric, bottom_symmetric):
    if top_symmetric:
        flux = tf.concat([tf.zeros_like(flux[:1, :]), flux[1:, :]], axis=0)
    if bottom_symmetric:
        flux = tf.concat([flux[:-1, :], tf.zeros_like(flux[-1:, :])], axis=0)
    return flux


def _compute_divflux_slope_limiter(
    u,
    v,
    h,
    dx,
    dy,
    dt,
    slope_type,
    smooth_sigma,
    x_face_mask,
    y_face_mask,
    left_symmetric=False,
    right_symmetric=False,
    top_symmetric=False,
    bottom_symmetric=False,
):
    """Trace the common limiter algebra into either public graph."""
    if not any(
        (left_symmetric, right_symmetric, top_symmetric, bottom_symmetric)
    ):
        # Preserve the original graph on the overwhelmingly common path.
        u = tf.concat(
            [u[:, :1], 0.5 * (u[:, :-1] + u[:, 1:]), u[:, -1:]],
            axis=1,
        )
        v = tf.concat(
            [v[:1, :], 0.5 * (v[:-1, :] + v[1:, :]), v[-1:, :]],
            axis=0,
        )
        hx = tf.pad(h, [[0, 0], [2, 2]], "CONSTANT")
        hy = tf.pad(h, [[2, 2], [0, 0]], "CONSTANT")
    else:
        left_u = tf.zeros_like(u[:, :1]) if left_symmetric else u[:, :1]
        right_u = tf.zeros_like(u[:, -1:]) if right_symmetric else u[:, -1:]
        top_v = tf.zeros_like(v[:1, :]) if top_symmetric else v[:1, :]
        bottom_v = tf.zeros_like(v[-1:, :]) if bottom_symmetric else v[-1:, :]
        u = tf.concat(
            [left_u, 0.5 * (u[:, :-1] + u[:, 1:]), right_u], axis=1
        )
        v = tf.concat(
            [top_v, 0.5 * (v[:-1, :] + v[1:, :]), bottom_v], axis=0
        )
        hx = _pad_x_two(h, left_symmetric, right_symmetric)
        hy = _pad_y_two(h, top_symmetric, bottom_symmetric)

    sigpx = (hx[:, 2:] - hx[:, 1:-1]) / dx
    sigmx = (hx[:, 1:-1] - hx[:, :-2]) / dx
    sigpy = (hy[2:, :] - hy[1:-1, :]) / dy
    sigmy = (hy[1:-1, :] - hy[:-2, :]) / dy

    if slope_type == "godunov":
        slopex = tf.zeros_like(sigpx)
        slopey = tf.zeros_like(sigpy)
    elif slope_type == "minmod":
        slopex = minmod(sigmx, sigpx)
        slopey = minmod(sigmy, sigpy)
    elif slope_type == "superbee":
        sig1x = minmod(sigpx, 2.0 * sigmx)
        sig2x = minmod(sigmx, 2.0 * sigpx)
        slopex = maxmod(sig1x, sig2x)
        sig1y = minmod(sigpy, 2.0 * sigmy)
        sig2y = minmod(sigmy, 2.0 * sigpy)
        slopey = maxmod(sig1y, sig2y)
    else:
        raise ValueError(
            "slope_type must be 'godunov', 'minmod', or 'superbee'; "
            f"got {slope_type!r}."
        )

    west = hx[:, 1:-2] + 0.5 * dx * (1.0 - u * dt / dx) * slopex[:, :-1]
    east = hx[:, 2:-1] - 0.5 * dx * (1.0 + u * dt / dx) * slopex[:, 1:]
    south = hy[1:-2, :] + 0.5 * dy * (1.0 - v * dt / dy) * slopey[:-1, :]
    north = hy[2:-1, :] - 0.5 * dy * (1.0 + v * dt / dy) * slopey[1:, :]

    flux_x = u * tf.where(u > 0.0, west, east)
    flux_y = v * tf.where(v > 0.0, south, north)

    if smooth_sigma > 0.0:
        kernel_size = 2 * int(3 * smooth_sigma) + 1
        flux_x = gaussian_filter_tf(
            flux_x, sigma=smooth_sigma, kernel_size=kernel_size
        )
        flux_y = gaussian_filter_tf(
            flux_y, sigma=smooth_sigma, kernel_size=kernel_size
        )

    # Filtering may redistribute interior face flux, but a symmetry boundary
    # remains an exact no-flux face.
    flux_x = _zero_x_boundary_flux(
        flux_x, left_symmetric, right_symmetric
    )
    flux_y = _zero_y_boundary_flux(
        flux_y, top_symmetric, bottom_symmetric
    )

    if x_face_mask is not None:
        flux_x = flux_x * tf.cast(x_face_mask, flux_x.dtype)
    if y_face_mask is not None:
        flux_y = flux_y * tf.cast(y_face_mask, flux_y.dtype)

    return (flux_x[:, 1:] - flux_x[:, :-1]) / dx + (
        flux_y[1:, :] - flux_y[:-1, :]
    ) / dy


@tf.function(autograph=False, reduce_retracing=True)
def compute_divflux_slope_limiter(
    u,
    v,
    h,
    dx,
    dy,
    dt,
    slope_type,
    smooth_sigma=0.0,
    x_face_mask=None,
    y_face_mask=None,
):
    """Historical zero-exterior slope-limited flux divergence.

    This entry point intentionally keeps the original all-open graph. The
    thickness backend calls it directly for the default configuration, so
    symmetric-boundary support adds no tensors or branches to the hot path.
    """
    return _compute_divflux_slope_limiter(
        u,
        v,
        h,
        dx,
        dy,
        dt,
        slope_type,
        smooth_sigma,
        x_face_mask,
        y_face_mask,
    )


@tf.function(autograph=False, reduce_retracing=True)
def compute_divflux_slope_limiter_symmetric(
    u,
    v,
    h,
    dx,
    dy,
    dt,
    slope_type,
    smooth_sigma=0.0,
    x_face_mask=None,
    y_face_mask=None,
    left=False,
    right=False,
    top=False,
    bottom=False,
):
    """Slope-limited divergence with selected reflecting domain sides."""
    return _compute_divflux_slope_limiter(
        u,
        v,
        h,
        dx,
        dy,
        dt,
        slope_type,
        smooth_sigma,
        x_face_mask,
        y_face_mask,
        left,
        right,
        top,
        bottom,
    )
