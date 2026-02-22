#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple

from ..stag.stag import stag2x, stag2y


@tf.function
def pad_x(X: tf.Tensor, mode: str = "symmetric") -> tf.Tensor:
    """Pad tensor by one cell in x-direction."""
    if mode == "periodic":
        X_l = X[..., :, -1:]
        X_r = X[..., :, 0:1]
    elif mode == "extrapolate":
        X_l = 2.0 * X[..., :, 0:1] - 1.0 * X[..., :, 1:2]
        X_r = 2.0 * X[..., :, -1:] - 1.0 * X[..., :, -2:-1]
    elif mode == "zero":
        X_l = tf.zeros_like(X[..., :, 0:1])
        X_r = tf.zeros_like(X[..., :, -1:])
    else:
        X_l = X[..., :, 0:1]
        X_r = X[..., :, -1:]
    return tf.concat([X_l, X, X_r], axis=-1)


@tf.function
def pad_y(X: tf.Tensor, mode: str = "symmetric") -> tf.Tensor:
    """Pad tensor by one cell in y-direction."""
    if mode == "periodic":
        X_l = X[..., -1:, :]
        X_r = X[..., 0:1, :]
    elif mode == "extrapolate":
        X_l = 2.0 * X[..., 0:1, :] - 1.0 * X[..., 1:2, :]
        X_r = 2.0 * X[..., -1:, :] - 1.0 * X[..., -2:-1, :]
    elif mode == "zero":
        X_l = tf.zeros_like(X[..., 0:1, :])
        X_r = tf.zeros_like(X[..., -1:, :])
    else:
        X_l = X[..., 0:1, :]
        X_r = X[..., -1:, :]
    return tf.concat([X_l, X, X_r], axis=-2)


@tf.function
def pad_z(X: tf.Tensor, mode: str = "symmetric") -> tf.Tensor:
    """Pad tensor by one cell in z-direction."""
    if mode == "periodic":
        X_l = X[..., -1:, :, :]
        X_r = X[..., 0:1, :, :]
    elif mode == "extrapolate":
        X_l = 2.0 * X[..., 0:1, :, :] - 1.0 * X[..., 1:2, :, :]
        X_r = 2.0 * X[..., -1:, :, :] - 1.0 * X[..., -2:-1, :, :]
    elif mode == "zero":
        X_l = tf.zeros_like(X[..., 0:1, :, :])
        X_r = tf.zeros_like(X[..., -1:, :, :])
    else:
        X_l = X[..., 0:1, :, :]
        X_r = X[..., -1:, :, :]
    return tf.concat([X_l, X, X_r], axis=-3)


@tf.function
def pad_xy(X: tf.Tensor, mode: str = "symmetric") -> tf.Tensor:
    """Pad tensor by one cell in both x and y directions."""
    return pad_y(pad_x(X, mode), mode)


def grad_stag(
    X: tf.Tensor,
    dx: tf.Tensor,
    dy: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute spatial gradient on staggered grid: (..., Ny, Nx) -> (..., Ny-1, Nx-1)."""

    X_is_2d = len(X.shape) == 2
    if X_is_2d:
        X = X[tf.newaxis, :, :]
    if len(dx.shape) == 2:
        dx = dx[tf.newaxis, :, :]
    if len(dy.shape) == 2:
        dy = dy[tf.newaxis, :, :]

    dXdx, dXdy = _grad_stag_impl(X, dx, dy)

    if X_is_2d:
        dXdx = tf.squeeze(dXdx, axis=0)
        dXdy = tf.squeeze(dXdy, axis=0)

    return dXdx, dXdy


@tf.function()
def _grad_stag_impl(
    X: tf.Tensor,
    dx: tf.Tensor,
    dy: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Internal implementation - assumes 3D inputs."""
    dXdx = (X[..., :, 1:] - X[..., :, :-1]) / dx[0, :, 1:]
    dXdy = (X[..., 1:, :] - X[..., :-1, :]) / dy[0, 1:, :]

    dXdx = stag2y(dXdx)
    dXdy = stag2x(dXdy)

    return dXdx, dXdy


def grad_unstag(
    X: tf.Tensor,
    dx: tf.Tensor,
    dy: tf.Tensor,
    mode: str = "symmetric",
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute spatial gradient on unstaggered grid: (..., Ny, Nx) -> (..., Ny, Nx)."""

    X_is_2d = len(X.shape) == 2
    if X_is_2d:
        X = X[tf.newaxis, :, :]
    if len(dx.shape) == 2:
        dx = dx[tf.newaxis, :, :]
    if len(dy.shape) == 2:
        dy = dy[tf.newaxis, :, :]

    dXdx, dXdy = _grad_unstag_impl(X, dx, dy, mode)

    if X_is_2d:
        dXdx = tf.squeeze(dXdx, axis=0)
        dXdy = tf.squeeze(dXdy, axis=0)

    return dXdx, dXdy


@tf.function()
def _grad_unstag_impl(
    X: tf.Tensor,
    dx: tf.Tensor,
    dy: tf.Tensor,
    mode: str = "symmetric",
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Internal implementation - assumes 3D inputs."""
    X_pad = pad_xy(X, mode)
    dX_pad = pad_xy(dx, mode)
    dY_pad = pad_xy(dy, mode)

    dXdx = (X_pad[..., 1:-1, 2:] - X_pad[..., 1:-1, :-2]) / (
        dX_pad[0, 1:-1, 1:-1] + dX_pad[0, 1:-1, 2:]
    )
    dXdy = (X_pad[..., 2:, 1:-1] - X_pad[..., :-2, 1:-1]) / (
        dY_pad[0, 1:-1, 1:-1] + dY_pad[0, 2:, 1:-1]
    )

    return dXdx, dXdy


@tf.function()
def grad_xy(
    X: tf.Tensor,
    dx: tf.Tensor,
    dy: tf.Tensor,
    staggered_grid: bool = True,
    mode: str = "symmetric",
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute spatial gradient."""
    if staggered_grid:
        return grad_stag(X, dx, dy)
    else:
        return grad_unstag(X, dx, dy, mode)
