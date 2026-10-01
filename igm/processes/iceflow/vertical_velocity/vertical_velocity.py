#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.iceflow.vertical_velocity.vertical_velocity_v1 import (
    compute_vertical_velocity_v1,
)
from igm.processes.iceflow.vertical_velocity.vertical_velocity_v2 import (
    compute_vertical_velocity_v2,
)
from igm.processes.iceflow.vertical_velocity.vertical_velocity_v3 import (
    compute_vertical_velocity_v3,
)
from igm.processes.iceflow.utils.velocities import get_velbase_1, get_velsurf_1
from igm.utils.math.gaussian_filter_tf import gaussian_filter_tf


def smooth_vertical_velocity(W: tf.Tensor, thk: tf.Tensor, sigma: float) -> tf.Tensor:
    """Ice-masked Gaussian filter (std `sigma` grid cells) of each vertical DOF of W.

    Normalised convolution, so ice-free cells do not bleed into the glacier and the mean
    over the ice is preserved. The filter is linear and horizontal, so it applies equally
    to nodal values (Lagrange, MOLHO) and to spectral coefficients (Legendre). It removes
    the one-cell texture that the coupled ice flow and thickness transport put into W
    at any grid resolution. W does not feed back into the ice dynamics."""
    kernel_size = 2 * int(3 * sigma) + 1
    mask = tf.cast(thk > 0.0, W.dtype)
    norm = gaussian_filter_tf(mask, sigma=sigma, kernel_size=kernel_size)
    out = []
    for l in range(int(W.shape[0])):
        f = gaussian_filter_tf(W[l] * mask, sigma=sigma, kernel_size=kernel_size)
        out.append(tf.where(norm > 1e-6, f / tf.maximum(norm, 1e-6), 0.0) * mask)
    return tf.stack(out, axis=0)


def update(cfg: DictConfig, state: State) -> None:
    version = cfg.processes.iceflow.vertical_velocity.version

    if version == 1:
        compute_vertical_velocity = compute_vertical_velocity_v1
    elif version == 2:
        compute_vertical_velocity = compute_vertical_velocity_v2
    elif version == 3:
        compute_vertical_velocity = compute_vertical_velocity_v3
    else:
        raise ValueError(f"❌ Unknown vertical_velocity version: <{version}>.")

    state.W = compute_vertical_velocity(cfg, state)

    smooth_sigma = float(cfg.processes.iceflow.vertical_velocity.get("smooth_sigma", 0.0))
    if smooth_sigma > 0.0:
        state.W = smooth_vertical_velocity(state.W, state.thk, smooth_sigma)

    state.wvelbase = get_velbase_1(state.W, state.iceflow.discr_v.V_b)
    state.wvelsurf = get_velsurf_1(state.W, state.iceflow.discr_v.V_s)
