#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Input tensors of the ice-flow mappings, built from the state.

The inputs are the fields listed in ``processes.iceflow.unified.inputs``,
stacked along the last (channel) axis:

- ``get_inputs_from_state``: full domain, ``[H, W, C]``;
- ``get_solver_inputs_from_state``: patches for the solver, ``[N, ly, lx, C]``;
- ``get_evaluator_inputs_from_state``: full domain for the evaluator, ``[1, H, W, C]``.
"""

from typing import Tuple

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.iceflow.energy.components.sliding import SlidingComponents
from igm.processes.thk.masks import weights_gr
from igm.utils.math.precision import normalize_precision


def get_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:
    """Stack the unified inputs read from ``state`` into a [H, W, C] tensor.

    With ``physics.sliding.use_mask_gr`` and ``use_subgrid_gl``, the
    ``friction_inputs`` of the sliding law are multiplied by the grounded
    weights of the nodes (``weights_gr``, from ``state.topg``); the inputs
    keep their names and number.
    """
    cfg_physics = cfg.processes.iceflow.physics
    cfg_sliding = cfg_physics.sliding
    fields = {
        name: getattr(state, name) for name in cfg.processes.iceflow.unified.inputs
    }
    if cfg_sliding.use_mask_gr and cfg_sliding.use_subgrid_gl:
        weight = weights_gr(
            state.thk,
            state.topg,
            state.water_level,
            cfg_physics.water_density / cfg_physics.ice_density,
        )
        for name in SlidingComponents[cfg_sliding.law].friction_inputs:
            if name in fields:
                fields[name] = fields[name] * weight
    return tf.stack(list(fields.values()), axis=-1)


def get_solver_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:
    """Returns [N, ly, lx, C] non-overlapping patches (see split_field_into_patches)."""
    X = get_inputs_from_state(cfg, state)
    framesizemax = cfg.processes.iceflow.unified.data_preparation.framesizemax
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)
    return tf.cast(split_field_into_patches(X, framesizemax), dtype)


def get_evaluator_inputs_from_state(cfg: DictConfig, state: State) -> tf.Tensor:
    """Returns the full-domain inputs as a single batch, [1, H, W, C]."""
    X = tf.expand_dims(get_inputs_from_state(cfg, state), axis=0)
    dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)
    return tf.cast(X, dtype)


def patch_grid(ny: int, nx: int, framesizemax: int) -> Tuple[int, int]:
    """Number of patch strips (sy, sx) in y and x for a [ny, nx] grid."""
    return ny // framesizemax + 1, nx // framesizemax + 1


def split_field_into_patches(X: tf.Tensor, framesizemax: int) -> tf.Tensor:
    """
    Split a [H, W, C] field into non-overlapping patches [N, ly, lx, C].

    Grid-division strategy:
    - sy = H // framesizemax + 1 strips in y
    - sx = W // framesizemax + 1 strips in x
    - each tile is ly = H // sy  by  lx = W // sx  pixels
    - patches are guaranteed to be smaller than framesizemax
    """
    ny = int(X.shape[0])
    nx = int(X.shape[1])
    sy, sx = patch_grid(ny, nx, framesizemax)
    ly = ny // sy
    lx = nx // sx

    patches = []
    for j in range(sy):
        for i in range(sx):
            patches.append(X[j * ly : (j + 1) * ly, i * lx : (i + 1) * lx, :])

    return tf.stack(patches, axis=0)  # [sy*sx, ly, lx, C]
