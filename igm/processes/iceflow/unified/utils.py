#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Callable

from igm.common import State
from igm.processes.iceflow.energy.components.sliding import SlidingComponents
from igm.processes.iceflow.energy.energy import iceflow_energy_UV
from igm.processes.iceflow.energy.utils import get_energy_components
from igm.processes.thk.masks import (
    compute_grounded_mask,
    weights_gr,
    water_level_from_inputs,
)
from igm.processes.iceflow.utils.velocities import compute_cell_ice_mask


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


def get_cost_fn(
    cfg: DictConfig, state: State
) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
    """Create cost function for ice flow optimization."""

    cfg_unified = cfg.processes.iceflow.unified

    energy_components = get_energy_components(cfg)
    input_names = tuple(cfg_unified.inputs)
    idx_thk = input_names.index("thk")
    idx_usurf = input_names.index("usurf")
    rho_ratio = (
        cfg.processes.iceflow.physics.water_density
        / cfg.processes.iceflow.physics.ice_density
    )
    # Pixel-robust training loss (see iceflow.yaml `loss_clip`): cap each cell's
    # energy contribution at factor x the domain-mean |energy|, so one
    # pathological cell (a velocity burst on extreme geometry) cannot dominate
    # the gradient and corrupt the global weights. 0 disables.
    loss_clip_factor = float(
        getattr(getattr(cfg_unified, "loss_clip", None), "factor", 0.0) or 0.0
    ) if bool(getattr(getattr(cfg_unified, "loss_clip", None), "enabled", False)) else 0.0

    def _cost_fn_impl(U: tf.Tensor, V: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
        """Cost function from velocity fields and inputs."""
        energy = iceflow_energy_UV(
            inputs_names=tuple(cfg_unified.inputs),
            inputs=input,
            U=U,
            V=V,
            discr_h=state.iceflow.discr_h,
            discr_v=state.iceflow.discr_v,
            energy_components=energy_components,
        )

        thk = input[..., idx_thk]
        topg = input[..., idx_usurf] - thk
        water_level = water_level_from_inputs(input, input_names, thk)
        grounded = compute_grounded_mask(thk, topg, water_level, rho_ratio)
        cell_mask = compute_cell_ice_mask(thk, grounded)
        energy = energy * tf.cast(cell_mask[tf.newaxis, :, :, :], energy.dtype)

        if loss_clip_factor > 0.0:
            abs_mean = tf.reduce_mean(
                tf.abs(energy), axis=[1, 2, 3], keepdims=True
            )
            energy = tf.minimum(energy, loss_clip_factor * abs_mean + 1e-30)

        energy_mean = tf.reduce_mean(energy, axis=[1, 2, 3])
        total_energy = tf.reduce_sum(energy_mean, axis=0)

        return total_energy

    # Keep the nested HVP tapes compact by differentiating a PartitionedCall.
    cost_fn = tf.function(_cost_fn_impl, reduce_retracing=True)
    return cost_fn
