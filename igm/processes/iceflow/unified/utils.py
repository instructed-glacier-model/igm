#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Callable

from igm.common import State
from igm.processes.iceflow.energy.energy import iceflow_energy_UV
from igm.processes.iceflow.energy.utils import get_energy_components


def get_cost_fn(
    cfg: DictConfig, state: State
) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
    """Create cost function for ice flow optimization."""

    cfg_unified = cfg.processes.iceflow.unified

    energy_components = get_energy_components(cfg)

    def cost_fn(U: tf.Tensor, V: tf.Tensor, input: tf.Tensor) -> tf.Tensor:
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

        energy_mean = tf.reduce_mean(energy, axis=[1, 2, 3])
        total_energy = tf.reduce_sum(energy_mean, axis=0)

        return total_energy

    return cost_fn
