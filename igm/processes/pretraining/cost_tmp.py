from __future__ import annotations

from typing import Callable, Dict

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.iceflow.energy.energy import iceflow_energy_UV
from igm.processes.iceflow.energy.utils import get_energy_components

DesiredInputs = ("thk", "usurf", "arrhenius", "slidingco", "dX")

"""
This is a temporary replacement for the default energy cost since that does not currently support certain fields missing from the inputs cfg 
(e.g. no Arrhenius-related channels, or no sliding coefficient channel). This is needed to allow for pretraining more varied networks but
hopefully can be removed in the future once we have a more flexible + robust energy cost implementation that can handle missing fields more gracefully.

"""

def get_cost_fn(cfg: DictConfig, state: State) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:
    cfg_unified  = cfg.processes.iceflow.unified
    cfg_physics  = cfg.processes.iceflow.physics
    cfg_numerics = cfg.processes.iceflow.numerics

    Nz = int(cfg_numerics.Nz)

    # Defaults when missing from cfg_unified.inputs
    slidingco0 = float(cfg_physics.init_slidingco)
    arrhenius0 = float(cfg_physics.init_arrhenius)
    dX0 = 90.0

    # Network input schema (single-channel per name, order matters)
    net_inputs = tuple(str(x) for x in cfg_unified.inputs)
    net_idx: Dict[str, int] = {}
    for i, nm in enumerate(net_inputs):
        key = nm.lower()
        if key in net_idx:
            raise ValueError(f"Duplicate input name in cfg.processes.iceflow.unified.inputs: '{nm}'")
        net_idx[key] = i

    expected_C = len(net_inputs)

    energy_components = get_energy_components(cfg)

    @tf.function(reduce_retracing=True)
    def build_energy_inputs(input_net: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_rank(input_net, 4)
        tf.debugging.assert_equal(
            tf.shape(input_net)[-1],
            expected_C,
            message="input_net channel count must equal len(cfg.processes.iceflow.unified.inputs).",
        )

        proto = input_net[..., :1]  # [B,Ny,Nx,1]
        ones1 = tf.ones_like(proto)

        def const_chan(val: float) -> tf.Tensor:
            return ones1 * tf.cast(val, input_net.dtype)

        chans = []
        for nm in DesiredInputs:
            key = nm.lower()
            if key in net_idx:
                i = net_idx[key]
                chans.append(input_net[..., i:i+1])
            else:
                if key == "dx":
                    chans.append(const_chan(dX0))
                elif key == "slidingco":
                    chans.append(const_chan(slidingco0))
                elif key == "arrhenius":
                    chans.append(const_chan(arrhenius0))
                else:
                    chans.append(const_chan(0.0))

        return tf.concat(chans, axis=-1)

    def cost_fn(U: tf.Tensor, V: tf.Tensor, input_net: tf.Tensor) -> tf.Tensor:
        input_energy = build_energy_inputs(input_net)

        energy = iceflow_energy_UV(
            inputs_names=list(DesiredInputs),
            inputs=input_energy,
            U=U,
            V=V,
            discr_h=state.iceflow.discr_h,
            discr_v=state.iceflow.discr_v,
            energy_components=energy_components,
        )

        energy_mean = tf.reduce_mean(energy, axis=[1, 2, 3])
        return tf.reduce_sum(energy_mean, axis=0)

    return cost_fn