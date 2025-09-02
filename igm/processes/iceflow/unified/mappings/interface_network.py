#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import warnings
from omegaconf import DictConfig
from typing import Any, Dict
import numpy as np

import igm
from igm.common.core import State
from igm.processes.iceflow.emulate.utils.misc import (
    get_pretrained_emulator_path,
    load_model_from_path,
)
from .interface import InterfaceMapping
from igm.processes.iceflow.emulate.utils.networks import cnn, unet, build_norm_layer


class InterfaceNetwork(InterfaceMapping):

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:

        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_physics = cfg.processes.iceflow.physics
        cfg_unified = cfg.processes.iceflow.unified

        if cfg_unified.pretrained:
            dir_path = get_pretrained_emulator_path(cfg, state)
            iceflow_model = load_model_from_path(dir_path, cfg_unified.inputs)
        else:
            warnings.warn("No pretrained emulator found. Starting from scratch.")

            nb_inputs = len(cfg_unified.fieldin) + (cfg_physics.dim_arrhenius == 3) * (
                cfg_numerics.Nz - 1
            )
            nb_outputs = 2 * cfg_numerics.Nz

            if np.all(cfg_unified.inputs_scales == 1):
                norm = None
            else:
                norm = build_norm_layer(cfg, nb_inputs, cfg_unified.inputs_scales)

            # Get the architecture function dynamically
            architecture_name = cfg_unified.network.architecture

            # Get the function from the networks module
            if hasattr(igm.processes.iceflow.emulate.utils.networks, architecture_name):
                architecture_fn = getattr(
                    igm.processes.iceflow.emulate.utils.networks, architecture_name
                )
                iceflow_model = architecture_fn(
                    cfg, nb_inputs, nb_outputs, input_normalizer=norm
                )
            else:
                raise ValueError(
                    f"Unknown network architecture: {architecture_name}. "
                    f"Available architectures: cnn, unet"
                )

        state.iceflow_model = iceflow_model
        state.iceflow_model.compile(jit_compile=True)

        return {
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": cfg_unified.network.output_scale,
        }
