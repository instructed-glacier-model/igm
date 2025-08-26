#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import warnings
from omegaconf import DictConfig
from typing import Any, Dict

import igm
from igm.common.core import State
from igm.processes.iceflow.emulate.utils.misc import (
    get_pretrained_emulator_path,
    load_model_from_path,
)
from .interface import InterfaceMapping


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

            # TO DO: construct the network parameters based on cfg_unified, not cfg_emulator
            iceflow_model = getattr(
                igm.processes.iceflow.emulate.utils.networks,
                cfg_unified.network.architecture,
            )(cfg, nb_inputs, nb_outputs)

        state.iceflow_model = iceflow_model
        state.iceflow_model.compile(jit_compile=True)

        return {
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": cfg_unified.network.output_scale,
        }
