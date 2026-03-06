#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import warnings
from omegaconf import DictConfig
from typing import Any, Dict


import igm
from igm.common import State
from igm.processes.iceflow.unified.bcs.utils import init_bcs
from igm.processes.iceflow.emulate.utils.misc import (
    get_pretrained_emulator_path,
    load_model_from_path,
)
from igm.utils.math.precision import normalize_precision
from .interface import InterfaceMapping
from igm.processes.iceflow.emulate import Architectures, NormalizationsDict


class InterfaceNetwork(InterfaceMapping):

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:

        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_physics = cfg.processes.iceflow.physics
        cfg_unified = cfg.processes.iceflow.unified

        dtype = normalize_precision(cfg_numerics.precision).name

        normalizing_method = cfg_unified.normalization.method
        normalizing_class = NormalizationsDict[normalizing_method]
        if normalizing_method == "adaptive":
            nb_channels = len(cfg.processes.iceflow.unified.inputs)
            normalizing_layer = normalizing_class(nb_channels, dtype=dtype)
        elif normalizing_method == "fixed":
            offsets = cfg_unified.normalization.fixed.inputs_offsets
            variances = cfg_unified.normalization.fixed.inputs_variances
            normalizing_layer = normalizing_class(offsets, variances, dtype=dtype)
        elif normalizing_method == "automatic":
            normalizing_layer = normalizing_class(dtype=dtype)
        elif normalizing_method == "none":
            normalizing_layer = normalizing_class(dtype=dtype)
        else:
            raise ValueError(f"Unknown normalizing method: {normalizing_method}")

        if cfg_unified.network.pretrained:
            dir_path = get_pretrained_emulator_path(cfg, state)
            iceflow_model = load_model_from_path(dir_path, cfg_unified.inputs)
        else:
            warnings.warn("No pretrained emulator found. Starting from scratch.")

            nb_inputs = len(cfg_unified.inputs)
            nb_outputs = 2 * cfg_numerics.Nz
            architecture_name = cfg_unified.network.architecture

            # Get the function from the networks module
            if architecture_name in Architectures:
                architecture_class = Architectures[architecture_name]
                iceflow_model = architecture_class(cfg, nb_inputs, nb_outputs)

            else:
                raise ValueError(
                    f"Unknown network architecture: {architecture_name}. "
                    f"Available architectures: {Architectures.keys()}"
                )

        state.iceflow_model = iceflow_model
        state.iceflow_model.compile(
            jit_compile=False
        )  # not all architectures support jit_compile=True

        bcs = init_bcs(cfg, state, cfg.processes.iceflow.unified.bcs)

        return {
            "bcs": bcs,
            "normalizer": normalizing_layer,
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": cfg_unified.network.output_scale,
            "precision": cfg_numerics.precision,
        }
