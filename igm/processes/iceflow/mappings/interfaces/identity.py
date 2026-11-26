#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.iceflow.bcs.utils import init_bcs
from igm.utils.math.precision import normalize_precision

from .interface import InterfaceMapping


class InterfaceIdentity(InterfaceMapping):

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:

        Nx = state.thk.shape[1]
        Ny = state.thk.shape[0]
        cfg_numerics = cfg.processes.iceflow.numerics
        Nz = cfg_numerics.Nz

        dtype = normalize_precision(cfg_numerics.precision)

        U_guess = tf.zeros((1, Nz, Ny, Nx), dtype=dtype)
        V_guess = tf.zeros((1, Nz, Ny, Nx), dtype=dtype)

        bcs = init_bcs(cfg, state, cfg.processes.iceflow.unified.bcs)

        return {
            "bcs": bcs,
            "U_guess": U_guess,
            "V_guess": V_guess,
            "precision": cfg_numerics.precision,
        }
