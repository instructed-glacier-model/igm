#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Dict

from igm.common.core import State
from .interface import InterfaceMapping


class InterfaceNodal(InterfaceMapping):

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:

        Nx = state.thk.shape[1]
        Ny = state.thk.shape[0]
        Nz = cfg.processes.iceflow.numerics.Nz

        U_guess = tf.zeros((1, Nz, Ny, Nx))
        V_guess = tf.zeros((1, Nz, Ny, Nx))

        return {
            "U_guess": U_guess,
            "V_guess": V_guess,
        }
