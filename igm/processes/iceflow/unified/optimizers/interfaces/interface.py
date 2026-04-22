#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Tuple

from ...mappings import Mapping
from .. import Optimizer


class Status(Enum):
    INIT = auto()
    WARM_UP = auto()
    DEFAULT = auto()
    IDLE = auto()


def get_save_args(cfg: DictConfig, state=None) -> Dict[str, Any]:
    """Load save-related optimizer arguments from cfg, including optional reference velocity and V_s."""

    cfg_save = cfg.processes.iceflow.unified.save

    if not cfg_save.enabled:
        return {
            "save_cost_file": "",
            "vel_ref": None,
            "v_s": None,
            "save_vel_error_file": "",
        }

    save_cost_file = cfg_save.cost_file
    save_vel_error_file = cfg_save.vel_error_file
    vel_ref_file = cfg_save.vel_ref_file

    vel_ref: Optional[tf.Tensor] = None
    v_s: Optional[tf.Tensor] = None

    if vel_ref_file:
        from netCDF4 import Dataset

        nc = Dataset(vel_ref_file, "r")
        vel_ref = tf.constant(
            np.squeeze(nc.variables[cfg_save.vel_ref_name][:]).astype("float32")
        )
        nc.close()

    if state is not None:
        v_s = state.iceflow.discr_v.V_s

    return {
        "save_cost_file": save_cost_file,
        "vel_ref": vel_ref,
        "v_s": v_s,
        "save_vel_error_file": save_vel_error_file,
    }


class InterfaceOptimizer(ABC):

    @staticmethod
    @abstractmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        save_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "❌ The get_optimizer_args static method is not implemented."
        )

    @staticmethod
    @abstractmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
    ) -> bool:
        raise NotImplementedError(
            "❌ The set_optimizer_params static method is not implemented."
        )
