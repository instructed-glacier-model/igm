#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Callable, Dict

from ..mappings import Mapping
from .optimizer import Optimizer
from .interface import InterfaceOptimizer, Status


class InterfaceLBFGS(InterfaceOptimizer):

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:

        cfg_unified = cfg.processes.iceflow.unified

        return {
            "cost_fn": cost_fn,
            "map": map,
            "iter_max": cfg_unified.nbit,
            "alpha_min": cfg_unified.lbfgs.alpha_min,
            "memory": cfg_unified.lbfgs.memory,
            "print_cost": cfg_unified.print_cost,
        }

    @staticmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
    ) -> bool:

        cfg_unified = cfg.processes.iceflow.unified

        if status == Status.INIT:
            iter_max = cfg_unified.nbit_init
        elif status == Status.WARM_UP:
            iter_max = cfg_unified.nbit_init
        elif status == Status.DEFAULT:
            iter_max = cfg_unified.nbit
        elif status == Status.IDLE:
            return False
        else:
            raise ValueError(f"❌ Unknown optimizer status: <{status.name}>.")

        alpha_min = cfg_unified.lbfgs.alpha_min

        optimizer.update_parameters(iter_max=iter_max, alpha_min=alpha_min)

        return True
