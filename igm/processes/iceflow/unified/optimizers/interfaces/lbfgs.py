#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Callable, Dict, Optional

from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status
from ...mappings import Mapping
from ...halt import Halt, InterfaceHalt


class InterfaceLBFGS(InterfaceOptimizer):

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        save_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics

        halt_args = InterfaceHalt.get_halt_args(cfg)
        halt = Halt(**halt_args)

        return {
            "cost_fn": cost_fn,
            "map": map,
            "halt": halt,
            "line_search_method": cfg_unified.line_search,
            "iter_max": cfg_unified.nbit,
            "alpha_min": cfg_unified.lbfgs.alpha_min,
            "memory": cfg_unified.lbfgs.memory,
            "print_cost": cfg_unified.display.print_cost,
            "print_cost_freq": cfg_unified.display.print_cost_freq,
            "precision": cfg_numerics.precision,
            "ord_grad_u": cfg_numerics.ord_grad_u,
            "ord_grad_theta": cfg_numerics.ord_grad_theta,
            "debug_mode": cfg_unified.network.debug_mode,
            "debug_freq": cfg_unified.network.debug_freq,
            **(save_args or {}),
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
