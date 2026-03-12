#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Callable, Dict

from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status
from ...mappings import Mapping
from ...halt import Halt, InterfaceHalt


class InterfaceCGNewton(InterfaceOptimizer):
    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:
        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics

        halt_args = InterfaceHalt.get_halt_args(cfg)
        halt = Halt(**halt_args)

        return {
            "cost_fn": cost_fn,
            "map": map,
            "halt": halt,
            "iter_max": cfg_unified.nbit,
            "damping": cfg_unified.hessian.damping,
            "print_cost": cfg_unified.display.print_cost,
            "print_cost_freq": cfg_unified.display.print_cost_freq,
            "precision": cfg_numerics.precision,
            "ord_grad_u": cfg_numerics.ord_grad_u,
            "line_search_method": cfg_unified.line_search,
            "ord_grad_theta": cfg_numerics.ord_grad_theta,
            "debug_mode": cfg_unified.network.debug_mode,
            "debug_freq": cfg_unified.network.debug_freq,
            "cg_max_iter": cfg_unified.cg_newton.cg_max_iter,
            "cg_tol": cfg_unified.cg_newton.cg_tol,
            "truncated": cfg_unified.cg_newton.truncated,
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

        damping = cfg_unified.hessian.damping
        optimizer.update_parameters(iter_max=iter_max, damping=damping)

        return True