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


class InterfaceMuon(InterfaceOptimizer):

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:
        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics

        halt = Halt(**InterfaceHalt.get_halt_args(cfg))

        return {
            "cost_fn": cost_fn,
            "map": map,
            "halt": halt,
            "lr": cfg_unified.muon.lr,
            "momentum": cfg_unified.muon.momentum,
            "ns_steps": cfg_unified.muon.ns_steps,
            "lr_1d": cfg_unified.muon.lr_1d,
            "iter_max": cfg_unified.nbit,
            "print_cost": cfg_unified.display.print_cost,
            "print_cost_freq": cfg_unified.display.print_cost_freq,
            "precision": cfg_numerics.precision,
            "ord_grad_u": cfg_numerics.ord_grad_u,
            "ord_grad_theta": cfg_numerics.ord_grad_theta,
        }

    @staticmethod
    def set_optimizer_params(
        cfg: DictConfig, status: Status, optimizer: Optimizer
    ) -> bool:
        cfg_unified = cfg.processes.iceflow.unified

        lr = cfg_unified.muon.lr

        if status == Status.INIT or status == Status.WARM_UP:
            iter_max = cfg_unified.nbit_init
        elif status == Status.DEFAULT:
            iter_max = cfg_unified.nbit
        elif status == Status.IDLE:
            return False
        else:
            iter_max = cfg_unified.nbit

        optimizer.update_parameters(iter_max=iter_max, lr=lr)

        return iter_max > 0
