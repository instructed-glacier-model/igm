#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Callable, Dict, Optional

from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status, nbit_at
from ...mappings import Mapping
from ...halt import Halt, InterfaceHalt


class InterfaceTrustRegion(InterfaceOptimizer):
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
            "iter_max": nbit_at(cfg_unified.nbit),
            "damping": cfg_unified.trust_region.damping,
            "print_cost": cfg_unified.display.print_cost,
            "print_cost_freq": cfg_unified.display.print_cost_freq,
            "precision": cfg_numerics.precision,
            "ord_grad_u": cfg_numerics.ord_grad_u,
            "ord_grad_theta": cfg_numerics.ord_grad_theta,
            "debug_mode": cfg_unified.network.debug_mode,
            "debug_freq": cfg_unified.network.debug_freq,
            "cg_max_iter": cfg_unified.trust_region.cg_max_iter,
            "cg_tol": cfg_unified.trust_region.cg_tol,
            "delta_init": cfg_unified.trust_region.delta_init,
            "delta_max": cfg_unified.trust_region.delta_max,
            "eta": cfg_unified.trust_region.eta,
        }

    @staticmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
        t: Optional[float] = None,
    ) -> bool:
        cfg_unified = cfg.processes.iceflow.unified

        if status == Status.INIT:
            iter_max = cfg_unified.nbit_init
        elif status == Status.WARM_UP:
            iter_max = cfg_unified.nbit_init
        elif status == Status.DEFAULT:
            iter_max = nbit_at(cfg_unified.nbit, t)
        elif status == Status.IDLE:
            return False
        else:
            raise ValueError(f"❌ Unknown optimizer status: <{status.name}>.")

        damping = tf.constant(cfg_unified.trust_region.damping, dtype=tf.float32)
        optimizer.update_parameters(iter_max=iter_max, damping=damping)

        return True
