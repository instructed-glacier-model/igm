#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Callable, Dict, Optional

from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status, nbit_at
from ...mappings import Mapping, MappingDataAssimilation
from ...halt import Halt, InterfaceHalt


class InterfaceSSESOAP(InterfaceOptimizer):
    """
    Config block expected under ``processes.iceflow.unified.ss_esoap``:

        lr: 3.0e-4            # base learning rate
        lr_init: 1.0e-3       # learning rate during INIT / WARM_UP
        beta1: 0.9            # momentum
        beta2: 0.95           # second moment + Kronecker factor decay
        eps: 1.0e-8
        tau_trigger: 0.2      # off-diagonal mass threshold for a basis update
        check_freq: 1         # iterations between two trigger checks (I_check)
        warmup: 0             # iterations before the trigger is armed (T_warm)
        tau_min: 0.1          # lower clip on the self-scaling factor
        self_scaling: true    # set false for the adaptive-basis-SOAP ablation
        damping: 1.0e-8
        relative_damping: 1.0e-6  # extra ridge before eigh, relative to trace(A) / m (0: off)
        weight_decay: 0.0
        lr_drop_iter: -1      # disabled; otherwise drop once at this iteration
        lr_drop_factor: 1.0   # multiplier applied after lr_drop_iter
        lr_auto_drop_patience: 0  # disabled; automatically choose drop time
        lr_auto_drop_warmup: 500
        lr_auto_drop_rel_improvement: 0.01

    Defaults follow Table 6 of arXiv:2608.29448.  On wide layers the O(m^3)
    trigger check dominates; raising ``check_freq`` to 5-10 amortises it with
    little loss, since the paper's default I_check = 1 is tuned for PINN-sized
    layers (width <= 256).
    """

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:

        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_opt = cfg_unified.ss_esoap

        if isinstance(map, MappingDataAssimilation):
            lr = cfg.assimilations.field_inversion.optimization.learning_rate
        else:
            lr = cfg_opt.lr

        halt_args = InterfaceHalt.get_halt_args(cfg)
        halt = Halt(**halt_args)

        return {
            "cost_fn": cost_fn,
            "map": map,
            "halt": halt,
            "lr": lr,
            "beta1": cfg_opt.beta1,
            "beta2": cfg_opt.beta2,
            "eps": cfg_opt.eps,
            "tau_trigger": cfg_opt.tau_trigger,
            "check_freq": cfg_opt.check_freq,
            "warmup": cfg_opt.warmup,
            "tau_min": cfg_opt.tau_min,
            "self_scaling": cfg_opt.self_scaling,
            "damping": cfg_opt.damping,
            "relative_damping": cfg_opt.relative_damping,
            "weight_decay": cfg_opt.weight_decay,
            "lr_drop_iter": cfg_opt.lr_drop_iter,
            "lr_drop_factor": cfg_opt.lr_drop_factor,
            "lr_auto_drop_patience": cfg_opt.lr_auto_drop_patience,
            "lr_auto_drop_warmup": cfg_opt.lr_auto_drop_warmup,
            "lr_auto_drop_rel_improvement": cfg_opt.lr_auto_drop_rel_improvement,
            "iter_max": nbit_at(cfg_unified.nbit),
            "print_cost": cfg_unified.display.print_cost,
            "print_cost_freq": cfg_unified.display.print_cost_freq,
            "precision": cfg_numerics.precision,
            "ord_grad_u": cfg_numerics.ord_grad_u,
            "ord_grad_theta": cfg_numerics.ord_grad_theta,
            "debug_mode": cfg_unified.network.debug_mode,
            "debug_freq": cfg_unified.network.debug_freq,
            "batch_size": cfg_unified.data_preparation.patches_per_batch,
        }

    @staticmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
        t: Optional[float] = None,
    ) -> bool:

        cfg_unified = cfg.processes.iceflow.unified
        cfg_opt = cfg_unified.ss_esoap

        if status == Status.INIT:
            iter_max = cfg_unified.nbit_init
            lr = cfg_opt.lr_init
        elif status == Status.WARM_UP:
            iter_max = cfg_unified.nbit_init
            lr = cfg_opt.lr_init
        elif status == Status.DEFAULT:
            iter_max = nbit_at(cfg_unified.nbit, t)
            lr = cfg_opt.lr
        elif status == Status.IDLE:
            return False
        else:
            raise ValueError(f"❌ Unknown optimizer status: <{status.name}>.")

        optimizer.update_parameters(iter_max=iter_max, lr=lr)

        return iter_max > 0
