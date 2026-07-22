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
            "cg_tol_relative": cfg_unified.cg_newton.cg_tol_relative,
            "warm_start": cfg_unified.cg_newton.warm_start,
            "truncated": cfg_unified.cg_newton.truncated,
            "hvp_mode": cfg_unified.cg_newton.hvp_mode,
            "damping": cfg_unified.cg_newton.damping,
            "precond": cfg_unified.cg_newton.precond,
            "preconditioner": cfg_unified.cg_newton.preconditioner,
            "precond_samples": cfg_unified.cg_newton.precond_samples,
            "precond_floor": cfg_unified.cg_newton.precond_floor,
            "input_names": tuple(cfg_unified.inputs),
            "mapping_name": cfg_unified.mapping,
            "basis_horizontal": cfg_numerics.basis_horizontal,
            "basis_vertical": cfg_numerics.basis_vertical,
            "sliding_law": cfg.processes.iceflow.physics.sliding.law,
            "sliding_exponent": cfg.processes.iceflow.physics.sliding.exponent,
            "sliding_regularization": cfg.processes.iceflow.physics.sliding.regularization,
            "sliding_u_ref": cfg.processes.iceflow.physics.sliding.u_ref,
            "sliding_use_mask_gr": cfg.processes.iceflow.physics.sliding.use_mask_gr,
            "rho_ratio": (
                cfg.processes.iceflow.physics.water_density
                / cfg.processes.iceflow.physics.ice_density
            ),
            "viscosity_exponent": cfg.processes.iceflow.physics.viscosity.exponent,
            "viscosity_regularization": (
                cfg.processes.iceflow.physics.viscosity.regularization
            ),
            "min_sr": cfg.processes.iceflow.physics.min_sr,
            "max_sr": cfg.processes.iceflow.physics.max_sr,
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

        damping = cfg_unified.cg_newton.damping
        optimizer.update_parameters(iter_max=iter_max, damping=damping)

        return True
