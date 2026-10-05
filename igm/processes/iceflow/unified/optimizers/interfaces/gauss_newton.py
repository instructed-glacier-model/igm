#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Configuration interface for network-weight Gauss--Newton optimization."""

from typing import Any, Callable, Dict, Optional

import tensorflow as tf
from omegaconf import DictConfig

from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status, nbit_at
from ...halt import Halt, InterfaceHalt
from ...mappings import Mapping
from ...operators import GaussNewtonOperator
from ...preconditioners import (
    IdentityPreconditioner,
    NystromPreconditioner,
    Preconditioner,
)


class InterfaceGaussNewton(InterfaceOptimizer):
    @staticmethod
    def _build_operator(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> GaussNewtonOperator:
        unified = cfg.processes.iceflow.unified
        numerics = cfg.processes.iceflow.numerics
        gauss_newton = unified.gauss_newton
        if getattr(map, "name", "") != "network":
            raise ValueError(
                "gauss_newton is a network-weight optimizer and requires "
                "unified.mapping=network."
            )

        return GaussNewtonOperator(
            cost_fn,
            map,
            precision=numerics.precision,
            hu_mode=gauss_newton.hu_mode,
            basis_vertical=numerics.basis_vertical,
            probe_mode=gauss_newton.probe_mode,
            verify_stencil=gauss_newton.hvp_verify,
        )

    @staticmethod
    def _build_preconditioner(
        cfg: DictConfig,
        map: Mapping,
        operator: GaussNewtonOperator,
    ) -> tuple[Preconditioner, str]:
        unified = cfg.processes.iceflow.unified
        numerics = cfg.processes.iceflow.numerics
        gauss_newton = unified.gauss_newton
        requested_preconditioner = str(gauss_newton.preconditioner).lower()
        preconditioner_name = requested_preconditioner
        if requested_preconditioner == "auto":
            architecture = str(unified.network.architecture).lower()
            preconditioner_name = "nystrom" if architecture == "dahunet" else "identity"
            if gauss_newton.print_timing:
                print(
                    f"[gauss_newton] preconditioner auto -> {preconditioner_name} "
                    f"(architecture={architecture})",
                    flush=True,
                )
        if preconditioner_name == "identity":
            preconditioner = IdentityPreconditioner()
        elif preconditioner_name == "nystrom":
            preconditioner = NystromPreconditioner(
                map,
                rank=gauss_newton.nystrom.rank,
                precision=numerics.precision,
                seed=gauss_newton.nystrom.seed,
                sketch_memory_gb=gauss_newton.nystrom.sketch_memory_gb,
                minimum_rank=gauss_newton.nystrom.minimum_rank,
                print_timing=gauss_newton.nystrom.print_timing,
            )
        else:
            raise ValueError(
                "Unknown gauss_newton.preconditioner: "
                f"<{requested_preconditioner}>. "
                "Use 'auto', 'identity', or 'nystrom'."
            )
        if preconditioner.needs_operator:
            preconditioner.set_operator(operator)
        return preconditioner, preconditioner_name

    @staticmethod
    def get_optimizer_args(
        cfg: DictConfig,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
    ) -> Dict[str, Any]:
        unified = cfg.processes.iceflow.unified
        numerics = cfg.processes.iceflow.numerics
        gauss_newton = unified.gauss_newton
        operator = InterfaceGaussNewton._build_operator(cfg, cost_fn, map)
        preconditioner, preconditioner_name = (
            InterfaceGaussNewton._build_preconditioner(cfg, map, operator)
        )

        return {
            "cost_fn": cost_fn,
            "map": map,
            "halt": Halt(**InterfaceHalt.get_halt_args(cfg)),
            "iter_max": nbit_at(unified.nbit),
            "print_cost": unified.display.print_cost,
            "print_cost_freq": unified.display.print_cost_freq,
            "precision": numerics.precision,
            "ord_grad_u": numerics.ord_grad_u,
            "ord_grad_theta": numerics.ord_grad_theta,
            "print_timing": gauss_newton.print_timing,
            "debug_mode": unified.network.debug_mode,
            "debug_freq": unified.network.debug_freq,
            "cg_max_iter": gauss_newton.cg_max_iter,
            "operator": operator,
            "preconditioner_obj": preconditioner,
            "preconditioner": preconditioner_name,
            "precond_update_freq": gauss_newton.precond_update_freq,
            "forcing_eta_min": gauss_newton.forcing_eta_min,
            "forcing_eta_max": gauss_newton.forcing_eta_max,
            "forcing_gamma": gauss_newton.forcing_gamma,
            "forcing_power": gauss_newton.forcing_power,
            "damping_rel_init": gauss_newton.damping_rel_init,
            "damping_rel_min": gauss_newton.damping_rel_min,
            "damping_rel_max": gauss_newton.damping_rel_max,
            "lm_ratio_low": gauss_newton.lm_ratio_low,
            "lm_ratio_high": gauss_newton.lm_ratio_high,
            "damping_down": gauss_newton.damping_down,
            "damping_up": gauss_newton.damping_up,
            "solve_retries": gauss_newton.solve_retries,
            "power_iterations": gauss_newton.power_iterations,
            "armijo_c1": gauss_newton.armijo_c1,
            "armijo_contraction": gauss_newton.armijo_contraction,
            "line_search_max_iter": gauss_newton.line_search_max_iter,
        }

    @staticmethod
    def set_optimizer_params(
        cfg: DictConfig,
        status: Status,
        optimizer: Optimizer,
        t: Optional[float] = None,
    ) -> bool:
        unified = cfg.processes.iceflow.unified
        if status in (Status.INIT, Status.WARM_UP):
            iter_max = unified.nbit_init
        elif status == Status.DEFAULT:
            iter_max = nbit_at(unified.nbit, t)
        elif status == Status.IDLE:
            return False
        else:
            raise ValueError(f"Unknown optimizer status: <{status.name}>.")
        optimizer.update_parameters(iter_max=iter_max)
        return int(iter_max) > 0
