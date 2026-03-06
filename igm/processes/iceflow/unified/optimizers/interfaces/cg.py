# interface_cg.py
import tensorflow as tf
from omegaconf import DictConfig
from typing import Any, Callable, Dict, Optional
from ...mappings import Mapping, MappingDataAssimilation, MappingCombinedDataAssimilation
from ..optimizer import Optimizer
from .interface import InterfaceOptimizer, Status
from ..cg import OptimizerCG


class InterfaceCG(InterfaceOptimizer):
    @staticmethod
    def get_optimizer_args(cfg: DictConfig,
                           cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
                           map: Mapping,
                           save_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        u = cfg.processes.iceflow.unified
        precision = cfg.processes.iceflow.numerics.precision

        if isinstance(map, MappingDataAssimilation) or isinstance(map, MappingCombinedDataAssimilation):
            nbit = cfg.processes.SR_DA.optimization.nbitmax
        else:
            nbit = u.nbit
        return {
            "cost_fn": cost_fn,
            "map": map,
            "iter_max": nbit,
            "alpha_min": u.lbfgs.alpha_min,   # reuse same key for simplicity
            "line_search_method": u.line_search,
            "print_cost": u.print_cost,
            "print_cost_freq": u.print_cost_freq,
            "precision": precision,
            "variant": "PR+",
            "restart_every": 50,
        }

    @staticmethod
    def set_optimizer_params(cfg: DictConfig, status: Status, optimizer: Optimizer) -> bool:
        u = cfg.processes.iceflow.unified
        if status in (Status.INIT, Status.WARM_UP):
            iter_max = u.nbit_init
            alpha_min = u.lbfgs.alpha_min
        elif status == Status.DEFAULT:
            iter_max = u.nbit
            alpha_min = u.lbfgs.alpha_min
        else:
            return False
        optimizer.update_parameters(iter_max=iter_max, alpha_min=alpha_min)
        return True
