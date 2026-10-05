#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig, ListConfig
from typing import List, Dict, Any

from .criteria import Criteria, Criterion
from .metrics import Metrics


class InterfaceHalt:
    """Interface for configuring halting criteria for optimization."""

    @staticmethod
    def get_halt_args(cfg: DictConfig) -> Dict[str, Any]:
        """Extract halting arguments from configuration."""
        cfg_halt = cfg.processes.iceflow.unified.halt
        cfg_numerics = cfg.processes.iceflow.numerics

        crit_success = InterfaceHalt._create_crit_list(
            cfg_halt.success, cfg_halt, cfg_numerics
        )
        crit_failure = InterfaceHalt._create_crit_list(
            cfg_halt.failure, cfg_halt, cfg_numerics
        )

        return {
            "crit_success": crit_success,
            "crit_failure": crit_failure,
            "freq": cfg_halt.freq,
            "dtype": cfg_numerics.precision,
            "raise_on_failure": cfg_halt.raise_on_failure,
        }

    @staticmethod
    def _create_crit_list(
        list_crit: ListConfig, cfg_halt: DictConfig, cfg_numerics: DictConfig
    ) -> List[Criterion]:
        """Create list of criterion objects from configurations."""

        crit_list = []

        for item in list_crit:
            if "criterion" not in item or "metric" not in item:
                raise ValueError("❌ Each rule must contain 'criterion' and 'metric'.")

            crit_name = item["criterion"]
            metric_name = item["metric"]

            default_metric_args = (
                dict(cfg_halt.metrics[metric_name].items())
                if metric_name in cfg_halt.metrics
                else {}
            )
            default_crit_args = (
                dict(cfg_halt.criteria[crit_name].items())
                if crit_name in cfg_halt.criteria
                else {}
            )

            override_metric_args = (
                dict(item[metric_name].items()) if metric_name in item else {}
            )
            override_crit_args = (
                dict(item[crit_name].items()) if crit_name in item else {}
            )

            metric_args = {**default_metric_args, **override_metric_args}
            crit_args = {**default_crit_args, **override_crit_args}

            metric_class = Metrics[metric_name]
            metric = metric_class(**metric_args)

            crit_class = Criteria[crit_name]
            crit = crit_class(metric=metric, dtype=cfg_numerics.precision, **crit_args)
            crit_list.append(crit)

        return crit_list

    @staticmethod
    def _get_metric_args(
        default_name: str,
        override_name: str,
        default_cfg: DictConfig,
        override_cfg: DictConfig,
    ) -> Dict[str, Any]:
        """Merge default and override metric arguments from configurations."""
        default_args = {}
        if default_name in default_cfg:
            default_args = dict(default_cfg[default_name].items())

        override_args = {}
        for name in (override_name, default_name):
            if hasattr(override_cfg, name):
                override_args = dict(getattr(override_cfg, name).items())
                break

        return {**default_args, **override_args}

    @staticmethod
    def _get_crit_args(
        default_name: str,
        override_name: str,
        default_cfg: DictConfig,
        override_cfg: DictConfig,
    ) -> Dict[str, Any]:
        """Merge default and override criterion arguments from configurations."""
        default_args = {}
        if default_name in default_cfg:
            default_args = dict(default_cfg[default_name].items())

        override_args = {}
        for name in (override_name, default_name):
            if hasattr(override_cfg, name):
                override_args = dict(getattr(override_cfg, name).items())
                break

        return {**default_args, **override_args}
