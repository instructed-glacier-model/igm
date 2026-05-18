#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), see LICENSE

from __future__ import annotations

from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from igm.common import State
from igm.processes.iceflow.unified.bcs.utils import init_bcs

from .interface import InterfaceMapping
from ..data_assimilation import VariableSpec
from ..network import MappingNetwork
from ..transforms import TRANSFORMS


def _optional_state_name(value: Any) -> Optional[str]:
    """
    Convert an optional config value into a literal state-field name.

    None means "not supplied" and lets the mapping apply its own default. Any
    non-None value is treated as a state attribute name after str().
    """
    return None if value is None else str(value)


class InterfaceDataAssimilation(InterfaceMapping):
    """
    Reads Hydra config:
      data_assimilation_SR:
        variables:
          - { name: thk,        transform: identity }
          - { name: slidingco,  transform: log10 }

    and produces the kwargs for `MappingDataAssimilation`.
    """

    @staticmethod
    def _parse_specs(cfg: DictConfig) -> List[VariableSpec]:
        specs = []
        for item in cfg.processes.data_assimilation_SR.variables:
            name = str(item["name"])
            transform = str(item.get("transform", "identity")).lower()
            if transform not in TRANSFORMS:
                raise ValueError(f"❌ Unsupported transform '{transform}' for '{name}'.")

            specs.append(
                VariableSpec(
                    name=name,
                    transform=transform,
                    lower_bound=item.get("lower_bound", None),
                    upper_bound=item.get("upper_bound", None),
                    mask=_optional_state_name(item.get("mask", None)),
                )
            )
        return specs

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        variables = InterfaceDataAssimilation._parse_specs(cfg)

        if not hasattr(state.iceflow, "mapping") or state.iceflow.mapping is None:
            raise ValueError(
                "❌ No base mapping found in state.iceflow.mapping. "
                "The main iceflow mapping must be initialized before data assimilation mapping."
            )

        if not hasattr(state.iceflow, "optimizer") or state.iceflow.optimizer is None:
            raise ValueError(
                "❌ No optimizer found in state.iceflow.optimizer. "
                "The main iceflow optimizer must be initialized before data assimilation mapping."
            )

        base_mapping = state.iceflow.mapping
        if not isinstance(base_mapping, MappingNetwork):
            raise TypeError(
                "❌ Data assimilation currently expects a MappingNetwork as the base mapping."
            )

        fieldin_names = cfg.processes.iceflow.unified.inputs
        field_to_channel = {name: i for i, name in enumerate(fieldin_names)}

        return {
            "bcs": init_bcs(cfg, state, cfg.processes.iceflow.unified.bcs),
            "network": base_mapping.network,
            "Nz": base_mapping.Nz,
            "output_scale": base_mapping.output_scale,
            "state": state,
            "variables": variables,
            "precision": cfg.processes.iceflow.numerics.precision,
            "field_to_channel": field_to_channel,
        }
