#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), see LICENSE

from __future__ import annotations

from typing import Any, Dict, List

from omegaconf import DictConfig

from igm.common import State
from igm.processes.iceflow.unified.bcs.utils import init_bcs
from .interface import InterfaceMapping
from ..data_assimilation import MappingDataAssimilation, VariableSpec
from ..network import MappingNetwork
from ..transforms import TRANSFORMS


class InterfaceDataAssimilation(InterfaceMapping):
    """
    Reads Hydra config:
      data_assimilation:
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
            if transform not in TRANSFORMS.keys():
                raise ValueError(
                    f"❌ Unsupported transform '{transform}' for '{name}'."
                )
            lb = item.get("lower_bound", None)
            ub = item.get("upper_bound", None)
            # Normalize "None"/"null" strings if they appear
            if isinstance(lb, str) and lb.lower() in ("none", "null"):
                lb = None
            if isinstance(ub, str) and ub.lower() in ("none", "null"):
                ub = None
            mask = item.get("mask", None)
            if isinstance(mask, str) and mask.lower() in ("none", "null", ""):
                mask = None
            specs.append(
                VariableSpec(
                    name=name,
                    transform=transform,
                    lower_bound=lb,
                    upper_bound=ub,
                    mask=mask,
                )
            )
        return specs

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        bcs = cfg.processes.iceflow.unified.bcs
        variables = InterfaceDataAssimilation._parse_specs(cfg)

        # Get the existing mapping from state (should be set up before data assimilation)
        if not hasattr(state.iceflow, "mapping") or state.iceflow.mapping is None:
            raise ValueError(
                "❌ No base mapping found in state.iceflow.mapping. "
                "The main iceflow mapping must be initialized before data assimilation mapping."
            )

        # Get the existing cost_fn from the iceflow optimizer (should also be set up)
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

        precision = cfg.processes.iceflow.numerics.precision

        bcs = init_bcs(cfg, state, cfg.processes.iceflow.unified.bcs)
        fieldin_names = cfg.processes.iceflow.unified.inputs
        field_to_channel = {name: i for i, name in enumerate(fieldin_names)}

        return {
            "bcs": bcs,
            "network": base_mapping.network,
            "Nz": base_mapping.Nz,
            "output_scale": base_mapping.output_scale,
            "state": state,  # Still needed for initialization to read field values
            "variables": variables,
            "precision": precision,
            "field_to_channel": field_to_channel,
        }
