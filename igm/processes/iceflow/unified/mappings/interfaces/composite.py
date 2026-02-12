#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict

from igm.common import State
from igm.processes.iceflow.unified.bcs.utils import init_bcs
from igm.processes.iceflow.unified.mappings.mapping import Mapping
from igm.processes.iceflow.unified.mappings.masks import interfaces_mask_gr
from .interface import InterfaceMapping


class InterfaceComposite(InterfaceMapping):

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:

        cfg_unified = cfg.processes.iceflow.unified
        cfg_numerics = cfg.processes.iceflow.numerics

        cfg_composite = cfg_unified.composite

        mapping_gr = InterfaceComposite._build_sub_mapping(cfg, state, cfg_composite.gr)
        mapping_fl = InterfaceComposite._build_sub_mapping(cfg, state, cfg_composite.fl)

        mask_gr_name = cfg_composite.mask_gr.method
        mask_gr_kwargs = interfaces_mask_gr[mask_gr_name](
            cfg, state, cfg_composite.mask_gr
        )

        bcs = init_bcs(cfg, state, cfg_unified.bcs)

        return {
            "bcs": bcs,
            "mapping_gr": mapping_gr,
            "mapping_fl": mapping_fl,
            "mask_gr_name": mask_gr_name,
            "mask_gr_kwargs": mask_gr_kwargs,
            "precision": cfg_numerics.precision,
        }

    @staticmethod
    def _build_sub_mapping(
        cfg: DictConfig,
        state: State,
        cfg_sub: DictConfig,
    ) -> Mapping:
        """Build a sub-mapping by overlaying its config onto the base
        unified config.  BCs are cleared — they are applied at the
        composite level instead."""

        # Lazy imports to avoid circular dependency with parent __init__
        from igm.processes.iceflow.unified.mappings import Mappings, InterfaceMappings

        mapping_name = cfg_sub.mapping

        cfg_merged = OmegaConf.to_container(cfg, resolve=True)
        sub_dict = OmegaConf.to_container(cfg_sub, resolve=True)

        unified = cfg_merged["processes"]["iceflow"]["unified"]
        for key in ("network", "normalization", "inputs"):
            if key in sub_dict:
                unified[key] = sub_dict[key]

        unified["bcs"] = []

        cfg_merged = OmegaConf.create(cfg_merged)

        args = InterfaceMappings[mapping_name].get_mapping_args(cfg_merged, state)
        return Mappings[mapping_name](**args)
