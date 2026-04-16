#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors – GNU GPL v3

"""
Extended InterfaceComposite with optional physics-informed preprocessing.

Drop-in replacement for the original composite interface.  When a sub-mapping
config contains a ``physics_features`` block the preprocessor is built and
composed with the sub-mapping's normalizer as its ``input_normalizer``.
Without that block, behaviour is identical to the original.
"""

import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict

from igm.common import State
from igm.processes.iceflow.unified.bcs.utils import init_bcs
from igm.processes.iceflow.unified.mappings.mapping import Mapping
from igm.processes.iceflow.unified.mappings.masks import interfaces_mask_gr
from .interface import InterfaceMapping

from igm.processes.iceflow.emulate.utils.architectures.physics_features import (
    _CLASSES as _PREPROC_CLASSES,
    build_preprocessor,
)


class _ComposedNorm(tf.keras.layers.Layer):
    """Chain a physics preprocessor with a downstream normalizer.

    ``call``        : preprocessor(x) → normalizer(...)
    ``compute_stats``: runs preprocessor first so the normalizer sees
                       physics-feature statistics, not raw-input statistics.
    ``set_stats``   : delegates to the inner normalizer.

    This lets the adaptive normalizer learn the mean/variance of the
    physics features without any manual unit bookkeeping.
    """

    def __init__(self, preprocessor, normalizer, **kw):
        super().__init__(**kw)
        self.preprocessor = preprocessor
        self.normalizer = normalizer

    def call(self, x, training=False):
        return self.normalizer(self.preprocessor(x), training=training)

    def compute_stats(self, x):
        return self.normalizer.compute_stats(self.preprocessor(x))

    def set_stats(self, m, v):
        self.normalizer.set_stats(m, v)


class InterfaceComposite(InterfaceMapping):

    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        cfg_unified = cfg.processes.iceflow.unified
        cfg_composite = cfg_unified.composite

        mapping_gr = InterfaceComposite._build_sub_mapping(cfg, state, cfg_composite.gr)
        mapping_fl = InterfaceComposite._build_sub_mapping(cfg, state, cfg_composite.fl)

        mask_gr_name = cfg_composite.mask_gr.method
        mask_gr_kwargs = interfaces_mask_gr[mask_gr_name](cfg, state, cfg_composite.mask_gr)
        bcs = init_bcs(cfg, state, cfg_unified.bcs)

        return dict(
            bcs=bcs, mapping_gr=mapping_gr, mapping_fl=mapping_fl,
            mask_gr_name=mask_gr_name, mask_gr_kwargs=mask_gr_kwargs,
            precision=cfg.processes.iceflow.numerics.precision,
        )

    @staticmethod
    def _build_sub_mapping(cfg: DictConfig, state: State,
                           cfg_sub: DictConfig) -> Mapping:
        """Build a sub-mapping, optionally wrapping it with a physics
        preprocessor composed with the sub-mapping's normalizer.

        When ``cfg_sub`` contains a ``physics_features`` block:

        1. Override ``inputs`` to a placeholder list of the correct
           length so the architecture is built with the right nb_inputs.
        2. Build the mapping normally via the standard interface (this
           creates an adaptive normalizer sized for N_feat channels).
        3. Compose the physics preprocessor with the existing normalizer
           into a ``_ComposedNorm`` and install it as ``input_normalizer``.
           The adaptive normalizer then learns statistics over the physics
           features rather than the raw input fields.
        """
        from igm.processes.iceflow.unified.mappings import Mappings, InterfaceMappings

        mapping_name = cfg_sub.mapping
        sub_dict = OmegaConf.to_container(cfg_sub, resolve=True)
        physics_cfg = sub_dict.pop("physics_features", None)

        # Determine nb_inputs from preprocessor output channels
        if physics_cfg is not None:
            n_feat = _PREPROC_CLASSES[physics_cfg["mode"]].NUM_FEATURES
            sub_dict["inputs"] = [f"_feat_{i}" for i in range(n_feat)]

        # ── merge sub-mapping overrides into full config ──
        cfg_merged = OmegaConf.to_container(cfg, resolve=True)
        iceflow = cfg_merged["processes"]["iceflow"]

        for key in ("network", "normalization", "inputs"):
            if key in sub_dict:
                iceflow["unified"][key] = sub_dict[key]

        # CNN variants read from emulator.network, FNO2 from unified.network.
        # Mirror to both paths so either architecture works.
        if "network" in sub_dict:
            iceflow.setdefault("emulator", {})["network"] = sub_dict["network"]

        iceflow["unified"]["bcs"] = []     # BCs applied at composite level
        cfg_merged = OmegaConf.create(cfg_merged)

        args = InterfaceMappings[mapping_name].get_mapping_args(cfg_merged, state)
        mapping = Mappings[mapping_name](**args)

        # ── attach physics preprocessor ──
        if physics_cfg is not None:
            preprocessor = build_preprocessor(
                mode=physics_cfg["mode"],
                full_input_names=list(cfg.processes.iceflow.unified.inputs),
                physics_cfg=physics_cfg,
                dx=float(state.dx),
                dy=float(getattr(state, "dy", state.dx)),
                precision=cfg.processes.iceflow.numerics.precision,
            )
            existing_norm = mapping.network.input_normalizer
            mapping.network.input_normalizer = _ComposedNorm(preprocessor, existing_norm)

        return mapping
