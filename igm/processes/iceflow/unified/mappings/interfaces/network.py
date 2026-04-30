import tensorflow as tf

import warnings
from typing import Any, Dict, Optional
from omegaconf import DictConfig

from .interface import InterfaceMapping
from igm.common import State
from igm.processes.iceflow.emulate.utils.artifacts import (
    build_emulator_from_cfg,
    load_emulator_artifact,
    _attach_config_normalizer
)
from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.processes.iceflow.unified.bcs.utils import init_bcs
from igm.utils.math.precision import normalize_precision
from igm.processes.iceflow.emulate.utils.misc import (
    get_pretrained_emulator_path,
    load_model_from_path,
)


class InterfaceNetwork(InterfaceMapping):
    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_unified = cfg.processes.iceflow.unified

        mapping_output_scale = float(cfg_unified.network.output_scale)
        manifest = None

        if cfg_unified.network.pretrained:
            if cfg_unified.network.old_format:
                warnings.warn(
                    "Loading old-format pretrained emulator. "
                    "This path is kept only for legacy compatibility."
                )
                dir_path = get_pretrained_emulator_path(cfg, state)
                iceflow_model = load_model_from_path(dir_path, cfg_unified.inputs)
            else:
                dtype = normalize_precision(cfg_numerics.precision)
                artifact_dir = cfg_unified.network.pretrained_path
                tf.keras.mixed_precision.set_global_policy(
                    "float64" if tf.as_dtype(dtype) == tf.float64 else "float32"
                )
                iceflow_model, manifest = load_emulator_artifact(artifact_dir, cfg)
                mapping_output_scale = float(manifest.output_scale)

        else:
            warnings.warn("No pretrained emulator selected. Starting from scratch.")

            if cfg_unified.network.old_format:
                nb_inputs = len(cfg_unified.inputs)
                nb_outputs = 2 * int(cfg_numerics.Nz)

                arch_name = str(cfg_unified.network.architecture)
                if arch_name not in Architectures:
                    raise ValueError(
                        f"Unknown network architecture: {arch_name}. "
                        f"Available: {list(Architectures.keys())}"
                    )

                iceflow_model = Architectures[arch_name](cfg, nb_inputs, nb_outputs)
                _attach_config_normalizer(cfg, iceflow_model)

            else:
                iceflow_model = build_emulator_from_cfg(
                    cfg,
                    attach_config_normalizer=not cfg.processes.iceflow.do_pretraining,
                )

        state.iceflow_model = iceflow_model
        state.iceflow_manifest = manifest
        state.iceflow_model.compile(jit_compile=False)

        bcs = init_bcs(cfg, state, cfg_unified.bcs)

        return {
            "bcs": bcs,
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": mapping_output_scale,
            "precision": cfg_numerics.precision,
        }
