import tensorflow as tf

import warnings
from typing import Any, Dict
from omegaconf import DictConfig
from .interface import InterfaceMapping
from igm.common import State
from igm.processes.iceflow.emulate.utils.artifacts import load_emulator_artifact
from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.processes.iceflow.emulate.utils import NormalizationsDict
from igm.processes.iceflow.unified.mappings import Mappings
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

        Nz = int(cfg_numerics.Nz)

        if cfg_unified.network.pretrained:
            if cfg_unified.network.pretrained_path:
                dtype = normalize_precision(cfg_numerics.precision)
                artifact_dir = cfg_unified.network.pretrained_path
                tf.keras.mixed_precision.set_global_policy("float64" if tf.as_dtype(dtype) == tf.float64 else "float32")
                iceflow_model, _manifest = load_emulator_artifact(artifact_dir, cfg)
            else:
                warnings.warn("Loading old format pretrained emulator. This may not be supported in future IGM versions.")
                dir_path = get_pretrained_emulator_path(cfg, state)
                iceflow_model = load_model_from_path(dir_path, cfg_unified.inputs)
        else:
            warnings.warn("No pretrained emulator selected. Starting from scratch.")

            nb_inputs = len(cfg_unified.inputs)
            nb_outputs = 2 * Nz

            arch_name = cfg_unified.network.architecture
            if arch_name not in Architectures:
                raise ValueError(f"Unknown network architecture: {arch_name}. Available: {Architectures.keys()}")

            kwargs = {}
            if arch_name == "DahuNet" and hasattr(cfg_unified.network, "dahunet"):
                cfg_dahu = cfg_unified.network.dahunet
                if hasattr(cfg_dahu, "backend"):
                    kwargs["backend"] = str(cfg_dahu.backend)
                if hasattr(cfg_dahu, "features"):
                    kwargs["features"] = tuple(cfg_dahu.features)

            iceflow_model = Architectures[arch_name](cfg, nb_inputs, nb_outputs, **kwargs)

            # Build normalizer and attach to model
            if "pretraining" in cfg.processes.keys():        
                iceflow_model.input_normalizer = None # this is handled in pretraining process
            else:
                # Inference / non-pretraining: keep config-driven behavior for now
                method = cfg_unified.normalization.method
                normalizing_class = NormalizationsDict[method]

                if method == "adaptive":
                    normalizing_layer = normalizing_class(nb_inputs)
                elif method == "fixed":
                    offsets = cfg_unified.normalization.fixed.inputs_offsets
                    variances = cfg_unified.normalization.fixed.inputs_variances
                    normalizing_layer = normalizing_class(offsets, variances)
                elif method in ("automatic", "none"):
                    normalizing_layer = normalizing_class()
                else:
                    raise ValueError(f"Unknown normalizing method: {method}")

                iceflow_model.input_normalizer = normalizing_layer


        state.iceflow_model = iceflow_model
        state.iceflow_model.compile(jit_compile=False)

        bcs = init_bcs(cfg, state, cfg_unified.bcs)

        return {
            "bcs": bcs,
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": cfg_unified.network.output_scale,
            "precision": cfg_numerics.precision,
        }
