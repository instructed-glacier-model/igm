import tensorflow as tf

import warnings
from typing import Any, Dict
from omegaconf import DictConfig

from .interface import InterfaceMapping
from igm.common import State
from igm.processes.iceflow.emulate.utils import NormalizationsDict
from igm.processes.iceflow.emulate.utils.artifacts import load_emulator_artifact
from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.processes.iceflow.unified.bcs.utils import init_bcs
from igm.utils.math.precision import normalize_precision
from igm.processes.iceflow.emulate.utils.misc import (
    get_pretrained_emulator_path,
    load_model_from_path,
)


def mapping_args_for_model(
    cfg: DictConfig, state: State, model: tf.keras.Model
) -> Dict[str, Any]:
    """Pack the network-mapping kwargs around an already-constructed model.

    Used by the pretraining-resume path, which loads the model from and
    just needs the surrounding BCs / Nz / output_scale / precision.
    """
    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_unified = cfg.processes.iceflow.unified

    state.iceflow_model = model
    bcs = init_bcs(cfg, state, cfg_unified.bcs)

    return {
        "bcs": bcs,
        "network": state.iceflow_model,
        "Nz": cfg_numerics.Nz,
        "output_scale": float(cfg_unified.network.output_scale),
        "precision": cfg_numerics.precision,
    }


class InterfaceNetwork(InterfaceMapping):
    @staticmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_unified = cfg.processes.iceflow.unified

        if cfg_unified.network.pretrained:
            if cfg_unified.network.pretrained_path:
                dtype = normalize_precision(cfg_numerics.precision)
                # might be worth breaking this out into a util with clear message to indicate a swap in precision
                tf.keras.mixed_precision.set_global_policy(
                    "float64" if tf.as_dtype(dtype) == tf.float64 else "float32"
                )
                iceflow_model = load_emulator_artifact(
                    cfg_unified.network.pretrained_path
                )
            else:
                warnings.warn(
                    "Loading old-format pretrained emulator. "
                    "This path is kept only for legacy compatibility."
                )
                dir_path = get_pretrained_emulator_path(cfg, state)
                iceflow_model = load_model_from_path(dir_path, cfg_unified.inputs)
        else:
            # should this still be a warning? Maybe we just need to make the themed
            # message very clear... it's perfectly valid after all.
            warnings.warn("No pretrained emulator selected. Starting from scratch.")

            nb_inputs = len(cfg_unified.inputs)
            nb_outputs = 2 * int(cfg_numerics.Nz)

            arch_name = str(cfg_unified.network.architecture)
            if arch_name not in Architectures:
                raise ValueError(
                    f"Unknown network architecture: {arch_name}. "
                    f"Available: {list(Architectures.keys())}"
                )

            iceflow_model = Architectures[arch_name](cfg, nb_inputs, nb_outputs)

            # Attach the input normalizer. During pretraining the trainer
            # supplies its own; otherwise read the cfg.
            if "pretraining" in cfg.processes.keys():
                iceflow_model.input_normalizer = None
            else:
                method = str(cfg_unified.normalization.method)
                if method not in NormalizationsDict:
                    raise ValueError(f"Unknown normalizing method: {method}")
                normalizing_class = NormalizationsDict[method]

                if method == "adaptive":
                    normalizing_layer = normalizing_class(nb_inputs)
                elif method == "fixed":
                    normalizing_layer = normalizing_class(
                        cfg_unified.normalization.fixed.inputs_offsets,
                        cfg_unified.normalization.fixed.inputs_variances,
                    )
                else:
                    normalizing_layer = normalizing_class()

                iceflow_model.input_normalizer = normalizing_layer

        state.iceflow_model = iceflow_model

        bcs = init_bcs(cfg, state, cfg_unified.bcs)

        return {
            "bcs": bcs,
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": float(cfg_unified.network.output_scale),
            "precision": cfg_numerics.precision,
        }
