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


def _build_cfg_model(cfg: DictConfig, attach_normalizer: bool) -> tf.keras.Model:
    """
    Build a fresh architecture from cfg for the non-pretrained path.

    When `attach_normalizer` is True, a cfg-driven normalization layer
    (FixedAffineLayer / AdaptiveAffineLayer / StandardizationLayer /
    IdentityTransformation) is attached. When False, the caller (typically
    pretraining) supplies its own normalizer.
    """
    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_unified = cfg.processes.iceflow.unified
    cfg_emulator = cfg.processes.iceflow.emulator

    arch_name = str(cfg_unified.network.architecture)
    if arch_name not in Architectures:
        raise ValueError(
            f"Unknown network architecture: {arch_name}. "
            f"Available: {list(Architectures.keys())}"
        )
    if cfg_emulator.network.params is None:
        raise ValueError(
            "cfg.processes.iceflow.emulator.network.params must be defined."
        )

    input_names = [str(x) for x in cfg_unified.inputs]
    model = Architectures[arch_name](
        input_names=input_names,
        Nz=int(cfg_numerics.Nz),
        network_params=dict(cfg_emulator.network.params),
        dx_const=None if "dX" in input_names else 90.0,
    )

    if attach_normalizer:
        method = str(cfg_unified.normalization.method)
        if method not in NormalizationsDict:
            raise ValueError(f"Unknown normalizing method: {method}")
        normalizing_class = NormalizationsDict[method]

        if method == "adaptive":
            model.input_normalizer = normalizing_class(len(input_names))
        elif method == "fixed":
            model.input_normalizer = normalizing_class(
                cfg_unified.normalization.fixed.inputs_offsets,
                cfg_unified.normalization.fixed.inputs_variances,
            )
        else:
            model.input_normalizer = normalizing_class()
    else:
        model.input_normalizer = None

    return model


def mapping_args_for_model(
    cfg: DictConfig, state: State, model: tf.keras.Model
) -> Dict[str, Any]:
    """
    Pack the network-mapping kwargs around an already-constructed model.

    Use this when the caller (e.g. pretraining resume) has loaded the model
    itself and just needs the surrounding BCs / Nz / output_scale / precision.
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

        mapping_output_scale = float(cfg_unified.network.output_scale)

        if cfg_unified.network.pretrained:
            if cfg_unified.network.pretrained_path:
                dtype = normalize_precision(cfg_numerics.precision)
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
            warnings.warn("No pretrained emulator selected. Starting from scratch.")

            do_pretraining = "pretraining" in cfg.processes.keys()
            if cfg_unified.network.old_format and not do_pretraining:
                nb_inputs = len(cfg_unified.inputs)
                nb_outputs = 2 * int(cfg_numerics.Nz)

                arch_name = str(cfg_unified.network.architecture)
                if arch_name not in Architectures:
                    raise ValueError(
                        f"Unknown network architecture: {arch_name}. "
                        f"Available: {list(Architectures.keys())}"
                    )

                iceflow_model = Architectures[arch_name](cfg, nb_inputs, nb_outputs)
            else:
                iceflow_model = _build_cfg_model(
                    cfg, attach_normalizer=not do_pretraining
                )

        state.iceflow_model = iceflow_model

        bcs = init_bcs(cfg, state, cfg_unified.bcs)

        return {
            "bcs": bcs,
            "network": state.iceflow_model,
            "Nz": cfg_numerics.Nz,
            "output_scale": mapping_output_scale,
            "precision": cfg_numerics.precision,
        }
