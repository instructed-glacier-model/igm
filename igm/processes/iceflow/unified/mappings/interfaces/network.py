import tensorflow as tf

import warnings
from typing import Any, Dict, Optional
from omegaconf import DictConfig

from .interface import InterfaceMapping
from igm.common import State
from igm.processes.iceflow.emulate.utils.artifacts import load_emulator_artifact
from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.processes.iceflow.emulate.utils import NormalizationsDict
from igm.processes.iceflow.unified.bcs.utils import init_bcs
from igm.utils.math.precision import normalize_precision
from igm.processes.iceflow.emulate.utils.misc import (
    get_pretrained_emulator_path,
    load_model_from_path,
)


class InterfaceNetwork(InterfaceMapping):
    @staticmethod
    def _build_new_format_model(cfg: DictConfig):
        cfg_numerics = cfg.processes.iceflow.numerics
        cfg_unified = cfg.processes.iceflow.unified
        cfg_emulator = cfg.processes.iceflow.emulator

        arch_name = str(cfg_unified.network.architecture)
        if arch_name not in Architectures:
            raise ValueError(
                f"Unknown network architecture: {arch_name}. "
                f"Available: {list(Architectures.keys())}"
            )

        if not hasattr(cfg_emulator.network, "params") or cfg_emulator.network.params is None:
            raise ValueError(
                "cfg.processes.iceflow.emulator.network.params must be defined "
                "for non-old-format architectures."
            )

        input_names = [str(x) for x in cfg_unified.inputs]
        network_params = dict(cfg_emulator.network.params)
        Nz = int(cfg_numerics.Nz)

        dx_const: Optional[float]
        if "dX" in input_names:
            dx_const = None
        else:
            dx_const = 90.0

        return Architectures[arch_name](
            input_names=input_names,
            Nz=Nz,
            network_params=network_params,
            dx_const=dx_const,
        )

    @staticmethod
    def _attach_config_normalizer(cfg: DictConfig, model: tf.keras.Model) -> None:
        cfg_unified = cfg.processes.iceflow.unified
        nb_inputs = len(cfg_unified.inputs)

        method = str(cfg_unified.normalization.method)
        if method not in NormalizationsDict:
            raise ValueError(f"Unknown normalizing method: {method}")

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

        model.input_normalizer = normalizing_layer

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
            else:
                iceflow_model = InterfaceNetwork._build_new_format_model(cfg)

            if cfg.processes.iceflow.do_pretraining:
                # Pretraining computes and attaches a fixed normalizer itself.
                iceflow_model.input_normalizer = None
            else:
                InterfaceNetwork._attach_config_normalizer(cfg, iceflow_model)

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
