#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
import yaml

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from igm.processes.iceflow.emulate.utils import NormalizationsDict
from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.utils.math.precision import normalize_precision

from .artifacts_schema_v3 import (
    SUPPORTED_SCHEMA_VERSION,
    EmulatorManifestV3,
    build_manifest_v3,
    build_fixed_input_normalizer_from_manifest,
    load_supported_manifest,
    validate_manifest_against_cfg_v3,
)


_emulator_theme = Theme(
    {
        "label": "bold #e5e7eb",
        "value": "#06b6d4",
        "path": "#a78bfa",
        "ok": "bold #22c55e",
        "muted": "italic #64748b",
    }
)
_console = Console(theme=_emulator_theme)


def _print_loaded_banner(
    artifact_dir: Path,
    manifest: EmulatorManifestV3,
    dtype: tf.DType,
) -> None:
    info = Table(show_header=False, border_style="green", expand=False)
    info.add_column("Label", style="label")
    info.add_column("Value", style="value")

    info.add_row("Architecture", str(manifest.architecture.name))
    info.add_row(
        "I/O",
        f"{manifest.nb_inputs} → {manifest.nb_outputs}   "
        f"(Nz={manifest.Nz}, dtype={dtype.name})",
    )
    info.add_row("Artifact", f"[path]{artifact_dir}[/path]")

    _console.print()
    _console.print(
        Panel(
            Group(info),
            title="[ok]✅ Emulator loaded successfully[/ok]",
            subtitle=f"[muted]Schema v{SUPPORTED_SCHEMA_VERSION}[/muted]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    _console.print()


def _build_model_from_cfg_constructor(cfg) -> tf.keras.Model:
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

    dx_const = None if "dX" in input_names else 90.0

    return Architectures[arch_name](
        input_names=input_names,
        Nz=Nz,
        network_params=network_params,
        dx_const=dx_const,
    )


def _attach_config_normalizer(cfg, model: tf.keras.Model) -> None:
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


def build_emulator_from_cfg(
    cfg,
    *,
    attach_config_normalizer: bool,
) -> tf.keras.Model:
    """
    Construct a new-format emulator purely from cfg.

    Use this when no manifest exists yet, i.e. fresh training from scratch.
    """
    model = _build_model_from_cfg_constructor(cfg)
    if attach_config_normalizer:
        _attach_config_normalizer(cfg, model)
    else:
        model.input_normalizer = None
    return model


def rebuild_emulator_from_manifest(
    artifact_dir: str | Path,
    cfg,
    *,
    load_weights: bool = False,
) -> Tuple[tf.keras.Model, EmulatorManifestV3]:
    """
    Rebuild a new-format emulator purely from manifest.

    This always restores the exact fixed input normalizer from the manifest.
    If load_weights=True, exported weights are loaded from export/weights.weights.h5.
    If load_weights=False, the returned model is only a structural skeleton suitable
    for subsequent checkpoint.restore(...).
    """
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.yaml at {manifest_path}")

    manifest = load_supported_manifest(manifest_path)
    validate_manifest_against_cfg_v3(cfg, manifest, artifact_dir)

    arch_name = str(manifest.architecture.name)
    if arch_name not in Architectures:
        raise ValueError(
            f"Unknown architecture {arch_name!r}. "
            f"Available: {list(Architectures.keys())}"
        )

    desired_dtype = normalize_precision(cfg.processes.iceflow.numerics.precision)
    constructor_kwargs = dict(manifest.architecture.params)
    model = Architectures[arch_name](**constructor_kwargs)

    model.input_normalizer = build_fixed_input_normalizer_from_manifest(
        manifest,
        desired_dtype,
        expected_nb_inputs=manifest.nb_inputs,
        name="input_norm",
    )

    input_shape = tf.TensorShape([None, None, None, manifest.nb_inputs])
    model.build(input_shape)

    if load_weights:
        weights_path = artifact_dir / "export" / "weights.weights.h5"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights file at {weights_path}")
        model.load_weights(str(weights_path))

        dummy = tf.zeros((1, 4, 4, manifest.nb_inputs), dtype=desired_dtype)
        y = model(dummy, training=False)

        if int(y.shape[-1]) != int(manifest.nb_outputs):
            raise RuntimeError(
                f"Loaded model output mismatch: got {int(y.shape[-1])}, "
                f"expected {int(manifest.nb_outputs)}."
            )

        if tf.as_dtype(y.dtype) != desired_dtype:
            raise RuntimeError(
                f"Model forward dtype is {tf.as_dtype(y.dtype).name}, "
                f"expected {desired_dtype.name}."
            )

        _print_loaded_banner(artifact_dir, manifest, desired_dtype)

    return model, manifest


def write_emulator_manifest(
    artifact_dir: str | Path,
    cfg,
    model: tf.keras.Model,
    inputs: List[str],
) -> Path:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest_v3(
        cfg=cfg,
        model=model,
        inputs=inputs,
    )

    manifest_path = artifact_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False))
    return manifest_path


def save_emulator_artifact(
    artifact_dir: str | Path,
    cfg,
    model: tf.keras.Model,
    inputs: List[str],
) -> Path:
    """
    Save a single supported manifest format plus exported weights.
    """
    artifact_dir = Path(artifact_dir)
    export_dir = artifact_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    write_emulator_manifest(
        artifact_dir=artifact_dir,
        cfg=cfg,
        model=model,
        inputs=inputs,
    )

    weights_path = export_dir / "weights.weights.h5"
    model.save_weights(str(weights_path))

    return artifact_dir


def load_emulator_artifact(
    artifact_dir: str | Path,
    cfg,
) -> Tuple[tf.keras.Model, EmulatorManifestV3]:
    """
    Load a finished exported artifact: manifest + exported weights.
    """
    return rebuild_emulator_from_manifest(
        artifact_dir=artifact_dir,
        cfg=cfg,
        load_weights=True,
    )
