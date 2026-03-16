#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
import yaml

from igm.processes.iceflow.emulate.utils.normalizations import FixedChannelStandardization

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme


SUPPORTED_SCHEMA_VERSION = 3


_theme = Theme(
    {
        "label": "bold #e5e7eb",
        "value": "#06b6d4",
        "path": "#a78bfa",
        "err": "bold #ef4444",
        "muted": "italic #64748b",
    }
)
_console = Console(theme=_theme)


@dataclass
class ArchitectureSpec:
    name: str
    params: Dict[str, Any]


@dataclass
class NormalizationSpec:
    method: str
    params: Dict[str, Any]


@dataclass
class EmulatorManifestV3:
    schema_version: int
    Nz: int
    basis_vertical: str
    basis_horizontal: str
    inputs: List[str]
    nb_inputs: int
    nb_outputs: int
    output_scale: float
    architecture: ArchitectureSpec
    normalization: NormalizationSpec

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_manifest_v3(raw: Dict[str, Any]) -> EmulatorManifestV3:
    schema = int(raw.get("schema_version", -1))
    if schema != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version={schema!r}. "
            f"Expected {SUPPORTED_SCHEMA_VERSION}."
        )

    required = {
        "Nz",
        "basis_vertical",
        "basis_horizontal",
        "inputs",
        "nb_inputs",
        "nb_outputs",
        "output_scale",
        "architecture",
        "normalization",
    }
    missing = sorted(required - set(raw.keys()))
    if missing:
        raise ValueError(
            f"Manifest schema v{SUPPORTED_SCHEMA_VERSION} is missing required fields: {missing}"
        )

    arch = dict(raw["architecture"])
    norm = dict(raw["normalization"])

    if str(norm.get("method", "")) != "fixed_channel_standardization":
        raise ValueError(
            "Only normalization.method='fixed_channel_standardization' is supported."
        )

    manifest = EmulatorManifestV3(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        Nz=int(raw["Nz"]),
        basis_vertical=str(raw["basis_vertical"]),
        basis_horizontal=str(raw["basis_horizontal"]),
        inputs=[str(x) for x in raw["inputs"]],
        nb_inputs=int(raw["nb_inputs"]),
        nb_outputs=int(raw["nb_outputs"]),
        output_scale=float(raw["output_scale"]),
        architecture=ArchitectureSpec(
            name=str(arch["name"]),
            params=dict(arch.get("params", {})),
        ),
        normalization=NormalizationSpec(
            method=str(norm["method"]),
            params=dict(norm.get("params", {})),
        ),
    )

    _validate_internal_manifest_consistency(manifest)
    return manifest


def _validate_internal_manifest_consistency(manifest: EmulatorManifestV3) -> None:
    if manifest.nb_inputs != len(manifest.inputs):
        raise ValueError(
            f"Manifest is inconsistent: nb_inputs={manifest.nb_inputs} but "
            f"len(inputs)={len(manifest.inputs)}."
        )

    params = dict(manifest.architecture.params)
    required = {"input_names", "Nz", "network_params", "dx_const"}
    missing = sorted(required - set(params.keys()))
    if missing:
        raise ValueError(
            f"Manifest architecture.params is missing required constructor keys: {missing}"
        )

    input_names = [str(x) for x in params["input_names"]]
    if input_names != manifest.inputs:
        raise ValueError(
            "Manifest is inconsistent: architecture.params['input_names'] does not match top-level inputs."
        )

    if int(params["Nz"]) != int(manifest.Nz):
        raise ValueError(
            "Manifest is inconsistent: architecture.params['Nz'] does not match top-level Nz."
        )

    net_params = params["network_params"]
    if not isinstance(net_params, dict):
        raise ValueError("architecture.params['network_params'] must be a dict.")

    norm_params = dict(manifest.normalization.params)
    for key in ("mean_1d", "var_1d", "epsilon"):
        if key not in norm_params:
            raise ValueError(
                f"Manifest normalization.params is missing required key {key!r}."
            )

    mean = np.asarray(norm_params["mean_1d"], dtype=np.float64).reshape(-1)
    var = np.asarray(norm_params["var_1d"], dtype=np.float64).reshape(-1)
    eps = float(norm_params["epsilon"])

    if mean.shape[0] != manifest.nb_inputs or var.shape[0] != manifest.nb_inputs:
        raise ValueError(
            "Normalization statistics length does not match manifest.nb_inputs."
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
        raise ValueError("Normalization statistics contain NaN/Inf.")
    if np.any(var < 0.0):
        raise ValueError("Normalization variance contains negative values.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("Normalization epsilon must be finite and > 0.")


def load_supported_manifest(manifest_path: str | Path) -> EmulatorManifestV3:
    """
    Read, parse, and validate the single supported manifest schema.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file at {manifest_path}")

    raw = yaml.safe_load(manifest_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"Manifest at {manifest_path} did not parse to a dict; got {type(raw)}"
        )

    return parse_manifest_v3(raw)


def build_fixed_input_normalizer_from_manifest(
    manifest: EmulatorManifestV3,
    dtype: tf.DType,
    *,
    expected_nb_inputs: int | None = None,
    name: str = "input_norm",
) -> FixedChannelStandardization:
    """
    Rebuild the exact FixedChannelStandardization layer stored in the manifest.
    """
    p = dict(manifest.normalization.params)

    mean_1d = np.asarray(p["mean_1d"], dtype=np.float64).reshape(-1)
    var_1d = np.asarray(p["var_1d"], dtype=np.float64).reshape(-1)
    eps = float(p["epsilon"])

    if expected_nb_inputs is None:
        expected_nb_inputs = int(manifest.nb_inputs)

    if mean_1d.shape[0] != expected_nb_inputs or var_1d.shape[0] != expected_nb_inputs:
        raise ValueError(
            "Normalization statistics length mismatch: "
            f"mean={mean_1d.shape[0]}, var={var_1d.shape[0]}, "
            f"expected_nb_inputs={expected_nb_inputs}"
        )

    return FixedChannelStandardization(
        mean_1d=mean_1d,
        var_1d=var_1d,
        epsilon=eps,
        dtype=dtype,
        name=name,
    )


def validate_model_matches_manifest_v3(
    model: tf.keras.Model,
    manifest: EmulatorManifestV3,
) -> None:
    """
    Ensure an already-constructed model matches the manifest reconstruction
    contract exactly.
    """
    if not hasattr(model, "resolved_params"):
        raise TypeError(
            f"Model of type {type(model)} does not expose resolved_params(); "
            "cannot verify compatibility against manifest."
        )

    actual = dict(model.resolved_params())
    expected = dict(manifest.architecture.params)

    if actual != expected:
        raise ValueError(
            "Current model structure does not match manifest reconstruction params.\n"
            f"manifest.architecture.params = {expected!r}\n"
            f"model.resolved_params()      = {actual!r}"
        )

def _extract_normalization_spec(model: tf.keras.Model) -> NormalizationSpec:
    norm = getattr(model, "input_normalizer", None)
    if norm is None:
        raise ValueError(
            "Cannot write manifest: model.input_normalizer is None. "
            "Attach FixedChannelStandardization before saving."
        )

    missing = [a for a in ("_mean_1d", "_var_1d", "epsilon") if not hasattr(norm, a)]
    if missing:
        raise TypeError(
            "Manifest writing requires model.input_normalizer to be FixedChannelStandardization "
            f"(missing attributes: {missing}). Got: {type(norm)}"
        )

    mean = np.asarray(getattr(norm, "_mean_1d").numpy(), dtype=np.float64).reshape(-1)
    var = np.asarray(getattr(norm, "_var_1d").numpy(), dtype=np.float64).reshape(-1)
    eps = float(getattr(norm, "epsilon"))

    if mean.size == 0 or var.size == 0:
        raise RuntimeError("FixedChannelStandardization statistics are empty.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
        raise RuntimeError("FixedChannelStandardization statistics contain NaN/Inf.")
    if np.any(var < 0.0):
        raise RuntimeError("FixedChannelStandardization variance contains negative values.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise RuntimeError(f"FixedChannelStandardization epsilon must be finite and > 0, got {eps!r}")

    return NormalizationSpec(
        method="fixed_channel_standardization",
        params={
            "axis": -1,
            "epsilon": eps,
            "mean_1d": mean.tolist(),
            "var_1d": var.tolist(),
        },
    )


def _extract_architecture_spec(cfg, model: tf.keras.Model, inputs: List[str]) -> ArchitectureSpec:
    arch_name = str(cfg.processes.iceflow.unified.network.architecture)

    if not hasattr(model, "resolved_params"):
        raise TypeError(
            f"Architecture {arch_name!r} does not expose resolved_params(); "
            "cannot write a reconstruction-safe manifest."
        )

    params = dict(model.resolved_params())
    required = {"input_names", "Nz", "network_params", "dx_const"}
    missing = sorted(required - set(params.keys()))
    if missing:
        raise ValueError(
            f"model.resolved_params() is missing required keys: {missing}"
        )

    params["input_names"] = [str(x) for x in params["input_names"]]
    params["Nz"] = int(params["Nz"])
    params["network_params"] = dict(params["network_params"])
    params["dx_const"] = None if params["dx_const"] is None else float(params["dx_const"])

    if params["input_names"] != list(inputs):
        raise ValueError(
            "Refusing to write manifest: model.resolved_params()['input_names'] "
            "does not match the provided inputs list."
        )

    return ArchitectureSpec(name=arch_name, params=params)


def build_manifest_v3(cfg, model: tf.keras.Model, inputs: List[str]) -> EmulatorManifestV3:
    cfg_unified = cfg.processes.iceflow.unified
    cfg_numerics = cfg.processes.iceflow.numerics
    nb_outputs = int(model.nb_outputs)

    inputs = [str(x) for x in inputs]
    arch = _extract_architecture_spec(cfg, model, inputs)
    norm = _extract_normalization_spec(model)

    if list(cfg_unified.inputs) != inputs:
        raise ValueError(
            f"Refusing to write manifest: inputs arg {inputs!r} != cfg.unified.inputs {list(cfg_unified.inputs)!r}"
        )

    if int(cfg_numerics.Nz) != int(arch.params["Nz"]):
        raise ValueError(
            f"Refusing to write manifest: cfg Nz={int(cfg_numerics.Nz)} != model Nz={int(arch.params['Nz'])}"
        )

    manifest = EmulatorManifestV3(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        Nz=int(cfg_numerics.Nz),
        basis_vertical=str(cfg_numerics.basis_vertical),
        basis_horizontal=str(cfg_numerics.basis_horizontal),
        inputs=inputs,
        nb_inputs=len(inputs),
        nb_outputs=int(nb_outputs),
        output_scale=float(cfg_unified.network.output_scale),
        architecture=arch,
        normalization=norm,
    )

    _validate_internal_manifest_consistency(manifest)
    return manifest


def _raise_cfg_incompatibility(artifact_dir: Path, manifest: EmulatorManifestV3, errors: List[str]) -> None:
    info = Table(show_header=False, border_style="red", expand=False)
    info.add_column("Label", style="label")
    info.add_column("Value", style="err")
    info.add_row("Artifact", f"[path]{artifact_dir}[/path]")
    info.add_row("Architecture", manifest.architecture.name)
    info.add_row("Inputs", str(manifest.inputs))
    info.add_row("Nz", str(manifest.Nz))

    err = Table(show_header=False, border_style="red", expand=False)
    err.add_column("", width=2)
    err.add_column("Incompatibilities", style="err")
    for m in errors:
        err.add_row("✖", m)

    _console.print()
    _console.print(
        Panel(
            Group(info, err),
            title="[err]✖ Emulator artifact incompatible with current config[/err]",
            subtitle="[muted]Refusing to mutate cfg or guess a reconstruction path[/muted]",
            border_style="red",
            padding=(1, 2),
        )
    )
    _console.print()
    raise ValueError("Emulator artifact incompatible with current cfg (see panel above).")


def validate_manifest_against_cfg_v3(cfg, manifest: EmulatorManifestV3, artifact_dir: Path) -> None:
    cfg_unified = cfg.processes.iceflow.unified
    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_emulator = cfg.processes.iceflow.emulator

    errors: List[str] = []

    if int(cfg_numerics.Nz) != int(manifest.Nz):
        errors.append(
            f"Nz mismatch: cfg={int(cfg_numerics.Nz)} vs artifact={int(manifest.Nz)}"
        )
    if str(cfg_numerics.basis_vertical) != str(manifest.basis_vertical):
        errors.append(
            "basis_vertical mismatch: "
            f"cfg={str(cfg_numerics.basis_vertical)!r} vs artifact={str(manifest.basis_vertical)!r}"
        )
    if str(cfg_numerics.basis_horizontal) != str(manifest.basis_horizontal):
        errors.append(
            "basis_horizontal mismatch: "
            f"cfg={str(cfg_numerics.basis_horizontal)!r} vs artifact={str(manifest.basis_horizontal)!r}"
        )

    cfg_inputs = [str(x) for x in cfg_unified.inputs]
    if cfg_inputs != list(manifest.inputs):
        errors.append(
            f"inputs mismatch: cfg={cfg_inputs!r} vs artifact={list(manifest.inputs)!r}"
        )

    cfg_output_scale = float(cfg_unified.network.output_scale)
    if cfg_output_scale != float(manifest.output_scale):
        errors.append(
            f"output_scale mismatch: cfg={cfg_output_scale} vs artifact={float(manifest.output_scale)}"
        )

    if errors:
        _raise_cfg_incompatibility(artifact_dir, manifest, errors)
