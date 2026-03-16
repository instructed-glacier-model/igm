#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import yaml

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from igm.processes.iceflow.emulate.utils.architectures import Architectures
from igm.processes.iceflow.emulate.utils.normalizations import FixedChannelStandardization
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


def save_emulator_artifact(
    artifact_dir: str | Path,
    cfg,
    model: tf.keras.Model,
    inputs: List[str],
) -> Path:
    """
    Save a single supported manifest format plus weights.
    """
    artifact_dir = Path(artifact_dir)
    export_dir = artifact_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest_v3(
        cfg=cfg,
        model=model,
        inputs=inputs,
    )

    weights_path = export_dir / "weights.weights.h5"
    model.save_weights(str(weights_path))

    manifest_path = artifact_dir / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest.to_dict(), sort_keys=False)
    )

    return artifact_dir


def load_emulator_artifact(
    artifact_dir: str | Path,
    cfg,
) -> Tuple[tf.keras.Model, EmulatorManifestV3]:
    """
    Load exactly one supported manifest schema.

    This path is intentionally strict:
      - one supported schema version only
      - no cfg mutation / reconciliation
      - rebuild model only from manifest constructor params
      - attach the fixed normalizer before build / weight restore
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

    cfg_numerics = cfg.processes.iceflow.numerics
    desired_dtype = normalize_precision(cfg_numerics.precision)

    constructor_kwargs = dict(manifest.architecture.params)
    model = Architectures[arch_name](**constructor_kwargs)

    # Attach the normalizer BEFORE build so the model trackable structure
    # matches what was saved as closely as possible.
    model.input_normalizer = build_fixed_input_normalizer_from_manifest(
        manifest,
        desired_dtype,
        expected_nb_inputs=manifest.nb_inputs,
        name="input_norm",
    )

    input_shape = tf.TensorShape([None, None, None, manifest.nb_inputs])
    model.build(input_shape)

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
