#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import importlib_resources
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

import igm.processes.iceflow.emulate.emulators as emulators

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from igm.processes.iceflow.emulate.utils.architectures import Architectures


EMULATOR_FILENAME = "emulator.keras"

# Config paths relative to processes.iceflow; all values live in compatibility.
HARD_CHECKS = (
    "numerics.Nz",
    "numerics.basis_vertical",
    "numerics.basis_horizontal",
    "physics.sliding.u_ref",
    "unified.inputs",
    "physics.ice_density",
    "physics.gravity_cst",
    "physics.energy_components",
    "physics.sliding.law",
    "physics.sliding.exponent",
    "physics.viscosity.exponent",
    "unified.network.output_scale",
)
WARNING_CHECKS = (
    "physics.sliding.regularization",
    "physics.viscosity.regularization",
    "physics.thr_ice_thk",
    "physics.min_sr",
    "physics.max_sr",
)


def _config_value(cfg, path):
    value = OmegaConf.select(cfg.processes.iceflow, path, throw_on_missing=True)
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def capture_artifact_compatibility(cfg):
    """Capture all settings used to validate an artifact."""
    return {path: _config_value(cfg, path)
            for path in HARD_CHECKS + WARNING_CHECKS}


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


def _resolve_emulator_path(path: str | Path) -> Path:
    path = Path(path)
    if path.parent == Path("."):  # bare name: an emulator shipped with igm
        path = Path(importlib_resources.files(emulators)) / path
    return path if path.suffix == ".keras" else path / EMULATOR_FILENAME


def _print_loaded_banner(artifact_path: Path, model: "EmulatorArtifact") -> None:
    info = Table(show_header=False, border_style="green", expand=False)
    info.add_column("Label", style="label")
    info.add_column("Value", style="value")
    info.add_row("Architecture", model.architecture_name)
    info.add_row(
        "I/O", f"{model.nb_inputs} -> {model.nb_outputs} (Nz={model.Nz})"
    )
    info.add_row("Artifact", f"[path]{artifact_path}[/path]")

    _console.print()
    _console.print(
        Panel(
            Group(info),
            title="[ok]Emulator loaded successfully[/ok]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    _console.print()


def _architecture_name_for(core_model: tf.keras.Model) -> str:
    for name, cls in Architectures.items():
        if cls is type(core_model):
            return name
    raise ValueError(
        f"Architecture class {type(core_model).__name__} is not registered "
        f"in Architectures."
    )


@tf.keras.utils.register_keras_serializable(package="igm")
class EmulatorArtifact(tf.keras.Model):
    """
    Thin wrapper that bundles an architecture and its input normalizer
    into a single .keras-serializable model.

    The wrapper exists because:
      - The underlying architectures are not @register_keras_serializable, so
        Keras cannot rebuild them from a saved config on its own. The wrapper
        records the architecture name + constructor kwargs and re-instantiates
        the architecture at load time.
      - It owns a `tf.keras.layers.Normalization` sublayer; its mean/variance
        round-trip through model.save / load_model as ordinary layer weights.

    To compute normalization statistics before training, adapt the layer
    directly on a dataset of x batches:

        artifact.input_normalizer.adapt(dataset)
    """

    def __init__(
        self,
        *,
        architecture_name: str,
        architecture_params: dict[str, Any],
        core_model: tf.keras.Model | None = None,
        compatibility: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.architecture_name = str(architecture_name)
        self.architecture_params = dict(architecture_params)
        self.compatibility = dict(compatibility or {})

        if core_model is None:
            if self.architecture_name.lower() not in Architectures:
                raise ValueError(
                    f"Unknown architecture {self.architecture_name!r}. "
                    f"Available: {list(Architectures.keys())}"
                )
            self.core = Architectures[self.architecture_name.lower()](**self.architecture_params)
        else:
            self.core = core_model

        self.input_normalizer = tf.keras.layers.Normalization(
            axis=-1, name="input_norm"
        )
        self.core.input_normalizer = self.input_normalizer

        self.nb_inputs = int(self.core.nb_inputs)
        self.nb_outputs = int(self.core.nb_outputs)
        self.Nz = int(self.core.Nz)
        self.input_names = list(self.core.input_names)
        self._build_input_shape = [None, None, None, self.nb_inputs]

    def build(self, input_shape) -> None:
        if self.built:
            return
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 4:
            raise ValueError(
                f"EmulatorArtifact expects rank-4 inputs, got {input_shape}"
            )
        self._build_input_shape = input_shape.as_list()
        self.core.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        return self.core(inputs, training=training)

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["architecture_name"] = self.architecture_name
        config["architecture_params"] = self.architecture_params
        config["compatibility"] = dict(self.compatibility)
        return config

    def get_build_config(self) -> dict[str, Any]:
        return {"input_shape": self._build_input_shape}

    def build_from_config(self, config: dict[str, Any]) -> None:
        self.build(
            config.get("input_shape", [None, None, None, self.nb_inputs])
        )


def wrap_emulator_artifact(core_model: tf.keras.Model) -> EmulatorArtifact:
    """
    Wrap a constructed architecture instance so it can be saved as a .keras file.
    """
    if isinstance(core_model, EmulatorArtifact):
        return core_model

    if not hasattr(core_model, "resolved_params"):
        raise TypeError(
            f"{type(core_model).__name__} does not expose resolved_params(); "
            "cannot wrap it for saving."
        )

    return EmulatorArtifact(
        architecture_name=_architecture_name_for(core_model),
        architecture_params=core_model.resolved_params(),
        core_model=core_model,
    )


def save_emulator_artifact(
    artifact_dir: str | Path, model: tf.keras.Model
) -> Path:
    """Save the model as a Keras emulator artifact (architecture + weights + normalizer)."""
    artifact_path = _resolve_emulator_path(artifact_dir)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(model, EmulatorArtifact):
        model = wrap_emulator_artifact(model)

    model.save(str(artifact_path), overwrite=True)
    return artifact_path


def validate_emulator_artifact(
    model: EmulatorArtifact, cfg: DictConfig, expected_inputs,
) -> None:
    """Require complete training metadata; report hard mismatches or soft warnings."""
    checks = HARD_CHECKS + WARNING_CHECKS
    saved = {path: model.compatibility.get(path) for path in checks}
    if missing := [path for path, value in saved.items() if value is None]:
        raise ValueError("Missing emulator artifact settings: " + ", ".join(missing))
    current = {path: _config_value(cfg, path) for path in checks}
    current["unified.inputs"] = list(expected_inputs)
    # Energy order is immaterial; input-channel order is part of the model contract.
    for values in (saved, current):
        values["physics.energy_components"] = sorted(values["physics.energy_components"])
    errors, notices = [], []
    for path in checks:
        if saved[path] != current[path]:
            messages = errors if path in HARD_CHECKS else notices
            messages.append(f"{path}: artifact={saved[path]!r}, config={current[path]!r}")
    if errors:
        raise ValueError("Incompatible emulator artifact:\n" + "\n".join(errors))
    if notices:
        warnings.warn("Emulator artifact training differences:\n" + "\n".join(notices),
                      UserWarning, stacklevel=2)


def load_emulator_artifact(
    artifact_dir: str | Path,
    cfg: DictConfig,
    expected_inputs,
) -> EmulatorArtifact:
    """Load a Keras emulator artifact and validate its training settings."""
    artifact_path = _resolve_emulator_path(artifact_dir)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing Keras emulator artifact at {artifact_path}"
        )

    model = tf.keras.models.load_model(
        str(artifact_path), compile=False, safe_mode=True
    )
    if not isinstance(model, EmulatorArtifact):
        raise TypeError(
            f"Expected {artifact_path} to contain an IGM emulator, "
            f"got {type(model)}"
        )
    
    validate_emulator_artifact(model, cfg, expected_inputs)

    _print_loaded_banner(artifact_path, model)

    return model
