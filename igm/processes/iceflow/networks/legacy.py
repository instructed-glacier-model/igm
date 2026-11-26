#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Loader for the previous-generation (.h5) pretrained emulators.

A legacy emulator is a folder holding ``model.h5`` and ``fieldin.dat`` (the
bundled ``pinnbp_*`` folders in ``networks/pretrained``). It is selected with
``processes.iceflow.unified.network.pretrained_path``, like a ``.keras``
artifact.
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional

import importlib_resources
import tensorflow as tf
from omegaconf import DictConfig

from . import pretrained


def resolve_legacy_emulator(
    path: str, base_dir: Optional[str] = None
) -> Optional[Path]:
    """Return the folder of a legacy emulator, or None if *path* is not one.

    A bare name first refers to an emulator shipped with igm. A relative path
    is then looked up in *base_dir* (the directory igm_run was launched from),
    then taken as given.
    """
    path = Path(path)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if path.parent == Path("."):
            candidates.append(Path(importlib_resources.files(pretrained)) / path)
        if base_dir is not None:
            candidates.append(Path(base_dir) / path)
        candidates.append(path)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "model.h5").exists():
            if not (candidate / "fieldin.dat").exists():
                raise FileNotFoundError(
                    f"❌ Legacy emulator folder {candidate} has no fieldin.dat "
                    "(the list of its input fields), so it cannot be loaded."
                )
            return candidate
    return None


def load_legacy_emulator(
    path: Path, cfg: DictConfig, cfg_inputs: Optional[List[str]]
) -> tf.keras.Model:
    """Load a legacy emulator folder, checking its inputs against *cfg_inputs*."""
    warnings.warn(
        "Loading old-format pretrained emulator. "
        "This path is kept only for legacy compatibility."
    )
    if cfg.processes.iceflow.physics.sliding.u_ref != 1.0:
        warnings.warn(
            "Legacy emulators were trained with the u_ref = 1 convention. Set "
            "processes.iceflow.physics.sliding.u_ref to 1.0 and tau_ref to match "
            "(e.g. 0.0464)."
        )
    return load_model_from_path(path, cfg_inputs)


def load_model_from_path(path: str, cfg_inputs: Optional[List[str]]) -> tf.keras.Model:

    inputs = []
    fid = open(os.path.join(path, "fieldin.dat"), "r")
    for fileline in fid:
        part = fileline.split()
        inputs.append(part[0])
    fid.close()

    # Legacy emulators call the friction input `slidingco`; the unified stack calls it `tau_ref`
    if cfg_inputs is not None and "tau_ref" in cfg_inputs:
        inputs = ["tau_ref" if name == "slidingco" else name for name in inputs]

    if cfg_inputs is not None:
        assert (
            list(cfg_inputs) == inputs
        ), f"Emulator at {path} expects inputs {inputs}, got {list(cfg_inputs)}"

    return tf.keras.models.load_model(os.path.join(path, "model.h5"), compile=False)
