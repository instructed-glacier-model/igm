"""
Unit tests for validate_emulator_artifact.

Includes lightweight validation tests and bundled artifact round trips.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np

import pytest
from omegaconf import OmegaConf

from igm.processes.iceflow.emulate.utils.artifacts import (
    validate_emulator_artifact, capture_artifact_compatibility, HARD_CHECKS,
    WARNING_CHECKS, load_emulator_artifact, save_emulator_artifact,
)


def _artifact(nz: int, input_names: list[str], *, u_ref: float = 1.0, **kwargs):
    cfg = _cfg(nz, input_names, u_ref=u_ref, **kwargs)
    return types.SimpleNamespace(compatibility=capture_artifact_compatibility(cfg))


def _cfg(nz: int, inputs: list[str], basis_vertical: str = "molho", basis_horizontal: str = "q1",
         u_ref: float = 1.0):
    defaults = OmegaConf.load(Path(__file__).parents[2] / "igm/conf/processes/iceflow.yaml")
    return OmegaConf.merge({"processes": defaults}, {"processes": {"iceflow": {
        "numerics": {"Nz": nz, "basis_vertical": basis_vertical, "basis_horizontal": basis_horizontal},
        "physics": {"sliding": {"u_ref": u_ref}},
        "unified": {"inputs": inputs},
    }}})


_INPUTS = ["thk", "usurf", "arrhenius", "tau_ref", "dX"]


@pytest.mark.unit
def test_valid_passes():
    validate_emulator_artifact(_artifact(2, _INPUTS), _cfg(2, _INPUTS), _INPUTS)


@pytest.mark.unit
def test_nz_mismatch_raises():
    with pytest.raises(ValueError, match=r"Nz"):
        validate_emulator_artifact(_artifact(2, _INPUTS), _cfg(10, _INPUTS), _INPUTS)


@pytest.mark.unit
def test_channel_order_mismatch_raises():
    wrong_order = ["thk", "usurf", "tau_ref", "arrhenius", "dX"]
    with pytest.raises(ValueError, match=r"unified.inputs"):
        validate_emulator_artifact(_artifact(2, _INPUTS), _cfg(2, wrong_order), wrong_order)


@pytest.mark.unit
def test_channel_set_mismatch_raises():
    wrong_set = ["thk", "usurf", "arrhenius", "slidingco", "dX"]
    with pytest.raises(ValueError, match=r"unified.inputs"):
        validate_emulator_artifact(_artifact(2, _INPUTS), _cfg(2, wrong_set), wrong_set)


@pytest.mark.unit
def test_basis_vertical_mismatch_raises():
    model = _artifact(2, _INPUTS, basis_vertical="molho", basis_horizontal="q1")
    with pytest.raises(ValueError, match=r"basis_vertical"):
        validate_emulator_artifact(model, _cfg(2, _INPUTS, basis_vertical="uniform"), _INPUTS)


@pytest.mark.unit
def test_basis_horizontal_mismatch_raises():
    model = _artifact(2, _INPUTS, basis_vertical="molho", basis_horizontal="q1")
    with pytest.raises(ValueError, match=r"basis_horizontal"):
        validate_emulator_artifact(model, _cfg(2, _INPUTS, basis_horizontal="q2"), _INPUTS)


@pytest.mark.unit
def test_u_ref_match_passes():
    model = _artifact(2, _INPUTS, u_ref=100.0)
    validate_emulator_artifact(model, _cfg(2, _INPUTS, u_ref=100.0), _INPUTS)


@pytest.mark.unit
def test_u_ref_mismatch_raises():
    model = _artifact(2, _INPUTS, u_ref=100.0)
    with pytest.raises(ValueError, match=r"u_ref"):
        validate_emulator_artifact(model, _cfg(2, _INPUTS, u_ref=1.0), _INPUTS)


@pytest.mark.parametrize("path", HARD_CHECKS)
def test_hard_mismatch_raises(path):
    cfg = _cfg(2, _INPUTS)
    value = OmegaConf.select(cfg.processes.iceflow, path)
    changed = (["viscosity"] if path.endswith("energy_components") else
               "budd" if isinstance(value, str) else
               ["thk"] if OmegaConf.is_list(value) else value * 2)
    OmegaConf.update(cfg.processes.iceflow, path, changed)
    with pytest.raises(ValueError, match=path):
        validate_emulator_artifact(_artifact(2, _INPUTS), cfg, cfg.processes.iceflow.unified.inputs)


@pytest.mark.parametrize("path", WARNING_CHECKS)
def test_numerical_mismatch_warns(path):
    cfg = _cfg(2, _INPUTS)
    OmegaConf.update(cfg.processes.iceflow, path,
                     OmegaConf.select(cfg.processes.iceflow, path) * 2)
    with pytest.warns(UserWarning, match=path):
        validate_emulator_artifact(_artifact(2, _INPUTS), cfg, _INPUTS)


def test_energy_order_is_irrelevant_but_duplicates_are_not():
    cfg = _cfg(2, _INPUTS)
    cfg.processes.iceflow.physics.energy_components = ["sliding", "gravity", "viscosity"]
    validate_emulator_artifact(_artifact(2, _INPUTS), cfg, _INPUTS)
    cfg.processes.iceflow.physics.energy_components.append("sliding")
    with pytest.raises(ValueError, match="energy_components"):
        validate_emulator_artifact(_artifact(2, _INPUTS), cfg, _INPUTS)


@pytest.mark.parametrize("path", HARD_CHECKS + WARNING_CHECKS)
def test_every_setting_is_required(path):
    model = _artifact(2, _INPUTS)
    del model.compatibility[path]
    with pytest.raises(ValueError, match=path):
        validate_emulator_artifact(model, _cfg(2, _INPUTS), _INPUTS)


def test_missing_metadata_lists_all_missing_keys():
    with pytest.raises(ValueError) as exc:
        validate_emulator_artifact(types.SimpleNamespace(compatibility={}),
                                   _cfg(2, _INPUTS), _INPUTS)
    assert all(path in str(exc.value) for path in HARD_CHECKS + WARNING_CHECKS)


def test_mismatches_are_reported_together():
    cfg = _cfg(10, _INPUTS, u_ref=100)
    with pytest.raises(ValueError) as exc:
        validate_emulator_artifact(_artifact(2, _INPUTS), cfg, _INPUTS)
    assert "numerics.Nz" in str(exc.value)
    assert "physics.sliding.u_ref" in str(exc.value)


def test_excluded_settings_do_not_change_compatibility():
    cfg = _cfg(2, _INPUTS)
    cfg.processes.iceflow.numerics.vert_spacing = 99
    cfg.processes.iceflow.physics.sliding.use_mask_gr = True
    cfg.processes.iceflow.unified.bcs = ["frozen_bed"]
    validate_emulator_artifact(_artifact(2, _INPUTS), cfg, _INPUTS)


@pytest.mark.parametrize("name", ["dahunet", "dahunet_mini", "dahunet_micro"])
def test_public_artifact_loads_and_roundtrips(name, tmp_path):
    import tensorflow as tf
    cfg = _cfg(2, _INPUTS, u_ref=100)
    model = load_emulator_artifact(name + ".keras", cfg, _INPUTS)
    assert dict(model.compatibility) == capture_artifact_compatibility(cfg)
    assert not {"u_ref", "basis_vertical", "basis_horizontal"} & model.get_config().keys()
    x = tf.ones((1, 8, 8, 5))
    before = model(x).numpy()
    saved = save_emulator_artifact(tmp_path / name, model)
    restored = load_emulator_artifact(saved, cfg, _INPUTS)
    assert dict(restored.compatibility) == dict(model.compatibility)
    np.testing.assert_array_equal(restored(x).numpy(), before)
