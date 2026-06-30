"""
Unit tests for validate_emulator_artifact.

All tests use a lightweight stub instead of loading a real .keras file.
"""

from __future__ import annotations

import types

import pytest
from omegaconf import OmegaConf

from igm.processes.iceflow.emulate.utils.artifacts import validate_emulator_artifact


def _artifact(nz: int, input_names: list[str], *, u_ref: float = 1.0, **kwargs):
    return types.SimpleNamespace(Nz=nz, input_names=input_names, u_ref=u_ref, **kwargs)


def _cfg(nz: int, inputs: list[str], basis_vertical: str = "molho", basis_horizontal: str = "q1",
         u_ref: float = 1.0):
    return OmegaConf.create({"processes": {"iceflow": {
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
    with pytest.raises(ValueError, match=r"channel"):
        validate_emulator_artifact(_artifact(2, _INPUTS), _cfg(2, wrong_order), wrong_order)


@pytest.mark.unit
def test_channel_set_mismatch_raises():
    wrong_set = ["thk", "usurf", "arrhenius", "slidingco", "dX"]
    with pytest.raises(ValueError, match=r"channel"):
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
