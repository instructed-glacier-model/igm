"""
Unit tests for validate_emulator_artifact.

All tests use a lightweight stub instead of loading a real .keras file.
"""

from __future__ import annotations

import types

import pytest
from omegaconf import OmegaConf

from igm.processes.iceflow.emulate.utils.artifacts import validate_emulator_artifact


def _artifact(nz: int, input_names: list[str], **kwargs):
    return types.SimpleNamespace(Nz=nz, input_names=input_names, **kwargs)


def _cfg(nz: int, inputs: list[str], basis_vertical: str = "molho", basis_horizontal: str = "q1"):
    return OmegaConf.create({"processes": {"iceflow": {
        "numerics": {"Nz": nz, "basis_vertical": basis_vertical, "basis_horizontal": basis_horizontal},
        "unified": {"inputs": inputs},
    }}})


_INPUTS = ["thk", "usurf", "arrhenius", "tau_ref", "dX"]


@pytest.mark.unit
def test_valid_passes():
    validate_emulator_artifact(_artifact(2, _INPUTS), _cfg(2, _INPUTS))


@pytest.mark.unit
def test_nz_mismatch_raises():
    with pytest.raises(ValueError, match=r"Nz"):
        validate_emulator_artifact(_artifact(2, _INPUTS), _cfg(10, _INPUTS))


@pytest.mark.unit
def test_channel_order_mismatch_raises():
    wrong_order = ["thk", "usurf", "tau_ref", "arrhenius", "dX"]
    with pytest.raises(ValueError, match=r"channel"):
        validate_emulator_artifact(_artifact(2, _INPUTS), _cfg(2, wrong_order))


@pytest.mark.unit
def test_channel_set_mismatch_raises():
    wrong_set = ["thk", "usurf", "arrhenius", "slidingco", "dX"]
    with pytest.raises(ValueError, match=r"channel"):
        validate_emulator_artifact(_artifact(2, _INPUTS), _cfg(2, wrong_set))


@pytest.mark.unit
def test_basis_vertical_mismatch_raises():
    model = _artifact(2, _INPUTS, basis_vertical="molho", basis_horizontal="q1")
    with pytest.raises(ValueError, match=r"basis_vertical"):
        validate_emulator_artifact(model, _cfg(2, _INPUTS, basis_vertical="uniform"))


@pytest.mark.unit
def test_basis_horizontal_mismatch_raises():
    model = _artifact(2, _INPUTS, basis_vertical="molho", basis_horizontal="q1")
    with pytest.raises(ValueError, match=r"basis_horizontal"):
        validate_emulator_artifact(model, _cfg(2, _INPUTS, basis_horizontal="q2"))
