"""
Unit tests for validate_emulator_artifact.

All tests use a lightweight stub instead of loading a real .keras file.
"""

from __future__ import annotations

import types

import pytest
from omegaconf import OmegaConf

from igm.processes.iceflow.emulate.utils.artifacts import validate_emulator_artifact


def _artifact(nz: int, input_names: list[str]):
    return types.SimpleNamespace(Nz=nz, input_names=input_names)


def _cfg(nz: int, inputs: list[str]):
    return OmegaConf.create({"processes": {"iceflow": {
        "numerics": {"Nz": nz},
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
