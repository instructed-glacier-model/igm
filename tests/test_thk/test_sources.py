#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""The thickness source term smb + bmb is shared by every transport scheme."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import tensorflow as tf

import igm
from igm.processes.thk import thk as thk_module
from igm.processes.thk.masks import WATER_LEVEL_NO_OCEAN
from igm.processes.thk.sources import mass_balance

SCHEMES = ("explicit", "implicit", "implicit_x", "adi", "ffsl")


def _cfg(scheme, bmb_process=True):
    defaults = Path(igm.__file__).parent / "conf" / "processes" / "thk.yaml"
    cfg = OmegaConf.create({"processes": OmegaConf.load(defaults)})
    cfg.processes.thk.scheme = scheme
    if bmb_process:
        cfg.processes.bmb = {}
    return cfg


def _state(thk=100.0, smb=0.5, bmb=None, shape=(6, 9)):
    thickness = tf.fill(shape, tf.constant(thk, tf.float32))
    state = SimpleNamespace(
        thk=thickness,
        topg=tf.zeros_like(thickness),
        water_level=tf.constant(WATER_LEVEL_NO_OCEAN),
        ubar=tf.zeros_like(thickness),
        vbar=tf.zeros_like(thickness),
        smb=tf.fill(shape, tf.constant(smb, tf.float32)),
        dx=tf.constant(100.0),
        dt=tf.constant(1.0),
        it=0,
    )
    if bmb is not None:
        state.bmb = tf.fill(shape, tf.constant(bmb, tf.float32))
    return state


def _step(scheme, state, bmb_process=True):
    cfg = _cfg(scheme, bmb_process)
    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)
    return state


@pytest.mark.fast
@pytest.mark.unit
def test_mass_balance_is_smb_itself_without_bmb():
    state = _state()
    assert mass_balance(state) is state.smb

    del state.smb
    source = mass_balance(state)
    assert source is state.smb
    np.testing.assert_array_equal(source.numpy(), 0.0)


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("scheme", SCHEMES)
def test_bmb_counts_only_with_the_bmb_process(scheme):
    """A bmb field read from an input file (e.g. an IGM output) is ignored."""
    with_process = _step(scheme, _state(thk=100.0, smb=0.5, bmb=-2.0))
    np.testing.assert_allclose(mass_balance(with_process).numpy(), -1.5)
    without = _step(scheme, _state(thk=100.0, smb=0.5, bmb=-2.0), bmb_process=False)
    np.testing.assert_allclose(without.thk.numpy(), 100.5, rtol=1e-6)


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("scheme", SCHEMES)
def test_zero_bmb_is_bit_identical(scheme):
    without = _step(scheme, _state())
    with_zero = _step(scheme, _state(bmb=0.0))
    np.testing.assert_array_equal(with_zero.thk.numpy(), without.thk.numpy())


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("scheme", SCHEMES)
def test_bmb_changes_thickness_like_smb(scheme):
    state = _step(scheme, _state(thk=100.0, smb=0.5, bmb=-2.0))
    np.testing.assert_allclose(state.thk.numpy(), 98.5, rtol=1e-6)
    np.testing.assert_allclose(state.divflux.numpy(), 0.0, atol=1e-4)


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("scheme", SCHEMES)
def test_melt_beyond_the_available_ice_is_clipped(scheme):
    state = _step(scheme, _state(thk=1.0, smb=0.0, bmb=-5.0))
    np.testing.assert_array_equal(state.thk.numpy(), 0.0)
