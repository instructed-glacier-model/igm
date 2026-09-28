#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import tensorflow as tf

import igm
from igm.processes.ocean import ocean
from igm.processes.thk.masks import WATER_LEVEL_NO_OCEAN


def _cfg(**overrides):
    defaults = Path(igm.__file__).parent / "conf" / "processes" / "ocean.yaml"
    cfg = OmegaConf.create({"processes": OmegaConf.load(defaults)})
    cfg.processes.ocean.merge_with(overrides)
    return cfg


def _state(lsurf, topg, thk, water_level=0.0, t=0.0, **fields):
    as_tensor = lambda a: tf.constant(np.asarray(a, np.float32))
    return SimpleNamespace(
        lsurf=as_tensor(lsurf),
        topg=as_tensor(topg),
        thk=as_tensor(thk),
        water_level=tf.fill(np.shape(thk), np.float32(water_level)),
        t=tf.Variable(np.float32(t)),
        **{name: as_tensor(value) for name, value in fields.items()},
    )


def _run(cfg, state):
    ocean.initialize(cfg, state)
    ocean.update(cfg, state)
    return state


@pytest.mark.fast
@pytest.mark.unit
def test_warm_profile_is_interpolated_at_the_ice_base():
    # Shelf drafts at -360 m and above/below the profile, open water at -900 m.
    state = _state(
        lsurf=[[-360.0, 10.0, -900.0, -1000.0]],
        topg=[[-800.0, -500.0, -900.0, -1000.0]],
        thk=[[400.0, 0.0, 0.0, 1100.0]],
    )
    _run(_cfg(), state)
    # Ice-free cell at topg -500 m (lsurf there is the sea surface).
    expected_z = np.array([[-360.0, -500.0, -900.0, -1000.0]])
    expected_t = np.interp(expected_z, [-720.0, 0.0], [1.0, -1.9])
    expected_s = np.interp(expected_z, [-720.0, 0.0], [34.7, 33.8])
    np.testing.assert_allclose(state.ocean_temp.numpy(), expected_t, atol=1e-6)
    np.testing.assert_allclose(state.ocean_salinity.numpy(), expected_s, atol=1e-5)
    np.testing.assert_allclose(state.ocean_temp.numpy()[0, 0], -0.45, atol=1e-6)

    tf_expected = expected_t - (-0.0573 * expected_s + 0.0832 + 7.61e-4 * expected_z)
    np.testing.assert_allclose(
        state.ocean_thermal_forcing.numpy(), tf_expected, atol=1e-5
    )


@pytest.mark.fast
@pytest.mark.unit
def test_depth_is_relative_to_the_water_level_and_finite_without_ocean():
    state = _state(lsurf=[[-350.0]], topg=[[-800.0]], thk=[[400.0]], water_level=10.0)
    np.testing.assert_allclose(
        ocean.ocean_depth(
            state.thk, state.lsurf, state.topg, state.water_level
        ).numpy(),
        -360.0,
    )

    state = _state(
        lsurf=[[-300.0]],
        topg=[[-300.0]],
        thk=[[400.0]],
        water_level=WATER_LEVEL_NO_OCEAN,
    )
    np.testing.assert_allclose(
        ocean.ocean_depth(
            state.thk, state.lsurf, state.topg, state.water_level
        ).numpy(),
        0.0,
    )
    _run(_cfg(), state)
    assert np.isfinite(state.ocean_thermal_forcing.numpy()).all()


@pytest.mark.fast
@pytest.mark.unit
def test_thermal_forcing_field_round_trips_with_the_anomaly():
    cfg = _cfg(
        method="fields",
        fields={"temp": "", "salinity": "", "thermal_forcing": "tf_input"},
        anomaly_array=[["time", "delta_temp"], [0.0, 0.0], [100.0, 2.0]],
    )
    state = _state(
        lsurf=[[-400.0, -700.0]],
        topg=[[-900.0, -900.0]],
        thk=[[450.0, 800.0]],
        t=25.0,
        tf_input=[[1.0, 3.0]],
    )
    _run(cfg, state)
    np.testing.assert_allclose(
        state.ocean_thermal_forcing.numpy(), [[1.5, 3.5]], atol=1e-5
    )
    np.testing.assert_allclose(state.ocean_salinity.numpy(), 34.5)


@pytest.mark.fast
@pytest.mark.unit
def test_temperature_and_salinity_fields_are_used_as_given():
    cfg = _cfg(method="fields")
    state = _state(
        lsurf=[[-400.0]],
        topg=[[-900.0]],
        thk=[[450.0]],
        thetao=[[0.5]],
        so=[[34.6]],
    )
    _run(cfg, state)
    np.testing.assert_allclose(state.ocean_temp.numpy(), 0.5)
    np.testing.assert_allclose(state.ocean_salinity.numpy(), 34.6, rtol=1e-6)


@pytest.mark.fast
@pytest.mark.unit
def test_configuration_errors_are_explicit():
    state = _state(lsurf=[[-400.0]], topg=[[-900.0]], thk=[[450.0]])
    with pytest.raises(ValueError, match="available methods: fields, profile"):
        ocean.initialize(_cfg(method="box"), state)
    with pytest.raises(ValueError, match="'thetao'"):
        ocean.initialize(_cfg(method="fields"), state)
    with pytest.raises(ValueError, match="exactly one"):
        ocean.initialize(
            _cfg(method="fields", fields={"thermal_forcing": "tf_input"}), state
        )
