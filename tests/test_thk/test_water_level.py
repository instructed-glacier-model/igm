#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""The always-present water level and its "no ocean" default."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
import xarray as xr
from omegaconf import OmegaConf

from igm.inputs.complete_data import complete_data
from igm.inputs.local import complete_data as complete_local_data
from igm.processes.thk.fronts.sub_grid import _ocean
from igm.processes.thk.rigid_body import remove_rigid_body_modes
from igm.processes.thk.surfaces import update_surfaces
from igm.processes.thk.masks import (
    WATER_LEVEL_NO_OCEAN,
    compute_grounded_mask,
    no_ocean_like,
)

RHO_RATIO = 1028.0 / 910.0
CFG = OmegaConf.create({"processes": {"thk": {"ratio_density": 910.0 / 1028.0}}})


def ensure_water_level(state, value=None):
    """Give a hand-built state a water level (no ocean when ``value`` is None)."""
    state.water_level = (
        no_ocean_like(state.topg)
        if value is None
        else tf.fill(tf.shape(state.topg), value)
    )


def _ismip_like():
    """A bed entirely below 0 m under 1 km of ice, with an ice-free margin."""
    topg = tf.constant(np.linspace(-1500.0, -500.0, 12).reshape(3, 4), tf.float32)
    thk = tf.constant([[1000.0] * 3 + [0.0]] * 3, tf.float32)
    return SimpleNamespace(topg=topg, thk=thk)


@pytest.mark.parametrize(
    "include, loaded, expected",
    [(False, None, WATER_LEVEL_NO_OCEAN), (True, None, -2.0), (True, 7.0, 7.0)],
)
def test_inputs_always_create_a_water_level(include, loaded, expected):
    state = SimpleNamespace(
        X=tf.zeros((2, 3)),
        dx=1.0,
        dX=tf.ones((2, 3)),
        thk=tf.zeros((2, 3)),
        topg=tf.ones((2, 3)),
    )
    if loaded is not None:
        state.water_level = tf.fill((2, 3), loaded)
    complete_data(
        state, water_level=OmegaConf.create({"include": include, "value": -2.0})
    )
    np.testing.assert_array_equal(state.water_level, np.full((2, 3), expected))


@pytest.mark.parametrize(
    "cfg, loaded, expected",
    [
        (
            OmegaConf.create({"include": False, "value": 0.0}),
            None,
            WATER_LEVEL_NO_OCEAN,
        ),
        (OmegaConf.create({"include": True, "value": -2.0}), None, -2.0),
        (OmegaConf.create({"include": True, "value": -2.0}), 7.0, 7.0),
    ],
)
def test_local_inputs_always_create_a_water_level(cfg, loaded, expected):
    ds = xr.Dataset(
        {"topg": (("y", "x"), np.ones((2, 3)))},
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0]},
    )
    if loaded is not None:
        ds["water_level"] = (("y", "x"), np.full((2, 3), loaded))
    ds = complete_local_data(ds, water_level=cfg)
    np.testing.assert_array_equal(ds["water_level"].values, np.full((2, 3), expected))


def test_no_ocean_keeps_a_sub_zero_bed_grounded_everywhere():
    state = _ismip_like()
    ensure_water_level(state)

    update_surfaces(CFG, state)
    np.testing.assert_array_equal(state.lsurf, state.topg)  # bit for bit

    grounded = compute_grounded_mask(
        state.thk, state.usurf - state.thk, state.water_level, RHO_RATIO
    )
    assert bool(tf.reduce_all(grounded))

    thk_before = state.thk
    remove_rigid_body_modes(state, RHO_RATIO)
    np.testing.assert_array_equal(state.thk, thk_before)  # nothing calved


def test_sea_level_floats_the_same_sub_zero_bed():
    state = _ismip_like()
    ensure_water_level(state, value=0.0)
    update_surfaces(CFG, state)
    grounded = compute_grounded_mask(
        state.thk, state.usurf - state.thk, state.water_level, RHO_RATIO
    )
    ice = state.thk.numpy() > 0.0
    expected = state.thk.numpy() + RHO_RATIO * state.topg.numpy() > 0.0
    np.testing.assert_array_equal(grounded.numpy()[ice], expected[ice])
    assert expected[ice].any() and not expected[ice].all()  # both regimes


def test_mountain_geometry_is_unchanged_by_the_water_level():
    topg = tf.constant([[1200.0, 1500.0, 2100.0], [900.0, 3000.0, 0.0]], tf.float32)
    thk = tf.constant([[0.0, 300.0, 10.0], [50.0, 0.0, 0.0]], tf.float32)
    for value in (None, 0.0):
        state = SimpleNamespace(topg=topg, thk=thk)
        ensure_water_level(state, value=value)
        update_surfaces(CFG, state)
        np.testing.assert_array_equal(state.lsurf, topg)
        np.testing.assert_array_equal(state.usurf, topg + thk)


def test_sub_grid_front_needs_an_ocean():
    topg = tf.constant([[-50.0, 100.0, -50.0]])
    thk = tf.constant([[0.0, 100.0, 0.0]])

    state = SimpleNamespace(topg=topg, thk=thk)
    ensure_water_level(state)
    np.testing.assert_array_equal(_ocean(state), [[False, False, False]])

    state = SimpleNamespace(topg=tf.constant([[-50.0, 100.0, 50.0]]), thk=thk)
    ensure_water_level(state, value=0.0)
    np.testing.assert_array_equal(_ocean(state), [[True, False, False]])


def test_iceflow_warns_about_an_ocean_missing_from_its_inputs():
    from igm.processes.iceflow.utils.fields import initialize_iceflow_fields

    def cfg(inputs):
        return OmegaConf.create(
            {
                "processes": {
                    "iceflow": {
                        "method": "unified",
                        "numerics": {"Nz": 2},
                        "unified": {"inputs": inputs},
                        "physics": {
                            "viscosity": {"arrhenius": 78.0, "enhancement_factor": 1.0},
                            "sliding": {"tau_ref": 0.1, "use_mask_gr": False},
                        },
                    }
                }
            }
        )

    def state(value):
        s = SimpleNamespace(thk=tf.zeros((2, 2)), topg=tf.zeros((2, 2)))
        ensure_water_level(s, value=value)
        return s

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        initialize_iceflow_fields(cfg(["thk", "usurf"]), state(None))  # no ocean
        initialize_iceflow_fields(cfg(["thk", "usurf", "water_level"]), state(0.0))
    with pytest.warns(UserWarning, match="processes.iceflow.unified.inputs"):
        initialize_iceflow_fields(cfg(["thk", "usurf"]), state(0.0))


def test_thk_requires_a_water_level():
    from igm.processes.thk import thk as thk_module

    state = SimpleNamespace(thk=tf.zeros((2, 2)), topg=tf.zeros((2, 2)))
    with pytest.raises(ValueError, match="state.water_level"):
        thk_module.initialize(CFG, state)
