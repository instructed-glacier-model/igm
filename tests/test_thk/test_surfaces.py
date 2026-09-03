#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Tests for common thickness geometry and flotation configuration."""

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import tensorflow as tf

from igm.processes.thk import thk as thk_module


def _cfg(ratio_density, ice_density=None, water_density=None):
    processes = {
        "thk": {
            "scheme": "explicit",
            "slope_type": "superbee",
            "divflux_smooth_sigma": 0.0,
            "ratio_density": ratio_density,
            "calving_front": False,
        }
    }
    if ice_density is not None or water_density is not None:
        processes["iceflow"] = {
            "physics": {
                "ice_density": ice_density,
                "water_density": water_density,
            }
        }
    return OmegaConf.create({"processes": processes})


def _state():
    thickness = tf.constant([[100.0, 200.0]], tf.float32)
    return SimpleNamespace(
        thk=thickness,
        topg=tf.constant([[-150.0, 20.0]], tf.float32),
        water_level=tf.constant(0.0, tf.float32),
        ubar=tf.zeros_like(thickness),
        vbar=tf.zeros_like(thickness),
        smb=tf.zeros_like(thickness),
        dx=tf.constant(1000.0),
        dt=tf.constant(1.0),
        it=0,
    )


def test_consistent_iceflow_density_ratio_is_accepted():
    state = _state()
    cfg = _cfg(0.893, ice_density=918.0, water_density=1028.0)

    thk_module.initialize(cfg, state)

    expected_lower = np.maximum(
        np.asarray(state.topg), -0.893 * np.asarray(state.thk)
    )
    np.testing.assert_allclose(state.lsurf, expected_lower)
    np.testing.assert_allclose(state.usurf, expected_lower + state.thk)


def test_inconsistent_iceflow_density_ratio_fails_before_tracing_transport():
    cfg = _cfg(0.91, ice_density=917.0, water_density=1027.0)

    with pytest.raises(ValueError, match="Inconsistent flotation densities"):
        thk_module.initialize(cfg, _state())


def test_nonpositive_thickness_density_ratio_is_rejected_without_iceflow():
    with pytest.raises(ValueError, match="ratio_density must be positive"):
        thk_module.initialize(_cfg(0.0), _state())


@pytest.mark.parametrize(
    ("ice_density", "water_density", "message"),
    [
        (0.0, 1000.0, "ice_density must be positive"),
        (900.0, 0.0, "water_density must be positive"),
    ],
)
def test_nonpositive_iceflow_densities_are_rejected(
    ice_density, water_density, message
):
    cfg = _cfg(
        0.9, ice_density=ice_density, water_density=water_density
    )
    with pytest.raises(ValueError, match=message):
        thk_module.initialize(cfg, _state())
