#!/usr/bin/env python3

"""Tests for composable thickness active-domain constraints."""

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import tensorflow as tf

from igm.processes.thk import thk as thk_module
from igm.processes.thk import transport
from igm.processes.thk.domains import update_active_domain
from igm.processes.time.time import compute_dt_from_cfl


def _cfg(constraints, scheme="explicit"):
    return OmegaConf.create(
        {
            "processes": {
                "thk": {
                    "scheme": scheme,
                    "slope_type": "superbee",
                    "divflux_smooth_sigma": 0.0,
                    "ratio_density": 0.9,
                    "domain": {"constraints": constraints},
                    "calving_front": False,
                }
            }
        }
    )


def _state(ny=5, nx=6):
    thickness = tf.ones((ny, nx), dtype=tf.float32) * 10.0
    return SimpleNamespace(
        thk=thickness,
        topg=tf.ones_like(thickness),
        ubar=tf.ones_like(thickness),
        vbar=tf.zeros_like(thickness),
        smb=tf.ones_like(thickness) * 2.0,
        dx=tf.constant(1.0),
        dt=tf.constant(0.1),
        it=0,
    )


def test_constraints_are_intersected_and_published_on_state():
    state = _state(ny=5, nx=5)
    state.basinmask = tf.constant(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=tf.float32,
    )
    cfg = _cfg(
        [
            {"method": "grounded"},
            {"method": "initial_ice", "min_thickness": 1.0e-3},
            {"method": "interior", "field": "basinmask"},
        ]
    )

    thk_module.initialize(cfg, state)

    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    np.testing.assert_array_equal(state.thk_active_mask, expected)
    np.testing.assert_array_equal(state.groundedmask, np.ones((5, 5)))
    np.testing.assert_array_equal(
        state.thk_initial_ice_mask, np.ones((5, 5), dtype=bool)
    )


def test_empty_constraints_leave_no_mask_or_default_path_allocation():
    state = _state()
    state.thk_active_mask = tf.zeros_like(state.thk, dtype=tf.bool)
    cfg = _cfg([])

    thk_module.initialize(cfg, state)
    assert not hasattr(state, "thk_active_mask")

    thk_module.update(cfg, state)
    assert not hasattr(state, "thk_active_mask")


def test_explicit_transport_freezes_inactive_cells_and_blocks_internal_flux():
    state = _state(ny=3, nx=5)
    state.active = tf.constant(
        [[0, 1, 1, 1, 0]] * 3, dtype=tf.float32
    )
    cfg = _cfg([{"method": "state_mask", "field": "active"}])
    thickness_old = tf.identity(state.thk)

    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)

    inactive = tf.logical_not(state.thk_active_mask)
    np.testing.assert_allclose(
        tf.boolean_mask(state.thk, inactive),
        tf.boolean_mask(thickness_old, inactive),
    )
    np.testing.assert_allclose(
        tf.boolean_mask(state.divflux, inactive), 0.0, atol=1.0e-7
    )

    # No mass crosses the internal active-domain boundary.  Uniform thickness
    # and velocity can only redistribute mass among the three active columns;
    # SMB is consequently their only net volume change.
    active_change = tf.reduce_sum(
        tf.boolean_mask(state.thk - thickness_old, state.thk_active_mask)
    )
    expected_change = (
        state.dt
        * tf.reduce_sum(tf.boolean_mask(state.smb, state.thk_active_mask))
    )
    np.testing.assert_allclose(active_change, expected_change, atol=2.0e-6)


def test_grounded_constraint_tracks_live_geometry():
    state = _state(ny=1, nx=3)
    state.topg = tf.constant([[-20.0, -5.0, 1.0]])
    state.thk = tf.constant([[10.0, 10.0, 10.0]])
    cfg = _cfg([{"method": "grounded"}])

    thk_module.initialize(cfg, state)
    np.testing.assert_array_equal(
        state.thk_active_mask, [[False, True, True]]
    )

    state.thk = tf.constant([[30.0, 1.0, 10.0]])
    update_active_domain(cfg, state, state.thk_components.domain_constraints)
    np.testing.assert_array_equal(
        state.thk_active_mask, [[True, False, True]]
    )


def test_backend_must_explicitly_support_active_domains(monkeypatch):
    class NoDomainTransport:
        @staticmethod
        def initialize(cfg, state):
            pass

        @staticmethod
        def update(cfg, state):
            pass

    monkeypatch.setitem(
        transport.TransportSchemes, "no_domain", NoDomainTransport
    )
    cfg = _cfg([{"method": "grounded"}], scheme="no_domain")
    with pytest.raises(ValueError, match="does not support active-domain"):
        thk_module.initialize(cfg, _state())


def test_unknown_constraint_lists_available_choices():
    cfg = _cfg([{"method": "coastline"}])
    with pytest.raises(ValueError, match="available constraints"):
        thk_module.initialize(cfg, _state())


def test_cfl_ignores_velocities_outside_the_active_domain():
    ubar = tf.constant([[1.0, 100.0]])
    active = tf.constant([[True, False]])

    dt = compute_dt_from_cfl(
        ubar,
        tf.zeros_like(ubar),
        0.5,
        tf.constant(10.0),
        20.0,
        active_mask=active,
    )

    assert float(dt) == pytest.approx(5.0)
