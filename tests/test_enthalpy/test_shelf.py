#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Basal boundary condition of the ice in contact with the ocean."""

import os

import numpy as np
import pytest
import tensorflow as tf

import igm
from igm.common import State
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes import enthalpy

pytestmark = [pytest.mark.fast, pytest.mark.unit]

NY, NX = 2, 6  # three grounded columns, then three floating ones


def _setup(water_level=True, salinity=None):
    cfg = load_yaml_recursive(
        os.path.join(igm.__path__[0], "conf"), exclude=["assimilations/pretraining"]
    )
    cfg.processes.iceflow.physics.water_density = 1028.0
    nz_u = cfg.processes.iceflow.numerics.Nz
    thk = np.full((NY, NX), 400.0, np.float32)
    topg = np.where(np.arange(NX) < 3, 100.0, -800.0) * np.ones((NY, 1), np.float32)
    rho = cfg.processes.iceflow.physics.ice_density / 1028.0
    lsurf = np.maximum(topg, -rho * thk).astype(np.float32)

    state = State()
    state.x = tf.constant(np.arange(NX, dtype=np.float32) * 1000.0)
    state.y = tf.constant(np.arange(NY, dtype=np.float32) * 1000.0)
    state.dx = tf.Variable(1000.0, trainable=False)
    state.dX = tf.Variable(tf.fill((NY, NX), 1000.0), trainable=False)
    state.thk = tf.Variable(thk)
    state.topg = tf.Variable(topg.astype(np.float32))
    state.lsurf = tf.constant(lsurf)
    state.usurf = tf.Variable(lsurf + thk)
    if water_level:
        state.water_level = tf.zeros((NY, NX))
    if salinity is not None:
        state.ocean_salinity = tf.fill((NY, NX), salinity)
    state.t = tf.Variable(0.0, trainable=False)
    state.dt = tf.Variable(10.0, trainable=False)
    state.air_temp = tf.fill((1, NY, NX), -20.0)
    state.basal_heat_flux = tf.fill((NY, NX), 0.1)
    state.h_water_till = tf.zeros((NY, NX))
    state.tau_ref = tf.zeros((NY, NX))
    state.U = tf.zeros((nz_u, NY, NX))
    state.V = tf.zeros((nz_u, NY, NX))
    state.W = tf.zeros((nz_u, NY, NX))
    enthalpy.initialize(cfg, state)
    return cfg, state


def _run(setup, steps=20):
    cfg, state = setup
    for _ in range(steps):
        enthalpy.update(cfg, state)
    return cfg, state


def test_floating_ice_takes_the_ice_melting_point_at_its_base():
    cfg, state = _run(_setup())
    E_pmp, _ = enthalpy.temperature.compute_pmp(cfg, state)
    np.testing.assert_allclose(state.E.numpy()[0, :, 3:], E_pmp.numpy()[0, :, 3:])
    np.testing.assert_array_equal(state.basal_melt_rate.numpy()[:, 3:], 0.0)

    # The grounded columns are those of a domain without ocean.
    _, land = _run(_setup(water_level=False))
    np.testing.assert_array_equal(state.E.numpy()[:, :, :3], land.E.numpy()[:, :, :3])


def test_floating_ice_takes_the_freezing_point_of_sea_water():
    cfg, state = _run(_setup(salinity=34.5))
    thermal = cfg.processes.enthalpy.thermal
    draft = state.lsurf.numpy()[:, 3:]
    T_f = -0.0573 * 34.5 + 0.0832 + 7.61e-4 * draft + thermal.T_pmp_ref
    expected = thermal.c_ice * (T_f - thermal.T_ref)
    np.testing.assert_allclose(state.E.numpy()[0, :, 3:], expected, rtol=1e-5)
    E_pmp, _ = enthalpy.temperature.compute_pmp(cfg, state)
    assert (state.E.numpy()[0, :, 3:] < E_pmp.numpy()[0, :, 3:]).all()
