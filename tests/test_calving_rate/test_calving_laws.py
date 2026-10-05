#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""The calving_rate process: laws, strain rates, front band, and the time step."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

import igm
from igm.common.runner.modules.src import check_module_needs
from igm.processes.calving_rate import calving_rate
from igm.processes.calving_rate import geometry as calving_geometry
from igm.processes.calving_rate.laws import available_calving_laws
from igm.processes.calving_rate.strain_rates import principal_strain_rates
from igm.processes.thk.surfaces import update_surfaces
from igm.processes.time.time import compute_dt_from_cfl

pytestmark = [pytest.mark.fast, pytest.mark.unit]

CONF = Path(igm.__file__).parent / "conf" / "processes"
DX = 1000.0
RATIO = 0.9
EXX, EYY = 2.0e-3, 1.0e-3  # 1/yr


def _cfg(law, **options):
    cfg = OmegaConf.merge(
        OmegaConf.load(CONF / "calving_rate.yaml"),
        OmegaConf.load(CONF / "thk.yaml"),
    )
    cfg.calving_rate.law = law
    for key, value in options.items():
        OmegaConf.update(cfg, f"calving_rate.{key}", value)
    cfg.thk.ratio_density = RATIO
    return OmegaConf.create({"processes": cfg})


def _shelf(n=21, front=12, hole=None, land=None):
    """A floating shelf x < front with u = EXX x, v = EYY y (spreading both ways)."""
    y, x = (np.mgrid[0:n, 0:n] * DX).astype(np.float32)
    thk = np.where(np.arange(n)[None, :] < front, 300.0, 0.0).repeat(n, 0)
    topg = np.full((n, n), -1000.0, np.float32)
    if hole is not None:
        thk[hole] = 0.0
    if land is not None:
        topg[land] = 50.0
    state = SimpleNamespace(
        thk=tf.constant(thk.astype(np.float32)),
        topg=tf.constant(topg),
        water_level=tf.zeros((n, n)),
        dx=tf.constant(DX),
        ubar=tf.constant(EXX * x),
        vbar=tf.constant(EYY * y),
        arrhenius=tf.fill((n, n), 4.6),
    )
    update_surfaces(
        OmegaConf.create({"processes": {"thk": {"ratio_density": RATIO}}}), state
    )
    return state


def _run(cfg, state):
    calving_rate.initialize(cfg, state)
    calving_rate.update(cfg, state)
    return state.calving_rate.numpy()


def test_every_law_is_available():
    assert available_calving_laws() == (
        "constant",
        "eigen",
        "ice_speed",
        "von_mises",
        "water_depth",
        "zero",
    )


def test_strain_rates_are_exact_at_the_front_for_a_linear_velocity():
    state = _shelf()
    mask = state.thk > 0.0
    e1, e2 = principal_strain_rates(state.ubar, state.vbar, mask, tf.constant(DX))
    ice = mask.numpy()
    np.testing.assert_allclose(e1.numpy()[ice], EXX, rtol=1e-4)
    np.testing.assert_allclose(e2.numpy()[ice], EYY, rtol=1e-4)
    assert np.all(e1.numpy()[~ice] == 0.0)


def test_strain_rates_are_second_order_at_the_front():
    """u = a x + b x**2: exact at the front node (first order is 0.45 % off)."""
    state = _shelf()
    x = np.arange(21) * DX
    b = 1.0e-8  # 1/(m yr)
    u = tf.constant(np.tile(EXX * x + b * x**2, (21, 1)).astype(np.float32))
    mask = state.thk > 0.0
    e1, _ = principal_strain_rates(u, tf.zeros_like(u), mask, tf.constant(DX))
    exact = EXX + 2.0 * b * x
    np.testing.assert_allclose(e1.numpy()[5, 11], exact[11], rtol=1e-4)  # front
    np.testing.assert_allclose(e1.numpy()[5, 1:11], exact[1:11], rtol=1e-4)


def test_eigencalving_on_floating_ice_is_carried_to_the_ice_free_front_cells():
    K = 1.0e3
    rate = _run(_cfg("eigen", **{"eigen.K": K}), _shelf())
    expected = K * EXX * EYY
    # Ice-free front cells (column 12) and the ice next to the front.
    np.testing.assert_allclose(rate[3:-3, 12], expected, rtol=1e-4)
    np.testing.assert_allclose(rate[3:-3, 11], expected, rtol=1e-4)
    # Zero outside the front band (3 cells on each side).
    assert np.all(rate[:, :8] == 0.0) and np.all(rate[:, 16:] == 0.0)


def test_water_depth_law_and_land_are_respected():
    land = (slice(None), slice(14, None))
    rate = _run(_cfg("water_depth", **{"water_depth.k": 2.0}), _shelf(land=land))
    np.testing.assert_allclose(rate[5, 12], 2.0 * 1000.0, rtol=1e-6)
    assert np.all(rate[:, 14:] == 0.0)


def test_von_mises_uses_the_hardness_and_the_tensile_strain_rate():
    cfg = _cfg("von_mises", **{"von_mises.sigma_max_floating": 0.2})
    state = _shelf()
    rate = _run(cfg, state)
    j = 11  # last ice column
    i = 10
    speed = np.hypot(EXX * j * DX, EYY * i * DX)
    tensile = np.sqrt(0.5 * (EXX**2 + EYY**2))
    sigma = np.sqrt(3.0) * 4.6 ** (-1.0 / 3.0) * tensile ** (1.0 / 3.0)
    assert rate[i, j] == pytest.approx(speed * sigma / 0.2, rel=1e-4)


def test_ice_speed_law_prescribes_the_front_velocity():
    """The rate uses the speed of the ice at the front, which moves the front."""
    cfg = _cfg("ice_speed", **{"ice_speed.front_velocity": 5.0})
    state = _shelf()
    rate = _run(cfg, state)
    speed = np.hypot(EXX * 11 * DX, EYY * 10 * DX)
    assert rate[10, 11] == pytest.approx(speed - 5.0, rel=1e-5)
    # The ice-free front cell: the velocity extrapolated linearly (u = EXX x).
    front = np.hypot(EXX * 12 * DX, EYY * 10 * DX)
    assert rate[10, 12] == pytest.approx(front - 5.0, rel=1e-5)


def test_compiled_geometry_reads_current_tensors_without_retracing():
    cfg = _cfg("ice_speed")
    state = _shelf()
    calving_rate.initialize(cfg, state)
    baseline = state.calving_rate.numpy().copy()
    trace_count = calving_geometry._front_geometry.experimental_get_tracing_count()

    state.ubar = state.ubar + tf.constant(100.0, state.ubar.dtype)
    calving_rate.update(cfg, state)

    assert not np.array_equal(state.calving_rate.numpy(), baseline)
    assert (
        calving_geometry._front_geometry.experimental_get_tracing_count() == trace_count
    )


def test_a_hole_in_the_shelf_does_not_calve():
    hole = (slice(9, 12), slice(4, 7))
    rate = _run(_cfg("constant", **{"constant.value": 100.0}), _shelf(hole=hole))
    assert np.all(rate[8:13, 3:8] == 0.0)
    assert rate[10, 12] == pytest.approx(100.0)
    rate = _run(
        _cfg("constant", **{"constant.value": 100.0, "ocean_connected_only": False}),
        _shelf(hole=hole),
    )
    assert rate[10, 5] == pytest.approx(100.0)


def test_frontal_melt_and_the_time_step():
    cfg = _cfg(
        "zero", **{"frontal_melt.method": "constant", "frontal_melt.value": 30.0}
    )
    state = _shelf()
    _run(cfg, state)
    assert state.frontal_melt_rate.numpy()[10, 12] == pytest.approx(30.0)
    assert np.all(state.calving_rate.numpy() == 0.0)
    front = tf.fill((3, 3), 4000.0)
    dt = compute_dt_from_cfl(
        tf.fill((3, 3), 100.0), tf.zeros((3, 3)), 0.5, DX, 10.0, ablation_speed=front
    )
    assert float(dt) == pytest.approx(0.5 * DX / 4000.0)


def test_needs_resolve_to_the_active_law():
    cfg = _cfg("von_mises")
    state = _shelf()
    del state.arrhenius
    state.calving_rate = tf.zeros_like(state.thk)  # as after initialization
    with pytest.raises(Exception, match="arrhenius"):
        check_module_needs([calving_rate], state, cfg)


def test_von_mises_ignores_the_enhancement_factor_already_in_arrhenius():
    """``state.arrhenius`` includes E, so the law must not apply it again."""
    reference = _run(_cfg("von_mises"), _shelf())
    cfg = _cfg("von_mises")
    OmegaConf.update(cfg, "processes.iceflow.physics.viscosity.enhancement_factor", 3.0)
    OmegaConf.update(cfg, "processes.iceflow.physics.viscosity.exponent", 3.0)
    rate = _run(cfg, _shelf())
    np.testing.assert_allclose(rate, reference, rtol=1e-6)


def test_von_mises_averages_a_3d_arrhenius_with_the_vertical_weights():
    """A 3-D rate factor is averaged as B with the ice-flow weights, or uniformly."""
    state = _shelf()
    n = int(state.thk.shape[0])
    A = np.stack([np.full((n, n), 2.0), np.full((n, n), 16.0)]).astype(np.float32)
    state.arrhenius = tf.constant(A)
    weights = tf.constant(np.array([0.75, 0.25], np.float32).reshape(2, 1, 1))
    state.iceflow = SimpleNamespace(
        discr_v=SimpleNamespace(enthalpy=SimpleNamespace(weights=weights))
    )
    i, j = 10, 11  # last ice column
    speed = np.hypot(EXX * j * DX, EYY * i * DX)
    tensile = np.sqrt(0.5 * (EXX**2 + EYY**2))
    sigma_over_b = np.sqrt(3.0) * tensile ** (1.0 / 3.0)

    rate = _run(_cfg("von_mises"), state)
    B = 0.75 * 2.0 ** (-1.0 / 3.0) + 0.25 * 16.0 ** (-1.0 / 3.0)
    assert rate[i, j] == pytest.approx(speed * B * sigma_over_b / 0.15, rel=1e-4)

    del state.iceflow  # without the ice-flow weights: uniform mean of B
    rate = _run(_cfg("von_mises"), state)
    B = 0.5 * (2.0 ** (-1.0 / 3.0) + 16.0 ** (-1.0 / 3.0))
    assert rate[i, j] == pytest.approx(speed * B * sigma_over_b / 0.15, rel=1e-4)


def test_the_rate_is_capped_at_max_rate():
    rate = _run(
        _cfg("constant", **{"constant.value": 500.0, "max_rate": 123.0}), _shelf()
    )
    assert np.max(rate) == pytest.approx(123.0)


def test_parameter_guards_fail_loudly():
    with pytest.raises(ValueError, match="sigma_max"):
        calving_rate.initialize(
            _cfg("von_mises", **{"von_mises.sigma_max_floating": 0.0}), _shelf()
        )
    with pytest.raises(ValueError, match="K must be"):
        calving_rate.initialize(_cfg("eigen", **{"eigen.K": -1.0}), _shelf())
    with pytest.raises(ValueError, match="was removed"):
        calving_rate.initialize(_cfg("zero", Hcr=250.0), _shelf())
    with pytest.raises(ValueError, match="min_thickness"):
        calving_rate.initialize(_cfg("thickness_threshold"), _shelf())
    cfg = _cfg("zero", band=2)
    OmegaConf.update(cfg, "processes.thk.front.method", "level_set")
    with pytest.raises(ValueError, match="level_set.band"):
        calving_rate.initialize(cfg, _shelf())
