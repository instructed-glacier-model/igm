#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Tests for conservative flux-form semi-Lagrangian thickness transport."""

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import tensorflow as tf

from igm.processes.thk.transport import ffsl
from igm.processes.thk import thk as thk_module


def _cfg(max_deformation=0.5, max_substeps=128):
    return OmegaConf.create(
        {
            "processes": {
                "thk": {
                    "scheme": "ffsl",
                    "slope_type": "superbee",
                    "divflux_smooth_sigma": 0.0,
                    "ratio_density": 0.91,
                    "ffsl": {
                        "max_deformation": max_deformation,
                        "max_substeps": max_substeps,
                    },
                    "calving_front": False,
                    "method": "sub_grid",
                }
            }
        }
    )


def _state(thickness, ubar=None, vbar=None, dx=1.0, dt=1.0, smb=None):
    thickness = tf.convert_to_tensor(thickness, tf.float32)
    zeros = tf.zeros_like(thickness)
    return SimpleNamespace(
        thk=thickness,
        topg=zeros,
        ubar=zeros if ubar is None else tf.cast(ubar, thickness.dtype),
        vbar=zeros if vbar is None else tf.cast(vbar, thickness.dtype),
        smb=zeros if smb is None else tf.cast(smb, thickness.dtype),
        dx=tf.constant(dx, thickness.dtype),
        dt=tf.constant(dt, thickness.dtype),
        it=0,
    )


def _solve(state, cfg=None):
    return ffsl.solve(state, _cfg() if cfg is None else cfg, state.smb)


def test_uniform_translation_is_exact_for_a_large_integer_cfl():
    ny, nx = 8, 80
    x = np.arange(nx, dtype=np.float32)
    profile = np.exp(-0.08 * (x - 25.0) ** 2).astype("float32")
    thickness = np.broadcast_to(profile, (ny, nx)).copy()
    speed, dt = 4.0, 3.0
    shift = int(speed * dt)
    state = _state(
        thickness,
        ubar=tf.fill((ny, nx), speed),
        dt=dt,
    )

    result = _solve(state)
    expected = np.pad(thickness[:, :-shift], ((0, 0), (shift, 0)))

    np.testing.assert_allclose(result.thickness, expected, rtol=2.0e-6, atol=2.0e-6)
    assert int(result.substeps) == 1
    assert float(result.max_deformation) == 0.0
    assert float(tf.reduce_min(result.thickness)) >= 0.0
    np.testing.assert_allclose(
        tf.reduce_sum(result.thickness), tf.reduce_sum(state.thk), rtol=2.0e-6
    )


def test_closed_deforming_flow_is_positive_and_conservative_beyond_cfl_one():
    ny, nx = 48, 64
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    thickness = 20.0 + 100.0 * np.exp(-6.0 * (xx**2 + yy**2))
    # The velocity vanishes at both x boundaries, making every row closed.
    ubar = 5.0 * np.sin(np.pi * (xx + 1.0) / 2.0)
    state = _state(thickness, ubar=ubar, dx=1.0, dt=4.0)

    result = _solve(state)
    relative_mass_error = tf.abs(
        tf.reduce_sum(result.thickness) - tf.reduce_sum(state.thk)
    ) / tf.reduce_sum(state.thk)

    assert 1 < int(result.substeps) < 128
    assert float(tf.reduce_min(result.thickness)) >= 0.0
    assert bool(tf.reduce_all(tf.math.is_finite(result.thickness)))
    assert float(relative_mass_error) < 2.0e-6


def test_zero_velocity_integrates_smb_and_reports_limited_ablation():
    thickness = tf.fill((8, 80), 1.0)
    state = _state(thickness, dt=2.0, smb=tf.fill((8, 80), -1.0))

    result = _solve(state)

    np.testing.assert_allclose(result.thickness, 0.0, atol=1.0e-7)
    expected_correction = float(np.prod(thickness.shape))
    assert float(result.source_limiter_volume) == pytest.approx(
        expected_correction, rel=2.0e-6
    )
    np.testing.assert_allclose(result.divflux, -0.5, atol=2.0e-6)


@pytest.mark.slow
def test_solid_body_rotation_retains_a_gaussian_at_large_cfl():
    resolution = 64
    domain_size = 100_000.0
    coordinates = np.linspace(
        -0.5 * domain_size,
        0.5 * domain_size,
        resolution,
        dtype=np.float32,
    )
    xx, yy = np.meshgrid(coordinates, coordinates)
    dx = float(coordinates[1] - coordinates[0])
    angular_velocity = 2.0 * np.pi / 1000.0
    ubar = tf.constant(-angular_velocity * yy, tf.float32)
    vbar = tf.constant(angular_velocity * xx, tf.float32)
    thickness_initial = tf.constant(
        1000.0
        * np.exp(
            -0.5
            * ((xx - 0.2 * domain_size) ** 2 + yy**2)
            / (0.06 * domain_size) ** 2
        ),
        tf.float32,
    )
    maximum_speed = float(tf.reduce_max(tf.abs(ubar) + tf.abs(vbar)))
    target_dt = 4.0 * dx / maximum_speed
    steps = int(np.ceil(1000.0 / target_dt))
    dt = 1000.0 / steps
    state = _state(
        thickness_initial, ubar=ubar, vbar=vbar, dx=dx, dt=dt
    )

    thickness = state.thk
    for step in range(steps):
        state.thk = thickness
        state.it = step
        thickness = _solve(state).thickness

    relative_l2_error = tf.norm(thickness - thickness_initial) / tf.norm(
        thickness_initial
    )
    relative_mass_error = tf.abs(
        tf.reduce_sum(thickness) - tf.reduce_sum(thickness_initial)
    ) / tf.reduce_sum(thickness_initial)
    peak_ratio = tf.reduce_max(thickness) / tf.reduce_max(thickness_initial)

    assert float(relative_l2_error) < 0.12
    assert float(relative_mass_error) < 2.0e-5
    assert float(peak_ratio) > 0.85
    assert float(tf.reduce_min(thickness)) >= 0.0


def test_public_thk_dispatch_exposes_device_diagnostics():
    state = _state(tf.ones((8, 80)), ubar=tf.ones((8, 80)), dt=5.0)
    cfg = _cfg()

    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)

    assert state.thk.shape == (8, 80)
    assert state.divflux.shape == (8, 80)
    assert state.ffsl_substeps.dtype == tf.int32
    assert state.ffsl_substep_limit_reached.dtype == tf.bool
    assert state.ffsl_source_limiter_volume.shape == ()
    assert bool(state.thk_step_accepted)


def test_public_ffsl_step_stops_and_reverts_when_substep_cap_is_hit():
    nx = 80
    x = tf.linspace(-1.0, 1.0, nx)
    state = _state(
        tf.ones((8, nx)),
        ubar=tf.broadcast_to(10.0 * x[None, :], (8, nx)),
        dt=10.0,
    )
    thickness_old = tf.identity(state.thk)
    cfg = _cfg(max_deformation=0.1, max_substeps=3)

    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)

    assert bool(state.ffsl_substep_limit_reached)
    assert int(state.ffsl_required_substeps) > 3
    assert int(state.ffsl_substeps) == 3
    assert not bool(state.thk_step_accepted)
    assert not bool(state.continue_run)
    np.testing.assert_allclose(state.thk, thickness_old)
    np.testing.assert_allclose(state.divflux, state.smb)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    (
        ("max_deformation", 0.0, "max_deformation must be positive"),
        ("max_deformation", float("nan"), "max_deformation must be positive"),
        ("max_substeps", 0, "max_substeps must be at least one"),
    ),
)
def test_invalid_ffsl_options_are_rejected(option, value, message):
    cfg = _cfg()
    cfg.processes.thk.ffsl[option] = value

    with pytest.raises(ValueError, match=message):
        ffsl.initialize(cfg, _state(tf.ones((8, 80))))


def test_invalid_ffsl_limit_policy_is_rejected():
    cfg = _cfg()
    cfg.processes.thk.ffsl.limit_policy = "truncate"
    with pytest.raises(ValueError, match="limit_policy must be"):
        thk_module.initialize(cfg, _state(tf.ones((8, 80))))


def test_ffsl_rejects_non_superbee_slope():
    cfg = _cfg()
    cfg.processes.thk.slope_type = "godunov"
    with pytest.raises(ValueError, match="ffsl currently requires.*superbee"):
        ffsl.initialize(cfg, _state(tf.ones((8, 80))))


def test_scalar_parameters_do_not_retrace_the_compiled_kernel():
    state = _state(tf.ones((8, 80)), ubar=tf.ones((8, 80)))
    before = ffsl._ffsl_step.experimental_get_tracing_count()

    for dt, step, limit, cap in (
        (1.0, 0, 0.5, 128),
        (8.0, 1, 0.25, 64),
        (0.2, 7, 0.8, 16),
    ):
        ffsl._ffsl_step(
            state.ubar,
            state.vbar,
            state.thk,
            state.dx,
            tf.constant(dt),
            state.smb,
            tf.constant(step),
            tf.constant(limit),
            tf.constant(cap),
        )

    after = ffsl._ffsl_step.experimental_get_tracing_count()
    assert after - before <= 1


def test_periodic_boundary_wraps_an_arbitrary_cfl_translation():
    ny, nx = 8, 80
    x = np.arange(nx, dtype=np.float32)
    thickness = np.broadcast_to(
        np.exp(-0.08 * (x - 73.0) ** 2), (ny, nx)
    ).copy()
    speed, dt = 4.0, 3.0
    state = _state(
        thickness,
        ubar=tf.fill((ny, nx), speed),
        dt=dt,
    )
    cfg = _cfg()
    cfg.processes.thk.boundary = {
        "left": "periodic",
        "right": "periodic",
        "top": "zero",
        "bottom": "zero",
    }

    result = _solve(state, cfg)

    expected = np.roll(thickness, int(speed * dt), axis=1)
    np.testing.assert_allclose(result.thickness, expected, rtol=3.0e-6, atol=3.0e-6)
    np.testing.assert_allclose(
        tf.reduce_sum(result.thickness), tf.reduce_sum(thickness), rtol=2.0e-6
    )


def test_symmetric_boundary_is_no_flux_for_ffsl():
    ny, nx = 6, 48
    thickness = tf.reshape(tf.linspace(1.0, 3.0, ny * nx), (ny, nx))
    state = _state(
        thickness,
        ubar=tf.ones((ny, nx)),
        dx=1.0,
        dt=2.0,
    )
    cfg = _cfg()
    cfg.processes.thk.boundary = {
        "left": "symmetric",
        "right": "symmetric",
        "top": "zero",
        "bottom": "zero",
    }

    result = _solve(state, cfg)

    assert int(result.substeps) >= 4
    assert float(tf.reduce_min(result.thickness)) >= 0.0
    np.testing.assert_allclose(
        tf.reduce_sum(result.thickness), tf.reduce_sum(thickness), rtol=3.0e-6
    )


def test_ffsl_supports_a_symmetric_left_and_open_right_side():
    ny, nx = 5, 40
    thickness = tf.ones((ny, nx), tf.float32)
    state = _state(
        thickness,
        ubar=-tf.ones((ny, nx)),
        dx=1.0,
        dt=1.0,
    )
    mixed_cfg = _cfg()
    mixed_cfg.processes.thk.boundary = {
        "left": "symmetric",
        "right": "zero",
        "top": "zero",
        "bottom": "zero",
    }
    open_cfg = _cfg()
    open_cfg.processes.thk.boundary = {
        "left": "zero",
        "right": "zero",
        "top": "zero",
        "bottom": "zero",
    }

    mixed = _solve(state, mixed_cfg)
    opened = _solve(state, open_cfg)

    np.testing.assert_allclose(
        tf.reduce_sum(mixed.thickness), tf.reduce_sum(thickness), rtol=3.0e-6
    )
    np.testing.assert_allclose(
        tf.reduce_sum(opened.thickness),
        tf.reduce_sum(thickness) - ny,
        rtol=3.0e-6,
    )
