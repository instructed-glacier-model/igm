#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Tests for shared thickness-transport boundary policies."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import numpy as np
import pytest
import tensorflow as tf

from igm.processes.thk import boundary
from igm.processes.thk import thk as thk_module


def _cfg(scheme="explicit", boundary_value=None):
    thk = {
        "scheme": scheme,
        "slope_type": "superbee",
        "divflux_smooth_sigma": 0.0,
        "ratio_density": 0.91,
        "calving_front": False,
    }
    if boundary_value is not None:
        thk["boundary"] = boundary_value
    return OmegaConf.create({"processes": {"thk": thk}})


def _state():
    thickness = tf.ones((4, 7), tf.float32)
    return SimpleNamespace(thk=thickness, topg=tf.zeros_like(thickness), it=0)


def test_side_modes_and_readable_aliases_are_normalized():
    cfg = _cfg(
        boundary_value={
            "left": "closed",
            "right": "open",
            "top": "zero",
            "bottom": "reflective",
        }
    )
    assert boundary.get_boundary_conditions(cfg) == boundary.BoundaryConditions(
        "symmetric", "zero", "zero", "symmetric"
    )


def test_omitted_boundary_defaults_every_side_to_zero():
    cfg = _cfg()
    assert boundary.get_boundary_conditions(cfg) == boundary.BoundaryConditions(
        "zero", "zero", "zero", "zero"
    )


@pytest.mark.parametrize(
    "boundary_value",
    [
        {"x": "symmetric", "y": "zero"},
        "symmetric",
    ],
)
def test_axis_and_scalar_boundary_shorthands_are_rejected(boundary_value):
    cfg = _cfg(boundary_value=boundary_value)
    with pytest.raises(ValueError, match="left, right, top, and bottom"):
        boundary.get_boundary_conditions(cfg)


def test_partial_four_side_boundary_block_is_rejected():
    cfg = _cfg(boundary_value={"left": "symmetric", "right": "zero"})
    with pytest.raises(ValueError, match="must define all four sides"):
        boundary.get_boundary_conditions(cfg)


def test_legacy_floating_boundary_options_are_rejected():
    cfg = _cfg()
    cfg.processes.thk.flux_mode_h = "symmetric"
    with pytest.raises(ValueError, match="flux_mode_h.*not supported"):
        boundary.get_boundary_conditions(cfg)


def test_face_velocities_encode_open_symmetric_and_periodic_semantics():
    ubar = tf.constant([[2.0, 4.0, 8.0]])
    vbar = tf.zeros_like(ubar)

    open_u, _ = boundary.face_velocities(ubar, vbar)
    mixed_u, _ = boundary.face_velocities(
        ubar, vbar, "symmetric", "zero", "zero", "zero"
    )
    opposite_mixed_u, _ = boundary.face_velocities(
        ubar, vbar, "zero", "symmetric", "zero", "zero"
    )
    periodic_u, _ = boundary.face_velocities(
        ubar, vbar, "periodic", "periodic", "zero", "zero"
    )

    np.testing.assert_allclose(open_u, [[2.0, 3.0, 6.0, 8.0]])
    np.testing.assert_allclose(mixed_u, [[0.0, 3.0, 6.0, 8.0]])
    np.testing.assert_allclose(opposite_mixed_u, [[2.0, 3.0, 6.0, 0.0]])
    np.testing.assert_allclose(periodic_u, [[5.0, 3.0, 6.0, 5.0]])

    vbar = tf.constant([[2.0], [4.0], [8.0]])
    ubar = tf.zeros_like(vbar)
    _, mixed_v = boundary.face_velocities(
        ubar, vbar, "zero", "zero", "symmetric", "zero"
    )
    _, opposite_mixed_v = boundary.face_velocities(
        ubar, vbar, "zero", "zero", "zero", "symmetric"
    )
    _, periodic_v = boundary.face_velocities(
        ubar, vbar, "zero", "zero", "periodic", "periodic"
    )
    np.testing.assert_allclose(mixed_v, [[0.0], [3.0], [6.0], [8.0]])
    np.testing.assert_allclose(opposite_mixed_v, [[2.0], [3.0], [6.0], [0.0]])
    np.testing.assert_allclose(periodic_v, [[5.0], [3.0], [6.0], [5.0]])


def test_unknown_boundary_fails_before_tracing_a_solver():
    cfg = _cfg(
        boundary_value={
            "left": "radiative",
            "right": "zero",
            "top": "zero",
            "bottom": "zero",
        }
    )
    with pytest.raises(ValueError, match="must be one of.*radiative"):
        boundary.get_boundary_conditions(cfg)


@pytest.mark.parametrize(
    "boundary_value",
    [
        {
            "left": "periodic",
            "right": "zero",
            "top": "zero",
            "bottom": "zero",
        },
        {
            "left": "zero",
            "right": "zero",
            "top": "periodic",
            "bottom": "symmetric",
        },
    ],
)
def test_periodic_boundaries_must_be_paired(boundary_value):
    cfg = _cfg(boundary_value=boundary_value)
    with pytest.raises(ValueError, match="Periodic [xy] boundaries must be paired"):
        boundary.get_boundary_conditions(cfg)


def test_explicit_backend_rejects_periodic_boundaries():
    cfg = _cfg(
        scheme="explicit",
        boundary_value={
            "left": "periodic",
            "right": "periodic",
            "top": "zero",
            "bottom": "zero",
        },
    )
    with pytest.raises(ValueError, match="explicit.*does not support.*periodic"):
        thk_module.initialize(cfg, _state())


@pytest.mark.parametrize("value", [-1.0, float("nan")])
def test_explicit_rejects_invalid_flux_smoothing(value):
    cfg = _cfg(scheme="explicit")
    cfg.processes.thk.divflux_smooth_sigma = value
    with pytest.raises(ValueError, match="must be finite and nonnegative"):
        thk_module.initialize(cfg, _state())


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
def test_explicit_symmetric_boundary_blocks_outflow_on_each_side(side):
    ny, nx = 5, 7
    thickness = tf.fill((ny, nx), 10.0)
    ubar = tf.zeros_like(thickness)
    vbar = tf.zeros_like(thickness)
    if side == "left":
        ubar = tf.concat([-tf.ones((ny, 1)), ubar[:, 1:]], axis=1)
    elif side == "right":
        ubar = tf.concat([ubar[:, :-1], tf.ones((ny, 1))], axis=1)
    elif side == "top":
        vbar = tf.concat([-tf.ones((1, nx)), vbar[1:, :]], axis=0)
    else:
        vbar = tf.concat([vbar[:-1, :], tf.ones((1, nx))], axis=0)

    state = SimpleNamespace(
        thk=thickness,
        topg=tf.zeros_like(thickness),
        ubar=ubar,
        vbar=vbar,
        smb=tf.zeros_like(thickness),
        dx=tf.constant(1.0),
        dt=tf.constant(0.1),
        it=0,
    )
    modes = {name: "zero" for name in ("left", "right", "top", "bottom")}
    modes[side] = "symmetric"
    cfg = _cfg(scheme="explicit", boundary_value=modes)

    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)

    np.testing.assert_allclose(
        tf.reduce_sum(state.thk), tf.reduce_sum(thickness), rtol=2.0e-6
    )


def test_explicit_symmetric_boundary_remains_no_flux_with_flux_smoothing():
    generator = tf.random.Generator.from_seed(41)
    thickness = 10.0 + generator.uniform((8, 11))
    state = SimpleNamespace(
        thk=thickness,
        topg=tf.zeros_like(thickness),
        ubar=generator.normal(thickness.shape),
        vbar=generator.normal(thickness.shape),
        smb=tf.zeros_like(thickness),
        dx=tf.constant(1.0),
        dt=tf.constant(0.01),
        it=0,
    )
    cfg = _cfg(
        scheme="explicit",
        boundary_value={name: "symmetric" for name in (
            "left", "right", "top", "bottom"
        )},
    )
    cfg.processes.thk.divflux_smooth_sigma = 0.8

    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)

    np.testing.assert_allclose(
        tf.reduce_sum(state.thk), tf.reduce_sum(thickness), rtol=2.0e-6
    )


@pytest.mark.parametrize("shape", [(1, 5), (4, 1), (1, 1)])
def test_explicit_symmetric_boundary_handles_single_cell_axes(shape):
    thickness = tf.fill(shape, 10.0)
    state = SimpleNamespace(
        thk=thickness,
        topg=tf.zeros_like(thickness),
        ubar=tf.ones_like(thickness),
        vbar=-tf.ones_like(thickness),
        smb=tf.zeros_like(thickness),
        dx=tf.constant(1.0),
        dt=tf.constant(0.1),
        it=0,
    )
    cfg = _cfg(
        scheme="explicit",
        boundary_value={
            name: "symmetric" for name in ("left", "right", "top", "bottom")
        },
    )

    thk_module.initialize(cfg, state)
    thk_module.update(cfg, state)

    np.testing.assert_allclose(
        tf.reduce_sum(state.thk), tf.reduce_sum(thickness), rtol=2.0e-6
    )
    assert float(tf.reduce_min(state.thk)) >= 0.0
