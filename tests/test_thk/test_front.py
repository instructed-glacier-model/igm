#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Numerics of the calving-front methods (sub_grid and level_set).

The velocity is prescribed and, as the ice flow does, zero off the ice nodes
that belong to an active Q1 cell; the ablation rate ``state.calving_rate`` is
prescribed as the calving_rate process would publish it.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from igm.processes.thk import thk as thk_module
from igm.processes.thk.fronts import sub_grid
from igm.processes.thk.fronts.common import threshold_thickness
from igm.processes.thk.fronts.level_set import fill_fraction, reinitialise
from igm.processes.thk.masks import (
    compute_grounded_mask,
    compute_node_ice_mask,
    iceflow_node_mask,
)
from igm.utils.math.neighbours import any_neighbour

METHODS = ("sub_grid", "level_set")
RATIO = 0.9  # ice / water density
DX = 1000.0
DT = 0.2

# Albrecht et al. (2011) shelf: Weertman's steady spreading profile.
RHO, RHO_W = 910.0, 1028.0
B_ICE = 1.9e8 * 1e-6 / 31556926.0 ** (1 / 3)  # MPa yr^(1/3)
WEERTMAN_C = (RHO * 9.81 * (1 - RHO / RHO_W) / (4 * B_ICE * 1e6)) ** 3
H0, U0 = 600.0, 300.0
Q0 = H0 * U0


def _weertman_thickness(x):
    return (4 * WEERTMAN_C * x / Q0 + H0**-4) ** -0.25


def _cfg(method, boundary=None, **front):
    boundary = boundary or {
        "left": "dirichlet",
        "right": "zero",
        "top": "symmetric",
        "bottom": "symmetric",
    }
    options = {
        "method": method,
        "first_order": True,
        "min_thickness": 0.0,
        "fixed": False,
        "sub_grid": {"max_iterations": 10, "residual": "redistribute"},
        "level_set": {"reinit_freq": 1, "reinit_iter": 5, "band": 3},
    }
    options.update(front)
    return OmegaConf.create(
        {
            "processes": {
                "thk": {
                    "scheme": "explicit",
                    "slope_type": "superbee",
                    "ratio_density": RATIO,
                    "boundary": boundary,
                    "remove_rigid_body_modes": False,
                    "front": options,
                },
                "calving_rate": {},
            }
        }
    )


def _state(thk, topg=-5000.0, dx=DX, dt=DT):
    thk = tf.constant(np.asarray(thk, np.float32))
    zeros = tf.zeros_like(thk)
    return SimpleNamespace(
        thk=thk,
        topg=zeros + topg,
        water_level=zeros,
        dx=tf.constant(dx),
        dt=tf.constant(dt),
        it=0,
        ubar=zeros,
        vbar=zeros,
        smb=zeros,
        calving_rate=zeros,
    )


def _ice_velocity(state, u, v=0.0):
    """Velocity on the ice-flow nodes only, as the unified evaluator returns it."""
    nodes = iceflow_node_mask(state.thk, state.usurf, state.water_level, 1.0 / RATIO)
    state.ubar = tf.where(nodes, u + tf.zeros_like(state.thk), 0.0)
    state.vbar = tf.where(nodes, v + tf.zeros_like(state.thk), 0.0)


def _volume(state):
    return float(tf.reduce_sum(state.thk + state.Href))


def _front_1d(state, dx=DX):
    """Front position (m, from the left edge of the first cell) on the middle row."""
    row = state.thk.numpy()[1]
    last = int(np.max(np.nonzero(row > 0.0)[0]))
    return (last + 1.0 + state.ice_area_fraction.numpy()[1, last + 1]) * dx


def _flowline(method, nx=60, n_ice=10, H=500.0, **front):
    thk = np.zeros((3, nx), np.float32)
    thk[:, :n_ice] = H
    cfg = _cfg(method, **front)
    state = _state(thk)
    thk_module.initialize(cfg, state)
    return cfg, state


def _run(cfg, state, steps, u, ablation=0.0):
    """Advance ``steps`` steps; check the mass budget closes every step."""
    inflow = u * 500.0 * DT / DX * state.thk.shape[0]  # Dirichlet ghost of 500 m
    for it in range(steps):
        state.it = it
        _ice_velocity(state, u)
        state.calving_rate = tf.fill(tf.shape(state.thk), ablation)
        before = _volume(state)
        thk_module.update(cfg, state)
        calved = float(tf.reduce_sum(state.calved_thk))
        # float32 sums: the budget closes to round-off of the total volume.
        assert _volume(state) - before == pytest.approx(
            inflow - calved, abs=1e-6 * before
        )


@pytest.mark.parametrize("method", METHODS)
def test_front_advances_at_the_ice_speed_as_a_sharp_cliff(method):
    cfg, state = _flowline(method)
    u = 1000.0
    _run(cfg, state, 100, u)
    assert _front_1d(state) == pytest.approx(10 * DX + u * 100 * DT, abs=0.01 * DX)
    ice = state.thk.numpy()[1]
    np.testing.assert_allclose(ice[ice > 0.0], 500.0, rtol=1e-5)
    assert float(tf.reduce_sum(state.calved_thk)) == 0.0


@pytest.mark.parametrize("method", METHODS)
def test_front_is_stationary_when_the_ablation_matches_the_ice_speed(method):
    cfg, state = _flowline(method)
    u = 1000.0
    x0 = _front_1d(state)
    _run(cfg, state, 150, u, ablation=u)
    assert _front_1d(state) == pytest.approx(x0, abs=0.05 * DX)
    ice = state.thk.numpy()[1]
    np.testing.assert_allclose(ice[ice > 0.0], 500.0, rtol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_front_retreats_at_the_ablation_minus_the_ice_speed(method):
    cfg, state = _flowline(method, n_ice=40)
    u, retreat = 1000.0, 400.0
    x0 = _front_1d(state)
    steps = 100
    _run(cfg, state, steps, u, ablation=u + retreat)
    assert _front_1d(state) == pytest.approx(x0 - retreat * steps * DT, abs=0.1 * DX)
    ice = state.thk.numpy()[1]
    np.testing.assert_allclose(ice[ice > 0.0], 500.0, rtol=1e-5)
    # Nothing is left unapplied at a front CFL number below 1.
    assert float(tf.reduce_max(state.calving_unapplied_thk)) == 0.0


@pytest.mark.parametrize("method", METHODS)
def test_fixed_front_calves_what_flows_past_it(method):
    cfg, state = _flowline(method, fixed=True)
    x0 = _front_1d(state)
    _run(cfg, state, 60, 1000.0)
    assert _front_1d(state) == pytest.approx(x0, abs=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_min_thickness_calves_thin_floating_front_cells(method):
    thk = np.zeros((3, 20), np.float32)
    thk[:, :8] = np.linspace(600.0, 180.0, 8)
    cfg = _cfg(method, min_thickness=250.0)
    state = _state(thk)
    thk_module.initialize(cfg, state)
    _ice_velocity(state, 0.0)
    thk_module.update(cfg, state)
    row = state.thk.numpy()[1]
    # The thin tail (240 and 180 m) calves in one step; the front is then
    # at least 250 m thick.
    assert row[7] == 0.0 and row[6] == 0.0 and row[5] >= 250.0
    assert float(tf.reduce_sum(state.calved_thk)) == pytest.approx(3 * (240.0 + 180.0))


@pytest.mark.parametrize("method", METHODS)
def test_a_cell_filling_at_the_front_does_not_shield_a_thin_one(method):
    thk = np.zeros((3, 20), np.float32)
    thk[:, :6] = [600.0, 500.0, 400.0, 300.0, 260.0, 240.0]
    cfg = _cfg(method, min_thickness=250.0)
    state = _state(thk)
    thk_module.initialize(cfg, state)
    href = state.Href.numpy()
    href[:, 6] = 239.0  # about to fill (at about 240 m, below the threshold)
    state.Href = tf.constant(href)
    _ice_velocity(state, 1000.0)
    thk_module.update(cfg, state)
    row = state.thk.numpy()[1]
    assert row[5] == 0.0 and row[6] == 0.0 and row[4] >= 250.0


def test_partial_cells_are_invisible_to_the_ice_flow():
    """A floating front is sharp on full Q1 cells; land margins keep partial ones."""
    thk = np.zeros((4, 7), np.float32)
    thk[:, :3] = 400.0  # floating shelf (column 0-2), front at column 2
    topg = np.full((4, 7), -2000.0, np.float32)
    topg[:, 4:] = 100.0  # land from column 4
    thk[1:3, 5:] = 50.0  # a grounded land margin, ice-free land at column 4
    thk = tf.constant(thk)
    topg = tf.constant(topg)
    wl = tf.zeros_like(thk)
    usurf = tf.maximum(topg, wl - RATIO * thk) + thk
    nodes = iceflow_node_mask(thk, usurf, wl, 1.0 / RATIO).numpy()
    grounded = compute_grounded_mask(thk, usurf - thk, wl, 1.0 / RATIO)
    np.testing.assert_array_equal(nodes, compute_node_ice_mask(thk, grounded).numpy())
    assert nodes[:, :3].all() and not nodes[:, 3:5].any()
    # The land-margin nodes belong to fully grounded, partially covered cells.
    assert nodes[1:3, 5:].all()


def test_threshold_thickness_follows_pism():
    thk = tf.constant([[0.0, 300.0, 0.0], [200.0, 0.0, 400.0], [0.0, 0.0, 0.0]])
    wl = tf.zeros_like(thk)
    deep = threshold_thickness(thk, wl - 1000.0, wl, 1.0 / RATIO).numpy()
    assert deep[1, 1] == pytest.approx((300.0 + 200.0 + 400.0) / 3.0)
    # A bed above the neighbours' mean ice base fills to their mean surface.
    topg = tf.constant([[0.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 0.0]])
    high = threshold_thickness(thk, topg, wl, 1.0 / RATIO).numpy()
    assert high[1, 1] == pytest.approx(300.0 - 50.0)  # mean surface 300 minus bed 50


def test_flux_threshold_follows_continuity():
    row = np.array([500.0, 450.0, 400.0, 0.0, 0.0], np.float32)
    thk = tf.constant(np.tile(row, (3, 1)))
    speed = tf.constant(
        np.tile(np.array([300.0, 330.0, 360.0, 0.0, 0.0], np.float32), (3, 1))
    )
    wl = tf.zeros_like(thk)
    topg = wl - 5000.0
    mean = threshold_thickness(thk, topg, wl, 1.0 / RATIO).numpy()
    flux = threshold_thickness(thk, topg, wl, 1.0 / RATIO, speed).numpy()
    assert mean[1, 3] == pytest.approx(400.0)  # PISM: the neighbour
    assert flux[1, 3] == pytest.approx(400.0 * 360.0 / 390.0)  # u H kept
    uniform = threshold_thickness(thk, topg, wl, 1.0 / RATIO, tf.ones_like(thk)).numpy()
    np.testing.assert_allclose(uniform, mean)


@pytest.mark.parametrize("threshold, tolerance", [("flux", 0.15), ("mean", 0.5)])
def test_spreading_shelf_front_follows_weertman(threshold, tolerance):
    """Albrecht et al. (2011) shelf with the exact Weertman velocity prescribed."""
    C, H = WEERTMAN_C, _weertman_thickness
    rho, rho_w = RHO, RHO_W

    def front(t):
        return Q0 / (4 * C) * ((3 * C * t + H0**-3) ** (4 / 3) - H0**-4)

    def front_time(x):
        return (4 * C * x / Q0 + H0**-4) ** 0.75 / (3 * C) - H0**-3 / (3 * C)

    dx = 5000.0
    xc = (np.arange(41) + 0.5) * dx
    thk = np.zeros((3, 41), np.float32)
    thk[:, :2] = H(xc[:2])
    thk[:, 0] = H0
    cfg = _cfg("sub_grid", threshold=threshold)
    cfg.processes.thk.ratio_density = rho / rho_w
    state = _state(thk, dx=dx, dt=3.0)
    thk_module.initialize(cfg, state)
    u_exact = np.tile(Q0 / H(xc), (3, 1)).astype(np.float32)
    for it in range(50):
        state.it = it
        nodes = iceflow_node_mask(
            state.thk, state.usurf, state.water_level, rho_w / rho
        )
        u = np.where(nodes.numpy(), u_exact, 0.0)
        u[:, 0] = U0
        state.ubar = tf.constant(u)
        thk_module.update(cfg, state)
    row = state.thk.numpy()[1]
    last = int(np.max(np.nonzero(row > 0.0)[0]))
    x_front = (last + 1 + state.ice_area_fraction.numpy()[1, last + 1]) * dx
    expected = front(150.0 + front_time(2 * dx))
    assert abs(x_front - expected) < tolerance * dx
    # The front ice is always the same parcel, so the flux imbalances of the
    # front cells accumulate in it: with the neighbour mean (PISM) it is about
    # 20 % too thick here, with the flux threshold 2-3 % (at any resolution).
    if threshold == "flux":
        np.testing.assert_allclose(
            row[last - 2 : last + 1], H(xc[last - 2 : last + 1]), rtol=0.03
        )


@pytest.mark.parametrize("method", METHODS)
def test_front_held_by_the_thickness_rule_then_retreats_at_the_prescribed_rate(
    method,
):
    """Weertman shelf held by the 250-m rule, then driven back at 300 m/yr.

    With no calving rate the level set moves with the ice while the rule
    removes the ice behind it; ``psi`` must stay tied to the ice, or a later
    retreat starts late (the ``psi`` of the ice drifts by ``-u t``).
    """
    dx, dt = 10000.0, 7.0
    x = np.arange(30) * dx
    thk = np.zeros((3, x.size), np.float32)
    thk[:, :2] = _weertman_thickness(x[:2])
    cfg = _cfg(method, min_thickness=250.0)
    cfg.processes.thk.ratio_density = RHO / RHO_W
    state = _state(thk, dx=dx, dt=dt)
    thk_module.initialize(cfg, state)
    u_exact = np.tile(Q0 / _weertman_thickness(x), (3, 1)).astype(np.float32)

    def step():
        nodes = iceflow_node_mask(
            state.thk, state.usurf, state.water_level, RHO_W / RHO
        )
        state.ubar = tf.constant(np.where(nodes.numpy(), u_exact, 0.0))
        thk_module.update(cfg, state)

    for _ in range(45):  # 315 yr: the front stops near x(H = 250 m) = 144 km
        step()
    x0 = _front_1d(state, dx)
    assert x0 == pytest.approx(144.45e3, abs=dx)
    if method == "level_set":  # the zero of psi stays at the ice front
        row = state.thk.numpy()[1]
        last = int(np.max(np.nonzero(row > 0.0)[0]))
        psi = state.psi.numpy()[1]
        zero = (last + 0.5 - psi[last] / (psi[last + 1] - psi[last])) * dx
        assert zero == pytest.approx(x0, abs=0.5 * dx)
    state.thk_components.component_state["front"]["min_thickness"] = 0.0
    state.calving_rate = tf.constant(u_exact + 300.0)
    for _ in range(15):  # 105 yr
        step()
    assert _front_1d(state, dx) == pytest.approx(x0 - 300.0 * 105.0, abs=0.3 * dx)


def test_sub_grid_residual_redistribution_conserves_mass():
    thk = tf.constant(np.pad(np.full((3, 3), 300.0, np.float32), ((1, 1), (1, 3))))
    href = tf.zeros_like(thk).numpy()
    href[2, 4] = 800.0  # far more than one cell: fills and spills
    href = tf.constant(href)
    topg = tf.fill(tf.shape(thk), -2000.0)
    wl = tf.zeros_like(thk)
    out = sub_grid.front_step(
        thk,
        href,
        topg,
        wl,
        tf.zeros_like(thk),
        tf.constant(0.1),
        tf.constant(DX),
        None,
        1.0 / RATIO,
        10,
        True,
        0.0,
    )
    new_thk, new_href, calved = out[0], out[1], out[2]
    assert float(tf.reduce_sum(new_thk + new_href)) == pytest.approx(
        float(tf.reduce_sum(thk + href)), rel=1e-6
    )
    assert float(tf.reduce_sum(calved)) == 0.0
    assert new_thk.numpy()[2, 4] == pytest.approx(300.0)


def test_sub_grid_discarded_residual_is_counted_as_calved():
    thk = tf.constant(np.pad(np.full((3, 3), 300.0, np.float32), ((1, 1), (1, 3))))
    href = np.zeros(thk.shape, np.float32)
    href[2, 4] = 450.0
    out = sub_grid.front_step(
        thk,
        tf.constant(href),
        tf.fill(tf.shape(thk), -2000.0),
        tf.zeros_like(thk),
        tf.zeros_like(thk),
        tf.constant(0.1),
        tf.constant(DX),
        None,
        1.0 / RATIO,
        10,
        False,
        0.0,
    )
    assert float(tf.reduce_sum(out[2])) == pytest.approx(150.0)


@pytest.mark.parametrize("method", METHODS)
def test_reservoir_without_ice_next_to_it_is_calved(method):
    thk = np.zeros((3, 12), np.float32)
    thk[:, :4] = 500.0
    cfg = _cfg(method)
    state = _state(thk)
    thk_module.initialize(cfg, state)
    href = state.Href.numpy()
    href[1, 9] = 100.0  # stranded, far from the ice
    state.Href = tf.constant(href)
    _ice_velocity(state, 0.0)
    thk_module.update(cfg, state)
    assert float(state.Href[1, 9]) == 0.0
    assert float(state.calved_thk[1, 9]) == pytest.approx(100.0)


def test_level_set_reinitialisation_keeps_the_fill_fractions():
    x = (np.arange(40) - 20.3) * DX
    psi = tf.constant(np.tile(0.5 * x + 3.0 * np.sin(x / 7e3) * DX, (5, 1)), tf.float32)
    new = reinitialise(psi, tf.constant(DX), 30)
    np.testing.assert_array_equal(
        fill_fraction(psi, tf.constant(DX)).numpy(),
        fill_fraction(new, tf.constant(DX)).numpy(),
    )
    # Away from the front it is a distance again.
    gradient = np.diff(new.numpy()[2, 26:34]) / DX
    np.testing.assert_allclose(gradient, 1.0, atol=0.05)


def test_old_front_keys_fail_loudly():
    cfg = _cfg("sub_grid")
    cfg.processes.thk.calving_front = True
    with pytest.raises(ValueError, match="moved to cfg.processes.thk.front"):
        thk_module.initialize(cfg, _state(np.ones((3, 4))))


# ---------------------------------------------------------------------------
# Circular 2-D shelf
# ---------------------------------------------------------------------------


def _circular(method, radius=20.0, n=61, H=500.0, U=1000.0):
    """Radial outflow u = U r/|r| from the centre; a source keeps H uniform."""
    c = (n - 1) / 2
    y, x = np.mgrid[0:n, 0:n] - c
    r = np.hypot(x, y) * DX
    thk = np.where(r <= radius * DX, H, 0.0).astype(np.float32)
    cfg = _cfg(method, boundary={s: "zero" for s in ("left", "right", "top", "bottom")})
    state = _state(thk)
    rr = np.maximum(r, 0.5 * DX)
    state.smb = tf.constant((H * U / rr).astype(np.float32))  # div(H U r_hat) = H U / r
    thk_module.initialize(cfg, state)
    ux = (U * x * DX / rr).astype(np.float32)
    uy = (U * y * DX / rr).astype(np.float32)
    return cfg, state, ux, uy


def _area_radius(state):
    return float(np.sqrt(np.sum(state.ice_area_fraction.numpy()) * DX * DX / np.pi))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("advance", [0.0, 300.0])
def test_circular_shelf_front_is_stationary_or_moves_at_the_prescribed_speed(
    method, advance
):
    U = 1000.0
    cfg, state, ux, uy = _circular(method, U=U)
    r0 = _area_radius(state)
    steps = 60
    for it in range(steps):
        state.it = it
        nodes = iceflow_node_mask(
            state.thk, state.usurf, state.water_level, 1.0 / RATIO
        )
        state.ubar = tf.where(nodes, ux, 0.0)
        state.vbar = tf.where(nodes, uy, 0.0)
        state.calving_rate = tf.fill(tf.shape(state.thk), U - advance)
        # The 2-D mass budget closes every step (as _run asserts in 1-D):
        # the source enters full cells, and partial cells over their fraction.
        ice = state.thk > 0.0
        partial = ~ice & any_neighbour(ice) & (state.Href > 0.0)
        smb, frac = state.smb, state.ice_area_fraction
        source = DT * (
            float(tf.reduce_sum(tf.where(ice, smb, 0.0 * smb)))
            + float(tf.reduce_sum(tf.where(partial, smb * frac, 0.0 * smb)))
        )
        before = _volume(state)
        thk_module.update(cfg, state)
        calved = float(tf.reduce_sum(state.calved_thk))
        assert _volume(state) - before == pytest.approx(
            source - calved, abs=1e-6 * before
        )
    expected = r0 + advance * steps * DT
    # The first-order level set drifts about +0.5 dx over this sustained 3.6 km
    # advance (see DESIGN.md); before the honest initial fill fraction the same
    # drift was hidden in an r0 inflated by a spurious ring of fraction 0.15.
    tolerance = 0.75 * DX if (advance and method == "level_set") else 0.5 * DX
    assert _area_radius(state) == pytest.approx(expected, abs=tolerance)
    # The front stays circular and exactly symmetric.
    fraction = state.ice_area_fraction.numpy()
    np.testing.assert_allclose(fraction, fraction.T, atol=1e-4)
    np.testing.assert_allclose(fraction, fraction[::-1, :], atol=1e-4)
    np.testing.assert_allclose(fraction, fraction[:, ::-1], atol=1e-4)
    # Along the axis and the diagonal the front radius agrees to within a cell.
    n = fraction.shape[0]
    c = (n - 1) // 2
    axis = np.sum(fraction[c, c:]) * DX
    diagonal = np.sum(np.diagonal(fraction)[c:]) * DX * np.sqrt(2.0)
    assert abs(axis - diagonal) < 1.5 * DX


# ---------------------------------------------------------------------------
# Regressions of the 2026-09 review
# ---------------------------------------------------------------------------


def test_level_set_slow_front_escapes_a_clamped_empty_cell():
    """A slow front (u dt below the clamp margin) still enters the cell ahead.

    A wipe or a rule leaves the empty cell ahead of the front clamped just
    outside half its width. The constrain used to push the advected psi back
    out to that margin every step, so a front with ``u dt`` below the margin
    (a slow front in a domain whose time step is set by faster ice elsewhere)
    never reached the cell, and its routed inflow was wiped as spurious
    calving every step, for ever. The advection must persist across steps:
    losses stop within a couple of steps and the front advances at ``u``.
    """
    cfg, state = _flowline("level_set")
    u, steps = 10.0, 150  # u dt = 2 m per step, against a clamp margin of 5 m
    psi = state.psi.numpy()
    psi[:, 10] = 0.505 * DX  # as the old outward clamp left it
    state.psi = tf.constant(psi)
    inflow_step = 3 * u * 500.0 * DT / DX  # three rows of Dirichlet-fed inflow
    total_calved = 0.0
    for it in range(steps):
        state.it = it
        _ice_velocity(state, u)
        state.calving_rate = tf.zeros_like(state.thk)
        thk_module.update(cfg, state)
        total_calved += float(tf.reduce_sum(state.calved_thk))
    # Only the steps before psi first crosses half the width lose their inflow.
    assert total_calved <= 3 * inflow_step
    assert _front_1d(state) >= 10 * DX + u * steps * DT - 3 * u * DT


@pytest.mark.parametrize("method", METHODS)
def test_free_slab_trailing_edge_moves_with_the_ice_without_calving(method):
    """Ice advected away from an ice-free cell is thinned only by the transport.

    The level set demoted such trailing cells by the fill fraction a second
    time and later wiped them as calved; a free slab must advect at the ice
    speed with nothing calved and its mass centroid at ``u t``.
    """
    thk = np.zeros((3, 60), np.float32)
    thk[:, 10:20] = 500.0
    boundary = {s: "zero" for s in ("left", "right")} | {
        s: "symmetric" for s in ("top", "bottom")
    }
    cfg = _cfg(method, boundary=boundary)
    state = _state(thk)
    thk_module.initialize(cfg, state)
    u, steps = 400.0, 25
    x = np.arange(60) * DX

    def centroid() -> float:
        mass = (state.thk + state.Href).numpy()[1]
        return float(np.sum(x * mass) / np.sum(mass))

    c0, v0 = centroid(), _volume(state)
    for it in range(steps):
        state.it = it
        _ice_velocity(state, u)
        state.calving_rate = tf.zeros_like(state.thk)
        thk_module.update(cfg, state)
        assert float(tf.reduce_sum(state.calved_thk)) == 0.0
    assert _volume(state) == pytest.approx(v0, rel=1e-6)
    assert centroid() - c0 == pytest.approx(u * steps * DT, abs=0.1 * DX)
    assert _front_1d(state) == pytest.approx(20 * DX + u * steps * DT, abs=0.1 * DX)


def test_level_set_initial_fraction_is_sharp_on_an_oblique_front():
    """A fresh psi is synchronised to the ice before anything runs.

    On a 45-degree front the raw ramp (built with width dx) put full cells
    inside by less than half their true width: the first step demoted them
    (about 15 % of their thickness calved), and the published fraction showed
    a spurious ring of about 0.15 around the ice.
    """
    n = 30
    y, x = np.mgrid[0:n, 0:n]
    thk = np.where(x + y <= 20, 500.0, 0.0).astype(np.float32)
    boundary = {s: "zero" for s in ("left", "right", "top", "bottom")}
    cfg = _cfg("level_set", boundary=boundary)
    state = _state(thk)
    thk_module.initialize(cfg, state)
    np.testing.assert_array_equal(
        state.ice_area_fraction.numpy(), (thk > 0.0).astype(np.float32)
    )
    _ice_velocity(state, 0.0)
    state.it = 0
    thk_module.update(cfg, state)
    assert float(tf.reduce_sum(state.calved_thk)) == 0.0
    np.testing.assert_array_equal(state.thk.numpy(), thk)
