#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Sub-shelf melt laws against closed forms and NumPy ports of Kori-ULB."""

from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.linalg
import tensorflow as tf

from igm.processes.bmb.geometry import compute_geometry, shelf_distances, shelf_labels
from igm.processes.bmb.geometry import grounding_line_depth
from igm.processes.bmb.laws.pico import pico
from igm.processes.bmb.laws.picop import picop
from igm.processes.bmb.laws.plume import plume
from igm.processes.bmb.laws.prescribed import prescribed
from igm.processes.bmb.laws.quadratic import quadratic
from igm.processes.bmb.utils import SECONDS_PER_YEAR

from conftest import RHO_I, RHO_W

pytestmark = [pytest.mark.fast, pytest.mark.unit]

LAMBDA = (-0.0573, 0.0832, 7.61e-4)  # ocean.freezing_point defaults
NU_LAMBDA = RHO_I / RHO_W * 3.34e5 / 3974.0


def _freezing(salinity, z):
    return LAMBDA[0] * salinity + LAMBDA[1] + LAMBDA[2] * z


def _pico_freezing(salinity, z):
    """Potential-temperature freezing point of PICO (Reese et al., 2018)."""
    return -0.0572 * salinity + 0.0788 + 7.77e-8 * RHO_W * 9.81 * z


def _setup(cfg_factory, state_factory, channel_factory, front=32, **bmb_cfg):
    cfg = cfg_factory(**bmb_cfg)
    thk, topg = channel_factory(front=front)
    state = state_factory(cfg, thk, topg, ubar=300.0)
    return cfg, state, compute_geometry(cfg, state)


# --- prescribed and quadratic ----------------------------------------------


def test_mismip_plus_melt(cfg_factory, state_factory, channel_factory):
    cfg, state, geom = _setup(
        cfg_factory, state_factory, channel_factory, prescribed={"form": "mismip_plus"}
    )
    melt = prescribed.melt_rate(cfg, state, geom).numpy()
    draft = geom.draft.numpy()
    column = draft - state.topg.numpy()
    expected = 0.2 * np.tanh(column / 75.0) * np.maximum(-100.0 - draft, 0.0)
    shelf = geom.shelf.numpy()
    np.testing.assert_allclose(melt[shelf], expected[shelf], rtol=1e-5)
    assert melt[shelf].max() > 10.0


def test_quadratic_melt_and_its_averages(cfg_factory, state_factory, channel_factory):
    cfg, state, geom = _setup(cfg_factory, state_factory, channel_factory)
    shelf = geom.shelf.numpy()
    rows = np.arange(shelf.shape[0])[:, None] * np.ones_like(shelf)
    cols = np.arange(shelf.shape[1])[None, :] * np.ones_like(shelf)
    thermal_forcing = np.where(cols < 26, 1.0, 3.0).astype(np.float32)
    state.ocean_thermal_forcing = tf.constant(thermal_forcing)
    state.basins = tf.constant(np.where(rows < 6, 1.0, 2.0).astype(np.float32))

    factor = 1.4477e4 * (RHO_W * 3974.0 / (RHO_I * 3.34e5)) ** 2 * 1000.0 / RHO_I
    q = cfg.processes.bmb.quadratic
    q.averaging = "local"
    local = quadratic.melt_rate(cfg, state, geom).numpy()
    np.testing.assert_allclose(
        local[shelf], factor * thermal_forcing[shelf] ** 2, rtol=1e-5
    )

    q.averaging = "shelf"
    mean = thermal_forcing[shelf].mean()
    melt = quadratic.melt_rate(cfg, state, geom).numpy()
    np.testing.assert_allclose(
        melt[shelf], factor * thermal_forcing[shelf] * mean, rtol=1e-5
    )

    # Two basins cut the shelf along y: each has the same mean here, then not.
    thermal_forcing[rows >= 6] += 1.0
    state.ocean_thermal_forcing = tf.constant(thermal_forcing)
    q.averaging = "basin"
    melt = quadratic.melt_rate(cfg, state, geom).numpy()
    for basin in (rows < 6, rows >= 6):
        in_basin = shelf & basin
        expected = factor * thermal_forcing[in_basin] * thermal_forcing[in_basin].mean()
        np.testing.assert_allclose(melt[in_basin], expected, rtol=1e-5)


# --- PICO ------------------------------------------------------------------


def test_pico_box_index_matches_the_box_boundaries():
    d, f = np.meshgrid(np.arange(1, 61), np.arange(1, 61), indexing="ij")
    labels = tf.ones(d.shape, tf.int32)
    for n in range(1, 11):
        box = pico.box_numbers(
            n, 60.0, labels, tf.constant(d, tf.int32), tf.constant(f, tf.int32)
        ).numpy()
        for i, j in [(0, 0), (4, 9), (29, 29), (59, 0), (0, 59), (17, 42)]:
            r = Fraction(int(d[i, j]), int(d[i, j] + f[i, j]))
            k = int(box[i, j])
            # 1 - sqrt(1-(k-1)/n) <= r < 1 - sqrt(1-k/n), unless capped at d_GL.
            if k < int(d[i, j]):
                assert Fraction(k - 1, n) <= 1 - (1 - r) ** 2 < Fraction(k, n)
        exact = 1 + (n * d * (d + 2 * f)) // (d + f) ** 2
        np.testing.assert_array_equal(box, np.minimum(np.minimum(exact, n), d))


def test_pico_boxes_conserve_heat_and_salt(cfg_factory, state_factory, channel_factory):
    cfg, state, geom = _setup(
        cfg_factory, state_factory, channel_factory, method="pico"
    )
    p = cfg.processes.bmb.pico
    boxes = pico.solve_boxes(cfg, state, geom)
    box, temp, salinity = (v.numpy() for v in (boxes.box, boxes.temp, boxes.salinity))
    melt, z = boxes.melt.numpy(), geom.draft.numpy()
    area = lambda k: (box == k).sum() * 1000.0**2
    assert set(np.unique(box[geom.shelf.numpy()])) == {1, 2, 3, 4, 5}

    # Open water in front of the shelf reaches -720 m: T_0 and S_0 of WARM.
    t0, s0 = 1.0, 34.7
    in1 = box == 1
    x = t0 - temp[in1]
    s = s0 / NU_LAMBDA
    q = p.overturning * p.rho_star * (p.beta * s - p.alpha) * x
    heat = area(1) * p.gamma_T * (temp[in1] - _pico_freezing(s0, z[in1]))
    np.testing.assert_allclose(q * x, heat, rtol=1e-3)
    np.testing.assert_allclose(s0 - salinity[in1], s * x, rtol=1e-3)

    q_mean = q.mean()
    for k in range(2, 6):
        ink, before = box == k, box == k - 1
        t_in, s_in = temp[before].mean(), salinity[before].mean()
        heat = area(k) * p.gamma_T * (temp[ink] - _pico_freezing(salinity[ink], z[ink]))
        np.testing.assert_allclose(q_mean * (t_in - temp[ink]), heat, rtol=2e-3)
        np.testing.assert_allclose(
            salinity[ink], s_in * (1.0 - (t_in - temp[ink]) / NU_LAMBDA), rtol=1e-6
        )

    exchange = p.gamma_T / NU_LAMBDA * SECONDS_PER_YEAR
    shelf = geom.shelf.numpy()
    np.testing.assert_allclose(
        melt[shelf], exchange * (temp - _pico_freezing(salinity, z))[shelf], rtol=1e-4
    )
    assert melt[box == 1].mean() > melt[box == 5].mean() > 0.0


def test_pico_reference_distance(cfg_factory, state_factory, channel_factory):
    """The domain's largest distance, given explicitly, changes nothing."""
    cfg, state, geom = _setup(cfg_factory, state_factory, channel_factory)
    default = pico.solve_boxes(cfg, state, geom).box.numpy()
    d_gl, _ = shelf_distances(geom, shelf_labels(geom))
    cfg.processes.bmb.pico.reference_distance = float(d_gl.numpy().max()) * 1000.0
    np.testing.assert_array_equal(
        pico.solve_boxes(cfg, state, geom).box.numpy(), default
    )
    cfg.processes.bmb.pico.reference_distance = 1.0e7  # a much larger shelf
    assert pico.solve_boxes(cfg, state, geom).box.numpy().max() < default.max()


def test_pico_cold_cavity_stays_finite(cfg_factory, state_factory, channel_factory):
    cfg = cfg_factory(method="pico")
    cfg.processes.ocean.profile.array = [["z", "temp", "salinity"], [0.0, -3.0, 34.5]]
    thk, topg = channel_factory(front=32)
    state = state_factory(cfg, thk, topg)
    boxes = pico.solve_boxes(cfg, state, compute_geometry(cfg, state))
    melt = boxes.melt.numpy()
    assert np.isfinite(melt).all()
    # Water entering at the freezing point of the deep grounding line is
    # supercooled once it rises in the outer boxes (the ice pump).
    assert np.abs(melt).max() < 0.1
    assert melt[boxes.box.numpy() == 5].mean() < 0.0


# --- grounding-line depth ---------------------------------------------------


def test_grounding_line_depth_is_carried_along_a_uniform_flow(
    cfg_factory, state_factory, channel_factory
):
    cfg, state, geom = _setup(cfg_factory, state_factory, channel_factory)
    z_gl = grounding_line_depth(cfg, state, geom).numpy()
    shelf = geom.shelf.numpy()
    first = np.argmax(shelf, axis=1)
    source = state.topg.numpy()[np.arange(shelf.shape[0]), first - 1]
    for row in range(shelf.shape[0]):
        np.testing.assert_allclose(z_gl[row, shelf[row]], source[row], atol=1e-3)


def test_grounding_line_depth_solves_the_upwind_system(
    cfg_factory, state_factory, channel_factory
):
    cfg, state, geom = _setup(
        cfg_factory,
        state_factory,
        channel_factory,
        grounding_line_depth={"epsilon": 1.0e4, "tol": 1.0e-5, "max_iter": 200000},
    )
    rng = np.random.default_rng(0)
    shape = state.thk.shape
    state.ubar = tf.constant(rng.normal(200.0, 150.0, shape), tf.float32)
    state.vbar = tf.constant(rng.normal(0.0, 150.0, shape), tf.float32)
    z_gl = grounding_line_depth(cfg, state, geom).numpy()

    # Reference: the same first-order upwind system, solved directly.
    shelf = geom.shelf.numpy()
    fixed = np.where(geom.grounded.numpy(), np.minimum(geom.draft.numpy(), 0.0), 0.0)
    u, v, dx, e = state.ubar.numpy(), state.vbar.numpy(), 1000.0, 1.0e4 / 1000.0**2
    index = -np.ones(shape, int)
    index[shelf] = np.arange(shelf.sum())
    rows, cols, vals, rhs = [], [], [], np.zeros(shelf.sum())
    for i, j in zip(*np.nonzero(shelf)):
        weights = {
            (i - 1, j): max(v[i, j], 0) / dx + e,
            (i + 1, j): max(-v[i, j], 0) / dx + e,
            (i, j - 1): max(u[i, j], 0) / dx + e,
            (i, j + 1): max(-u[i, j], 0) / dx + e,
        }
        n = index[i, j]
        for (a, b), w in weights.items():
            if not (0 <= a < shape[0] and 0 <= b < shape[1]):
                continue
            rows.append(n), cols.append(n), vals.append(w)
            if shelf[a, b]:
                rows.append(n), cols.append(index[a, b]), vals.append(-w)
            else:
                rhs[n] += w * fixed[a, b]
    matrix = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(len(rhs),) * 2)
    reference = np.zeros(shape)
    reference[shelf] = scipy.sparse.linalg.spsolve(matrix, rhs)
    draft = geom.draft.numpy()
    reference = np.where(reference < 0.0, np.minimum(reference, draft), 0.0)
    np.testing.assert_allclose(z_gl[shelf], reference[shelf], atol=0.05)


# --- plume laws against Kori-ULB ------------------------------------------


def _kori_picop(ta, sa, hb, zgl, sina):
    """Port of Kori-ULB PICOPmelt.m (without its box lookup), with the
    temperature floor of Pelle et al. (2019) at sea level instead of Kori's
    at the draft."""
    ta = np.maximum(ta, _freezing(sa, 0.0))
    tfgl = _freezing(sa, zgl)
    e = 3.6e-2 * sina
    gamma = 1.1e-3 * (0.545 + 3.5e-5 * (ta - tfgl) / LAMBDA[2] * e / (6e-4 + e))
    galfa = (
        np.sqrt(sina / (2.5e-3 + e)) * np.sqrt(gamma / (gamma + e)) * e / (gamma + e)
    )
    length = (ta - tfgl) / LAMBDA[2] * (0.56 * gamma + e) / (0.56 * (gamma + e))
    xhat = np.minimum((hb - zgl) / length, 1.0)
    poly = sum(c * xhat**i for i, c in enumerate(picop.POLYNOMIAL))
    return 10.0 * poly * galfa * (ta - tfgl) ** 2


def _kori_plume(t, s, hb, zgl, sina):
    """Port of Kori-ULB PlumeMelt2019.m and its helpers."""
    gamma, e0, lat, cp = 5.9e-4, 3.6e-2, 3.34e5, 3974.0
    alpha, beta = 3.87e-5, 7.86e-4
    c1 = lat * alpha / (cp * gamma * beta * s)
    ctau = (-LAMBDA[0] * alpha / beta) / c1
    tf_ = _freezing(s, zgl)
    e = e0 * sina
    x = (
        LAMBDA[2]
        * (hb - zgl)
        / ((t - tf_) * (1 + 0.6 * (e / (gamma + ctau + e)) ** 0.75))
    )
    x = np.clip(x, 0.0, 1.0)
    x[zgl >= 0] = 0.0
    mhat = (
        (3 * (1 - x) ** (4 / 3) - 1)
        * np.sqrt(1 - (1 - x) ** (4 / 3))
        / (2 * np.sqrt(2))
    )
    mterm = (
        np.sqrt(beta * s * 9.81 / (LAMBDA[2] * (lat / cp) ** 3))
        * np.sqrt(np.maximum(0, (1 - c1 * gamma) / (2.5e-3 + e)))
        * (gamma * e / (gamma + ctau + e)) ** 1.5
        * (t - tf_) ** 2
    )
    return mterm * mhat * SECONDS_PER_YEAR


@pytest.mark.parametrize("law, kori", [(picop, _kori_picop), (plume, _kori_plume)])
def test_plume_melts_match_kori_up_to_the_ice_conversion(cfg_factory, law, kori):
    cfg = cfg_factory()
    rng = np.random.default_rng(1)
    n = 200
    zgl = rng.uniform(-1200.0, -300.0, n)
    hb = zgl + rng.uniform(1.0, 600.0, n)
    temp = rng.uniform(-1.0, 2.0, n)
    salinity = rng.uniform(34.0, 35.0, n)
    sina = rng.uniform(1e-4, 5e-2, n)
    driving = np.maximum(temp, _freezing(salinity, 0.0)) - _freezing(salinity, zgl)
    keep = driving > 0.05
    args = [a[keep] for a in (temp, salinity, hb, zgl, sina)]

    melt = law.plume_melt(cfg, *(tf.constant(a, tf.float64) for a in args)).numpy()
    np.testing.assert_allclose(melt, kori(*args) * RHO_W / RHO_I, rtol=1e-6, atol=1e-9)


def test_plume_guards_and_shapes(cfg_factory):
    cfg = cfg_factory()
    np.testing.assert_allclose(
        picop.dimensionless_melt(tf.constant(0.0)).numpy(),
        picop.POLYNOMIAL[0],
        atol=1e-5,
    )
    # A source at or above sea level, or a flat base: no melt.
    for law in (picop, plume):
        melt = law.plume_melt(
            cfg,
            tf.constant([1.0, 1.0]),
            tf.constant([34.5, 34.5]),
            tf.constant([-300.0, -300.0]),
            tf.constant([0.0, -600.0]),
            tf.constant([0.01, 0.0]),
        ).numpy()
        np.testing.assert_array_equal(melt, 0.0)
    # Water colder than the freezing point at the source drives no plume; in
    # PICOP, the ambient temperature is never below the sea-level freezing
    # point (Pelle et al., 2019), so the plume is still driven.
    args = [tf.constant([v]) for v in (-3.0, 34.5, -600.0, -600.0, 0.01)]
    assert plume.plume_melt(cfg, *args).numpy()[0] == 0.0
    assert picop.plume_melt(cfg, *args).numpy()[0] > 0.0


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
def test_grounding_line_depth_follows_the_flow_in_every_direction(
    direction, cfg_factory, state_factory, channel_factory
):
    """The shelf takes the bed depth of the last grounded node upstream,
    lowered to the local ice draft where that is deeper (Pelle et al., 2019)."""
    thk, topg = channel_factory(ny=12, nx=30)
    along_x = direction in ("+x", "-x")
    orient = {
        "+x": lambda a: a,
        "-x": lambda a: a[:, ::-1],
        "+y": lambda a: a.T,
        "-y": lambda a: a.T[::-1, :],
    }[direction]
    thk, topg = orient(thk).copy(), orient(topg).copy()
    speed = 300.0
    u = {"+x": speed, "-x": -speed, "+y": 0.0, "-y": 0.0}[direction]
    v = {"+x": 0.0, "-x": 0.0, "+y": speed, "-y": -speed}[direction]
    cfg = cfg_factory()
    for side in ("left", "right", "top", "bottom"):
        cfg.processes.thk.boundary[side] = "zero"
    state = state_factory(cfg, thk, topg, ubar=u)
    state.vbar = tf.fill(thk.shape, np.float32(v))
    geom = compute_geometry(cfg, state)
    z_gl = grounding_line_depth(cfg, state, geom).numpy()

    lines = lambda a: a if along_x else a.T
    shelf = lines(geom.shelf.numpy())
    grounded = lines(geom.grounded.numpy() & (thk > 0))
    assert shelf.any()
    for line, bed, ground, draft, value in zip(
        shelf, lines(topg), grounded, lines(geom.draft.numpy()), lines(z_gl)
    ):
        upstream = bed[ground].min()  # the bed deepens downstream
        expected = np.minimum(min(upstream, 0.0), draft)
        np.testing.assert_allclose(value[line], expected[line], atol=1e-3)


def test_picop_needs_a_source_below_sea_level(
    cfg_factory, state_factory, channel_factory
):
    """A shelf fed from ground above sea level has no plume source."""
    cfg = cfg_factory(method="picop")
    thk, topg = channel_factory(front=32)
    first = int(np.argmax(thk[0] < -(1028.0 / 918.0) * topg[0]))
    topg[:, :first] = 50.0  # grounded on land up to the grounding line
    state = state_factory(cfg, thk, topg, ubar=300.0)
    geom = compute_geometry(cfg, state)
    melt = picop.melt_rate(cfg, state, geom).numpy()
    shelf = geom.shelf.numpy()
    assert shelf.any()
    np.testing.assert_array_equal(state.grounding_line_depth.numpy()[shelf], 0.0)
    np.testing.assert_array_equal(melt[shelf], 0.0)


def test_pico_ice_rises_are_not_grounding_line(
    cfg_factory, state_factory, channel_factory
):
    cfg = cfg_factory(method="pico")
    thk, topg = channel_factory(front=36)
    topg[5:7, 26:28] = -150.0  # a pinning point in the shelf
    state = state_factory(cfg, thk, topg)
    geom = compute_geometry(cfg, state)
    rise = (geom.grounded & geom.ice).numpy()
    rise[:, :20] = False
    assert rise[5:7, 26:28].all() and rise.sum() == 4

    boxes = pico.solve_boxes(cfg, state, geom).box.numpy()
    assert (boxes[4, 26:28] == 1).all()  # a box-1 ring around the rise
    cfg.processes.bmb.pico.maximum_ice_rise_area = 10.0  # km2, more than 4 nodes
    boxes = pico.solve_boxes(cfg, state, geom).box.numpy()
    assert (boxes[4, 26:28] > 1).all()
    np.testing.assert_array_equal(boxes[rise], 0)
