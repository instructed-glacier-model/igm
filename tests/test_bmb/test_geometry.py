#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from collections import deque

import numpy as np
import pytest

from igm.processes.bmb.geometry import compute_geometry, shelf_distances, shelf_labels
from igm.processes.thk.masks import WATER_LEVEL_NO_OCEAN, compute_grounded_mask

pytestmark = [pytest.mark.fast, pytest.mark.unit]


def _bfs(seed, domain):
    dist = np.zeros(domain.shape, int)
    queue = deque(zip(*np.nonzero(seed)))
    dist[seed] = 1
    while queue:
        i, j = queue.popleft()
        for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= a < domain.shape[0] and 0 <= b < domain.shape[1]:
                if domain[a, b] and not dist[a, b]:
                    dist[a, b] = dist[i, j] + 1
                    queue.append((a, b))
    return dist


def test_shelf_is_the_floating_ice_of_the_thickness_masks(
    cfg_factory, state_factory, channel_factory
):
    cfg = cfg_factory()
    thk, topg = channel_factory(front=32)
    state = state_factory(cfg, thk, topg)
    geom = compute_geometry(cfg, state)
    floating = (thk > 0) & ~compute_grounded_mask(
        state.thk, state.topg, state.water_level, 1028.0 / 918.0
    ).numpy()
    assert floating.any()
    np.testing.assert_array_equal(geom.shelf.numpy(), floating)
    np.testing.assert_array_equal(geom.open_ocean.numpy(), thk == 0)
    # The ice base under ice, the sea floor in open water.
    base = np.where(thk > 0, state.lsurf.numpy(), topg)
    np.testing.assert_allclose(geom.draft.numpy(), np.minimum(base, 0.0))


def test_floating_ice_over_a_subglacial_lake_is_not_exposed(
    cfg_factory, state_factory, channel_factory
):
    thk, topg = channel_factory(front=32)
    topg[4:7, 3:6] = -2000.0  # a deep pocket under thick grounded ice
    cfg = cfg_factory()
    state = state_factory(cfg, thk, topg)
    lake = (thk > 0) & ~compute_grounded_mask(
        state.thk, state.topg, state.water_level, 1028.0 / 918.0
    ).numpy()
    lake[:, 10:] = False
    assert lake.sum() == 9

    geom = compute_geometry(cfg, state)
    assert not geom.shelf.numpy()[lake].any()
    assert geom.shelf.numpy()[:, 20:32].all()

    cfg = cfg_factory(ocean_connected_only=False)
    assert compute_geometry(cfg, state).shelf.numpy()[lake].all()


def test_a_shelf_ending_on_an_open_side_is_exposed(
    cfg_factory, state_factory, channel_factory
):
    thk, topg = channel_factory()  # no ice-free ocean at all
    cfg = cfg_factory()
    state = state_factory(cfg, thk, topg)
    assert compute_geometry(cfg, state).shelf.numpy()[:, 20:].all()

    cfg.processes.thk.boundary.right = "symmetric"
    assert not compute_geometry(cfg, state).shelf.numpy().any()


def test_no_ocean_means_no_shelf(cfg_factory, state_factory, channel_factory):
    thk, topg = channel_factory(front=32)
    cfg = cfg_factory()
    state = state_factory(cfg, thk, topg, water_level=WATER_LEVEL_NO_OCEAN)
    geom = compute_geometry(cfg, state)
    assert not geom.shelf.numpy().any()
    assert geom.grounded.numpy().all()


def test_shelf_distances_follow_pico_seeding(
    cfg_factory, state_factory, channel_factory
):
    cfg = cfg_factory()
    thk, topg = channel_factory(front=32)
    thk[:3, 26:32] = 0.0  # a notch of open water in the front
    state = state_factory(cfg, thk, topg)
    geom = compute_geometry(cfg, state)
    labels = shelf_labels(geom)
    d_gl, d_cf = (d.numpy() for d in shelf_distances(geom, labels))

    shelf = geom.shelf.numpy()
    grounded_ice = geom.grounded.numpy() & (thk > 0)
    padded = np.pad(grounded_ice, 1)
    near_gl = np.zeros_like(shelf)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            near_gl |= padded[
                1 + di : 1 + di + shelf.shape[0], 1 + dj : 1 + dj + shelf.shape[1]
            ]
    np.testing.assert_array_equal(d_gl, _bfs(shelf & near_gl, shelf))

    water = np.pad(thk == 0, 1)
    near_front = water[:-2, 1:-1] | water[2:, 1:-1] | water[1:-1, :-2] | water[1:-1, 2:]
    np.testing.assert_array_equal(d_cf, _bfs(shelf & near_front, shelf))
    assert len(np.unique(labels.numpy()[shelf])) == 1

    # Without ice-free ocean the front starts on the open right side.
    thk, topg = channel_factory()
    state = state_factory(cfg, thk, topg)
    geom = compute_geometry(cfg, state)
    _, d_cf = shelf_distances(geom, shelf_labels(geom))
    np.testing.assert_array_equal(d_cf.numpy()[:, -1], 1)


def test_periodic_sides_do_not_expose_to_the_ocean(
    cfg_factory, state_factory, channel_factory
):
    thk, topg = channel_factory()  # the shelf reaches the right side
    cfg = cfg_factory()
    cfg.processes.thk.boundary.left = cfg.processes.thk.boundary.right = "periodic"
    state = state_factory(cfg, thk, topg)
    assert not compute_geometry(cfg, state).shelf.numpy().any()


def test_a_hole_in_a_shelf_is_not_open_ocean(
    cfg_factory, state_factory, channel_factory
):
    cfg = cfg_factory()
    thk, topg = channel_factory(front=36)
    thk[5:7, 25:27] = 0.0
    state = state_factory(cfg, thk, topg)
    geom = compute_geometry(cfg, state)
    hole = np.zeros_like(thk, bool)
    hole[5:7, 25:27] = True
    np.testing.assert_array_equal(geom.holes.numpy(), hole)
    np.testing.assert_array_equal(geom.open_ocean.numpy(), (thk == 0) & ~hole)
    _, d_cf = shelf_distances(geom, shelf_labels(geom))
    # Around the hole, the distance to the front goes on through it.
    assert d_cf.numpy()[5, 24] == 36 - 24 and d_cf.numpy()[5, 27] == 36 - 27


def test_floating_fraction_on_a_single_row(cfg_factory, state_factory, channel_factory):
    from igm.processes.bmb.geometry import floating_fraction

    phi = np.linspace(40.0, -40.0, 12, dtype=np.float32)
    one_row = floating_fraction(phi[None, :], 4).numpy()[0]
    two_rows = floating_fraction(np.stack([phi, phi]), 4).numpy()[0]
    np.testing.assert_array_equal(one_row, two_rows)


def test_segment_means_are_exact_over_millions_of_cells():
    import tensorflow as tf

    from igm.processes.bmb.utils import segment_means

    rng = np.random.default_rng(0)
    salinity = tf.constant(34.5 + 0.2 * rng.standard_normal((1000, 3000)), tf.float32)
    ids = tf.ones(salinity.shape, tf.int32)
    (mean,), count = segment_means([salinity], ids, tf.ones(salinity.shape, bool), 2)
    np.testing.assert_allclose(mean.numpy()[0, 0], salinity.numpy().mean(), rtol=1e-6)
    assert count.numpy()[0, 0] == 3.0e6
