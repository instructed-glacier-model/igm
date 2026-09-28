#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""PICO against a plain NumPy implementation of Reese et al. (2018).

The reference derives its own masks from thickness and bed with scipy, uses
breadth-first distances, loops over shelves and boxes and applies the
square-root box boundaries, so that it shares no construct with the
vectorised implementation. The geometry has two shelves over four basins, a
hole in a shelf (not a calving front) and an iceberg (Beckmann-Goosse melt);
the cold ocean activates the clamp of the input temperature.
"""

from collections import deque

import numpy as np
import pytest
from scipy import ndimage

from igm.processes.bmb.geometry import compute_geometry
from igm.processes.bmb.laws.pico import pico

from conftest import RHO_I, RHO_W

pytestmark = [pytest.mark.fast, pytest.mark.unit]

SPY = 31556926.0
GRAVITY = 9.81


def _bfs(seed, domain):
    dist = np.zeros(domain.shape, int)
    queue = deque(zip(*np.nonzero(seed & domain)))
    dist[seed & domain] = 1
    while queue:
        i, j = queue.popleft()
        for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= a < domain.shape[0] and 0 <= b < domain.shape[1]:
                if domain[a, b] and not dist[a, b]:
                    dist[a, b] = dist[i, j] + 1
                    queue.append((a, b))
    return dist


def _near(mask, diagonal):
    p = np.pad(mask, 1)
    ny, nx = mask.shape
    out = np.zeros_like(mask)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di, dj) != (0, 0) and (diagonal or di == 0 or dj == 0):
                out |= p[1 + di : 1 + di + ny, 1 + dj : 1 + dj + nx]
    return out


def _touching_right_side(mask):
    labels, _ = ndimage.label(mask)
    return np.isin(labels, np.unique(labels[:, -1][mask[:, -1]]))


def reference_pico(thk, topg, temp, salt, basins, dx, p):
    """PICO melt (m ice yr-1) with the ocean open on the right side only."""
    freezing = (
        lambda s, z: p.freezing_point.a * s
        + p.freezing_point.b
        + (p.freezing_point.c * RHO_W * GRAVITY * z)
    )
    nu_lambda = RHO_I / RHO_W * 3.34e5 / 3974.0
    ice = thk > 0
    grounded = thk + RHO_W / RHO_I * topg > 0
    draft = np.minimum(np.where(ice, np.maximum(topg, -RHO_I / RHO_W * thk), topg), 0)
    exposed = _touching_right_side(~grounded)
    shelf = exposed & ice
    open_ocean = _touching_right_side(exposed & ~ice)
    holes = exposed & ~ice & ~open_ocean

    labels, n_shelves = ndimage.label(shelf)
    domain = shelf | holes
    d_gl = _bfs(shelf & _near(grounded & ice, True), domain)
    d_cf = _bfs(shelf & _near(open_ocean, False), domain)

    sea = open_ocean & (topg > p.continental_shelf_depth)
    basin_t = {b: temp[sea & (basins == b)].mean() for b in np.unique(basins[sea])}
    basin_s = {b: salt[sea & (basins == b)].mean() for b in np.unique(basins[sea])}

    d_ref = d_gl[shelf].max()
    melt = np.zeros(shelf.shape)
    for s in range(1, n_shelves + 1):
        nodes = labels == s
        known = [b for b in basins[nodes] if b in basin_t]
        t0 = np.mean([basin_t[b] for b in known])
        s0 = np.mean([basin_s[b] for b in known])
        t0_node = np.maximum(t0, freezing(s0, draft) + 1e-3)
        if d_gl[nodes].max() == 0:  # no grounding line: Beckmann and Goosse
            bg = 5e-3 * p.gamma_T / nu_lambda * (t0_node - freezing(s0, draft)) * SPY
            melt[nodes] = bg[nodes]
            continue
        n_d = int(
            np.floor(1 + np.sqrt(d_gl[nodes].max() / d_ref) * (p.n_boxes - 1) + 0.5)
        )
        n_d = min(max(n_d, 1), p.n_boxes)

        box = np.zeros(shelf.shape, int)
        for i, j in zip(*np.nonzero(nodes)):
            r = d_gl[i, j] / (d_gl[i, j] + d_cf[i, j])
            for k in range(1, n_d + 1):
                if 1 - np.sqrt(1 - (k - 1) / n_d) <= r < 1 - np.sqrt(1 - k / n_d):
                    box[i, j] = min(k, d_gl[i, j])
        area = {k: (box == k).sum() * dx * dx for k in range(1, n_d + 1)}

        temp_box, salt_box = np.zeros(shelf.shape), np.zeros(shelf.shape)
        in1 = box == 1
        t_star = freezing(s0, draft[in1]) - t0_node[in1]
        s_fac = s0 / nu_lambda
        pc = (
            area[1]
            * p.gamma_T
            / (p.overturning * p.rho_star * (p.beta * s_fac - p.alpha))
        )
        x = -pc / 2 + np.sqrt(pc**2 / 4 - pc * t_star)
        temp_box[in1], salt_box[in1] = t0_node[in1] - x, s0 - s_fac * x
        q = (p.overturning * p.rho_star * (p.beta * s_fac - p.alpha) * x).mean()
        t_in, s_in = temp_box[in1].mean(), salt_box[in1].mean()
        a = p.freezing_point.a
        for k in range(2, n_d + 1):
            ink = box == k
            g1 = area[k] * p.gamma_T
            t_star = freezing(s_in, draft[ink]) - t_in
            x = -g1 * t_star / (q + g1 - g1 / nu_lambda * a * s_in)
            temp_box[ink], salt_box[ink] = t_in - x, s_in * (1 - x / nu_lambda)
            t_in, s_in = temp_box[ink].mean(), salt_box[ink].mean()
        melt[nodes] = (
            p.gamma_T / nu_lambda * SPY * (temp_box - freezing(salt_box, draft))[nodes]
        )
    return melt, shelf, holes


@pytest.mark.parametrize(
    "profile", [None, [["z", "temp", "salinity"], [0.0, -2.3, 34.5]]]
)
def test_pico_matches_an_independent_reference(
    profile, cfg_factory, state_factory, channel_factory
):
    cfg = cfg_factory(method="pico")
    if profile is not None:  # cold enough to clamp the input temperature
        cfg.processes.ocean.profile.array = profile
    thk, topg = channel_factory(ny=13, nx=44, front=36)
    topg[6, :] = 400.0  # a grounded ridge splits the shelf in two
    thk[6, :36] = 800.0
    thk[7:, :36] *= 0.8  # the southern shelf is thinner, with a different draft
    thk[2:4, 27:29] = 0.0  # a hole in the northern shelf
    thk[9:11, 40:42] = 200.0  # an iceberg in the open ocean
    # Four basins cut each shelf unevenly (2 and 4 rows, 3 and 3 rows), with
    # outer oceans of different depths, so each shelf mixes two basins.
    rows = np.arange(13)[:, None] * np.ones((1, 44))
    basins = 1.0 + (rows >= 2) + (rows >= 7) + (rows >= 10)
    topg[:, 36:] = np.where((basins[:, 36:] % 2) == 0, -560.0, -720.0)
    state = state_factory(cfg, thk, topg, basins=basins)

    melt = pico.solve_boxes(cfg, state, compute_geometry(cfg, state)).melt.numpy()
    reference, shelf, holes = reference_pico(
        thk.astype(np.float64),
        topg.astype(np.float64),
        state.ocean_temp.numpy().astype(np.float64),
        state.ocean_salinity.numpy().astype(np.float64),
        basins.astype(int),
        1000.0,
        cfg.processes.bmb.pico,
    )
    assert holes.sum() == 4 and len(np.unique(ndimage.label(shelf)[0][shelf])) == 3
    assert np.abs(reference[shelf]).max() > 0.05
    np.testing.assert_allclose(melt, reference, rtol=2e-4, atol=1e-4)
