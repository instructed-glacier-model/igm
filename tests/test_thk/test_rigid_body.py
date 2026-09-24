#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from igm.processes.thk import thk as thk_module
from igm.processes.thk.rigid_body import anchored_ice_mask, remove_rigid_body_modes

RHO_RATIO = 1028.0 / 918.0


def _geometry():
    """Build grounded, shelf, iceberg, hinged, and isolated test ice."""
    thk = np.zeros((9, 12), np.float32)
    topg = np.full((9, 12), -900.0, np.float32)  # deep water ...
    topg[:, :2] = 100.0  # ... but a grounded strip on the left
    thk[1:6, 0:5] = 500.0  # grounded ice (cols 0-1) + attached shelf (cols 2-4)
    thk[1:3, 7:9] = 200.0  # iceberg: no grounded node in its component
    # Hinged piece: its only shelf contact is the (5, 4)-(6, 5) diagonal.
    thk[6:8, 5:7] = 200.0
    topg[8, 10] = 100.0
    thk[8, 10] = 50.0  # isolated grounded node
    return thk, topg


def test_anchored_mask_keeps_grounded_ice_and_attached_shelf_only():
    thk, topg = _geometry()
    ice = thk > 0
    grounded = ice & (thk + RHO_RATIO * topg > 0)
    keep = anchored_ice_mask(tf.constant(ice), tf.constant(grounded)).numpy()
    assert keep[1:6, 0:5].all()  # grounded strip and its shelf
    assert not keep[1:3, 7:9].any()  # iceberg removed
    # A node-only hinge carries no membrane stress and must not anchor the piece.
    assert not keep[6:8, 5:7].any()
    assert keep[8, 10]  # an isolated grounded node stays


def test_anchored_mask_does_not_propagate_along_a_diagonal_node_chain():
    ice = np.eye(7, dtype=bool)
    grounded = np.zeros_like(ice)
    grounded[0, 0] = True

    keep = anchored_ice_mask(tf.constant(ice), tf.constant(grounded)).numpy()

    assert keep[0, 0]
    assert not keep[1:, 1:].any()


def test_anchored_mask_drops_a_floating_line_without_active_cells():
    ice = np.zeros((5, 8), dtype=bool)
    ice[2, 1:7] = True
    grounded = np.zeros_like(ice)
    grounded[2, 1] = True

    keep = anchored_ice_mask(tf.constant(ice), tf.constant(grounded)).numpy()

    assert keep[2, 1]
    assert not keep[2, 2:].any()


def test_anchored_mask_reaches_the_end_of_a_winding_shelf():
    # This one-cell-wide path in cell space has graph distance greater than
    # either grid dimension, so max(Ny, Nx) flood-fill iterations truncate it.
    cells = np.zeros((9, 9), dtype=bool)
    for row in range(0, 9, 2):
        cells[row, :] = True
        if row + 1 < 9:
            cells[row + 1, 8 if (row // 2) % 2 == 0 else 0] = True
    ice = np.zeros((10, 10), dtype=bool)
    for row, col in np.argwhere(cells):
        ice[row : row + 2, col : col + 2] = True
    grounded = np.zeros_like(ice)
    grounded[0, 0] = True

    keep = anchored_ice_mask(tf.constant(ice), tf.constant(grounded)).numpy()

    np.testing.assert_array_equal(keep, ice)


def test_remove_rigid_body_modes_touches_only_the_dropped_columns():
    thk, topg = _geometry()
    thk_true = thk.copy()
    thk_padded = thk.copy()
    thk_padded[1:6, 5] = 30.0  # sub-grid front padding next to the shelf
    thk_padded[1:3, 9] = 25.0  # padding belonging only to the iceberg
    href = np.where(thk > 0, 1.0, 0.0).astype(np.float32)
    href[3, 5] = 0.4  # a partial front cell
    href[1:3, 9] = 0.3
    state = SimpleNamespace(
        thk=tf.constant(thk_padded),
        thk_true=tf.Variable(thk_true),
        Href=tf.Variable(href),
        topg=tf.constant(topg),
        water_level=tf.constant(0.0),
    )
    remove_rigid_body_modes(state, RHO_RATIO)
    out = state.thk.numpy()
    assert not out[1:3, 7:9].any() and not state.thk_true.numpy()[1:3, 7:9].any()
    assert not state.Href.numpy()[6:8, 5:7].any()
    np.testing.assert_array_equal(out[1:6, 0:5], 500.0)  # anchored ice untouched
    np.testing.assert_array_equal(out[1:6, 5], 30.0)  # front padding untouched
    assert state.Href.numpy()[3, 5] == np.float32(0.4)
    assert not out[1:3, 9].any() and not state.Href.numpy()[1:3, 9].any()
    assert out[8, 10] == 50.0  # grounded, however isolated, is never removed


def test_thickness_update_runs_rigid_body_cleanup_only_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        thk_module,
        "remove_rigid_body_modes",
        lambda state, rho_ratio: calls.append(rho_ratio),
    )
    monkeypatch.setattr(thk_module, "update_surfaces", lambda cfg, state: None)
    cfg = SimpleNamespace(outputs=None)

    for enabled in (False, True):
        state = SimpleNamespace(
            it=0,
            thk_components=SimpleNamespace(
                domain_constraints=(),
                pipeline=(),
                remove_rigid_body_modes=enabled,
                rho_ratio=RHO_RATIO,
            ),
        )
        thk_module.update(cfg, state)

    assert calls == [RHO_RATIO]
