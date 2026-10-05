#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import pytest
import tensorflow as tf

from igm.processes.bmb.geometry import floating_fraction
from igm.processes.bmb.grounding_line import extend, melt_weights

pytestmark = [pytest.mark.fast, pytest.mark.unit]


def test_floating_fraction_is_zero_or_one_away_from_the_grounding_line():
    # phi changes sign at x = 5 + 50/60, inside the dual cell of node 6.
    phi = np.where(np.arange(12) < 6, 50.0, -10.0)[None, :].repeat(5, 0)
    fraction = floating_fraction(tf.constant(phi, tf.float32), 4).numpy()
    np.testing.assert_array_equal(fraction[:, :6], 0.0)
    np.testing.assert_array_equal(fraction[:, 7:], 1.0)
    assert (0.0 < fraction[:, 6]).all() and (fraction[:, 6] < 1.0).all()


@pytest.mark.parametrize("x0", [3.1, 3.5, 4.2, 4.9])
def test_floating_fraction_is_exact_for_a_linear_flotation_function(x0):
    """Bilinear interpolation is exact for phi = x0 - x, so only sampling errs."""
    n = 4
    x = np.arange(9, dtype=np.float64)
    phi = tf.constant(np.repeat((x0 - x)[None, :], 4, axis=0), tf.float32)
    fraction = floating_fraction(phi, n).numpy()[0]

    offsets = (np.arange(n) + 0.5) / (2 * n)
    samples = x[:, None] + np.concatenate([-offsets, offsets])[None, :]
    expected = (x0 - samples <= 0.0).mean(axis=1)
    np.testing.assert_allclose(fraction[1:-1], expected[1:-1])
    # The dual cell of an edge node is mirrored inside the domain.
    np.testing.assert_allclose(fraction[0], float(x0 <= offsets[0]))


def test_melt_weights_of_each_treatment():
    fraction = tf.constant([0.0, 0.25, 1.0, 1.0])
    grounded = tf.constant([True, True, False, False])
    shelf = tf.constant([False, False, True, False])  # the last node is a lake
    expected = {
        "nmp": ([0, 0, 1, 0], [1, 0.75, 0, 0]),
        "fmp": ([0, 1, 1, 0], [1, 0, 0, 0]),
        "pmp": ([0, 0.25, 1, 0], [1, 0.75, 0, 0]),
    }
    for treatment, (w_ref, g_ref) in expected.items():
        w, g = melt_weights(treatment, fraction, grounded, shelf)
        np.testing.assert_allclose(w.numpy(), w_ref, err_msg=treatment)
        np.testing.assert_allclose(g.numpy(), g_ref, err_msg=treatment)


def test_extension_averages_the_shelf_neighbours():
    shelf = np.zeros((3, 4), bool)
    shelf[:, 2:] = True
    melt = np.where(shelf, np.arange(12, dtype=np.float32).reshape(3, 4), 0.0)
    extended = extend(tf.constant(melt), tf.constant(shelf)).numpy()
    np.testing.assert_array_equal(extended[:, 2:], melt[:, 2:])
    np.testing.assert_allclose(extended[1, 1], np.mean([2.0, 6.0, 10.0]))
    np.testing.assert_allclose(extended[0, 1], np.mean([2.0, 6.0]))
    np.testing.assert_array_equal(extended[:, 0], 0.0)
