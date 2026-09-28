#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from collections import deque

import numpy as np
import pytest
import tensorflow as tf
from scipy import ndimage

from igm.utils.math.connectivity import graph_distance, label_components, reach

EDGES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bfs(seed, domain):
    """Reference: breadth-first search, seed cells at distance 1."""
    dist = np.zeros(domain.shape, np.int64)
    queue = deque()
    for i, j in zip(*np.nonzero(seed & domain)):
        dist[i, j] = 1
        queue.append((i, j))
    while queue:
        i, j = queue.popleft()
        for di, dj in EDGES:
            a, b = i + di, j + dj
            if 0 <= a < domain.shape[0] and 0 <= b < domain.shape[1]:
                if domain[a, b] and dist[a, b] == 0:
                    dist[a, b] = dist[i, j] + 1
                    queue.append((a, b))
    return dist


def _same_partition(labels, reference):
    """Both labelings define the same components (label values may differ)."""
    pairs = set(zip(labels[reference > 0].ravel(), reference[reference > 0].ravel()))
    return (
        len(pairs) == len(np.unique(labels[reference > 0]))
        and len(pairs) == len(np.unique(reference[reference > 0]))
        and np.array_equal(labels == 0, reference == 0)
    )


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_labels_match_scipy_partition(seed):
    domain = np.random.default_rng(seed).random((37, 53)) < 0.55
    labels = label_components(tf.constant(domain)).numpy()
    reference, _ = ndimage.label(domain)  # edge connectivity by default
    assert _same_partition(labels, reference)
    assert labels.max() <= domain.size


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1])
def test_distance_and_reach_match_bfs(seed):
    rng = np.random.default_rng(seed)
    domain = rng.random((41, 29)) < 0.7
    sources = domain & (rng.random(domain.shape) < 0.01)
    dist = graph_distance(tf.constant(sources), tf.constant(domain)).numpy()
    np.testing.assert_array_equal(dist, _bfs(sources, domain))
    reached = reach(tf.constant(sources), tf.constant(domain)).numpy()
    np.testing.assert_array_equal(reached, dist > 0)


@pytest.mark.fast
@pytest.mark.unit
def test_serpentine_corridor_converges():
    """A winding corridor, whose graph diameter is most of its cells."""
    domain = np.zeros((21, 21), bool)
    domain[::2, :] = True
    for k, row in enumerate(range(1, 21, 2)):
        domain[row, -1 if k % 2 == 0 else 0] = True
    sources = np.zeros_like(domain)
    sources[0, 0] = True
    dist = graph_distance(tf.constant(sources), tf.constant(domain)).numpy()
    np.testing.assert_array_equal(dist, _bfs(sources, domain))
    assert dist.max() == domain.sum()
