import numpy as np
import pytest
import tensorflow as tf

from igm.utils.math.tridiagonal import solve_block_tridiagonal, solve_tridiagonal_pcr


def _dense_from_blocks(lower, diag, upper):
    """Assemble a dense reference matrix from (N,2,2) block bands (batch=1)."""
    n = diag.shape[0]
    size = 2 * n
    dense = np.zeros((size, size), dtype=diag.dtype)
    for i in range(n):
        dense[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = diag[i]
        if i > 0:
            dense[2 * i : 2 * i + 2, 2 * i - 2 : 2 * i] = lower[i]
        if i < n - 1:
            dense[2 * i : 2 * i + 2, 2 * i + 2 : 2 * i + 4] = upper[i]
    return dense


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 13, 64, 100, 257])
def test_solve_block_tridiagonal_matches_dense_reference(n):
    """Checks PCR against a dense solve across power-of-2 and odd chain lengths."""
    dtype = np.float64
    rng = np.random.default_rng(n)

    # Random SPD-ish blocks: strictly diagonally dominant diag blocks.
    lower = rng.normal(size=(n, 2, 2)).astype(dtype) * 0.1
    upper = rng.normal(size=(n, 2, 2)).astype(dtype) * 0.1
    diag = rng.normal(size=(n, 2, 2)).astype(dtype) * 0.1
    for i in range(n):
        diag[i] += 5.0 * np.eye(2, dtype=dtype)
    lower[0] = 0.0
    upper[-1] = 0.0

    rhs = rng.normal(size=(n, 2)).astype(dtype)

    dense = _dense_from_blocks(lower, diag, upper)
    expected = np.linalg.solve(dense, rhs.reshape(-1))

    lower_tf = tf.constant(lower.transpose(1, 2, 0)[np.newaxis], dtype=tf.float64)
    diag_tf = tf.constant(diag.transpose(1, 2, 0)[np.newaxis], dtype=tf.float64)
    upper_tf = tf.constant(upper.transpose(1, 2, 0)[np.newaxis], dtype=tf.float64)
    rhs_tf = tf.constant(rhs.transpose(1, 0)[np.newaxis], dtype=tf.float64)

    solution = solve_block_tridiagonal(lower_tf, diag_tf, upper_tf, rhs_tf)
    solution_np = solution.numpy()[0].transpose(1, 0).reshape(-1)

    np.testing.assert_allclose(solution_np, expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("n", [1, 3, 8, 13, 257])
def test_solve_scalar_tridiagonal_matches_dense_reference(n):
    rng = np.random.default_rng(1000 + n)
    lower = rng.normal(size=n) * 0.1
    upper = rng.normal(size=n) * 0.1
    diagonal = rng.normal(size=n) * 0.1 + 4.0
    lower[0] = 0.0
    upper[-1] = 0.0
    rhs = rng.normal(size=n)

    dense = np.diag(diagonal)
    if n > 1:
        dense += np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    expected = np.linalg.solve(dense, rhs)
    actual = solve_tridiagonal_pcr(
        tf.constant(lower[None], tf.float64),
        tf.constant(diagonal[None], tf.float64),
        tf.constant(upper[None], tf.float64),
        tf.constant(rhs[None], tf.float64),
    )[0]
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-10, atol=1e-10)


def test_solve_block_tridiagonal_handles_fully_degenerate_node():
    """A node with an exactly-zero block (damping=0, dead dof) must not poison
    the rest of the chain via NaN/Inf -- this is what broke the earlier
    sequential Thomas implementation."""
    dtype = np.float64
    n = 9
    rng = np.random.default_rng(11)

    lower = rng.normal(size=(n, 2, 2)).astype(dtype) * 0.1
    upper = rng.normal(size=(n, 2, 2)).astype(dtype) * 0.1
    diag = rng.normal(size=(n, 2, 2)).astype(dtype) * 0.1
    for i in range(n):
        diag[i] += 5.0 * np.eye(2, dtype=dtype)
    lower[0] = 0.0
    upper[-1] = 0.0

    # Node 3 is fully dead: zero row/column in the dense matrix, consistent
    # with a Dirichlet/duplicated-row degree of freedom under damping=0.
    dead = 3
    diag[dead] = 0.0
    lower[dead] = 0.0
    upper[dead] = 0.0
    lower[dead + 1] = 0.0
    upper[dead - 1] = 0.0

    rhs = rng.normal(size=(n, 2)).astype(dtype)
    rhs[dead] = 0.0

    dense = _dense_from_blocks(lower, diag, upper)
    # The dead row is 0*x = 0: solvable for any value there, so drop it (and
    # its column) before inverting, matching what the floor regularization
    # effectively decides for that direction.
    keep = [i for i in range(2 * n) if i not in (2 * dead, 2 * dead + 1)]
    reduced = np.linalg.solve(dense[np.ix_(keep, keep)], rhs.reshape(-1)[keep])
    expected = np.zeros(2 * n, dtype=dtype)
    expected[keep] = reduced

    lower_tf = tf.constant(lower.transpose(1, 2, 0)[np.newaxis], dtype=tf.float64)
    diag_tf = tf.constant(diag.transpose(1, 2, 0)[np.newaxis], dtype=tf.float64)
    upper_tf = tf.constant(upper.transpose(1, 2, 0)[np.newaxis], dtype=tf.float64)
    rhs_tf = tf.constant(rhs.transpose(1, 0)[np.newaxis], dtype=tf.float64)

    solution = solve_block_tridiagonal(lower_tf, diag_tf, upper_tf, rhs_tf)
    solution_np = solution.numpy()[0].transpose(1, 0).reshape(-1)

    assert np.all(np.isfinite(solution_np))
    np.testing.assert_allclose(
        np.delete(solution_np, [2 * dead, 2 * dead + 1]),
        np.delete(expected, [2 * dead, 2 * dead + 1]),
        rtol=1e-6,
        atol=1e-6,
    )
