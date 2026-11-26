import numpy as np
import tensorflow as tf

from igm.processes.iceflow.optimizers.utils.pcg import (
    pcg_init,
    pcg_relres,
    pcg_run,
    pcg_solve,
)

HESSIAN = np.array(
    [
        [4.0, 1.0, 0.0, 0.5, 0.0, 0.0],
        [1.0, 5.0, 1.0, 0.0, 0.5, 0.0],
        [0.0, 1.0, 4.0, 0.0, 0.0, 0.5],
        [0.5, 0.0, 0.0, 3.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5, 4.0, 0.5],
        [0.0, 0.0, 0.5, 0.0, 0.5, 3.0],
    ]
)
RHS = np.array([1.0, -2.0, 0.75, 3.0, -1.0, 2.5])


def _system(dtype=tf.float64):
    H = tf.constant(HESSIAN, dtype)
    b = tf.constant(RHS, dtype)
    A = lambda v: tf.linalg.matvec(H, v)
    return H, b, A


def test_pcg_solve_matches_direct_solve():
    H, b, A = _system()
    exact = tf.linalg.solve(H, b[:, None])[:, 0]
    result = pcg_solve(A, lambda v: v, b, tf.zeros_like(b), 6, 1.0e-14)
    np.testing.assert_allclose(result.x.numpy(), exact.numpy(), rtol=1e-12, atol=1e-12)
    assert int(result.iters) <= 6
    assert float(result.relres) < 1e-12
    # <x, b> accumulated in the recurrence equals the direct inner product.
    np.testing.assert_allclose(
        float(result.xb), float(tf.tensordot(result.x, b, 1)), rtol=1e-12
    )


def test_pcg_solve_with_exact_preconditioner_converges_in_one_iteration():
    H, b, A = _system()
    H_inv = tf.linalg.inv(H)
    result = pcg_solve(
        A, lambda v: tf.linalg.matvec(H_inv, v), b, tf.zeros_like(b), 6, 1e-14
    )
    assert int(result.iters) == 1
    assert float(result.relres) < 1e-12


def test_pcg_zero_iterations_returns_initial_guess():
    _, b, A = _system()
    x0 = tf.constant(np.arange(6, dtype=np.float64))
    result = pcg_solve(A, lambda v: v, b, x0, 0, 1e-14)
    np.testing.assert_array_equal(result.x.numpy(), x0.numpy())
    assert int(result.iters) == 0


def test_pcg_run_is_resumable():
    _, b, A = _system()
    pre = lambda v: v
    state = pcg_init(A, pre, b, tf.zeros_like(b))
    two_segments = pcg_run(A, pre, pcg_run(A, pre, state, 3, 0.0), 3, 0.0)
    one_segment = pcg_run(A, pre, state, 6, 0.0)
    np.testing.assert_allclose(
        two_segments.x.numpy(), one_segment.x.numpy(), rtol=1e-12, atol=1e-12
    )
    assert int(two_segments.iters) == 6
    # The energy <x, b> = ||x||^2_A is monotone along the Krylov iterates.
    energies = [float(pcg_run(A, pre, state, k, 0.0).xb) for k in range(1, 7)]
    assert all(
        later >= earlier - 1e-12 for earlier, later in zip(energies, energies[1:])
    )


def test_pcg_relres_is_true_residual():
    H, b, A = _system()
    x = tf.constant(np.ones(6))
    expected = np.linalg.norm(RHS - HESSIAN @ np.ones(6)) / np.linalg.norm(RHS)
    np.testing.assert_allclose(float(pcg_relres(A, b, x)), expected, rtol=1e-12)


def test_pcg_traces_inside_tf_function():
    _, b, A = _system()

    @tf.function
    def solve(rhs):
        return pcg_solve(A, lambda v: v, rhs, tf.zeros_like(rhs), 6, 1e-14).x

    exact = np.linalg.solve(HESSIAN, RHS)
    np.testing.assert_allclose(solve(b).numpy(), exact, rtol=1e-12, atol=1e-12)
