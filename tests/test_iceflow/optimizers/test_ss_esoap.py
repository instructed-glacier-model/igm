import numpy as np
import pytest
import tensorflow as tf

from igm.processes.iceflow.optimizers.ss_esoap import (
    OptimizerSSESOAP,
    _SSESOAPLayerState,
)


def _optimizer(**kwargs):
    return OptimizerSSESOAP(
        cost_fn=lambda U, V, inputs: tf.reduce_sum(U + V),
        map=None,
        precision="double",
        print_cost=False,
        **kwargs,
    )


def test_rank4_matrixization_is_balanced_and_invertible():
    state = _SSESOAPLayerState([4, 4, 24, 8], tf.float64)
    tensor = tf.reshape(tf.range(4 * 4 * 24 * 8, dtype=tf.float64), [4, 4, 24, 8])

    matrix = state.matrixize(tensor)

    assert state.uses_balanced_matrix
    assert state.matrix_shape == [96, 32]
    np.testing.assert_array_equal(matrix.shape, [96, 32])
    np.testing.assert_array_equal(state.tensorize(matrix), tensor)

    small = _SSESOAPLayerState([2, 2, 48, 48], tf.float64)
    assert not small.uses_balanced_matrix
    assert small.matrix_shape == [192, 48]


def test_incremental_rayleigh_diagonals_match_direct_computation():
    opt = _optimizer(tau_trigger=2.0)
    dtype = tf.float64
    size = 128
    G = tf.random.stateless_normal([size, size], seed=[3, 7], dtype=dtype) / 10.0
    L_old = tf.linalg.diag(tf.linspace(tf.constant(0.5, dtype), 1.5, size))
    R_old = tf.linalg.diag(tf.linspace(tf.constant(0.7, dtype), 1.7, size))
    QL = tf.reverse(tf.eye(size, dtype=dtype), axis=[0])
    QR = tf.roll(tf.eye(size, dtype=dtype), shift=7, axis=0)
    DL_old = tf.linalg.diag_part(tf.matmul(tf.matmul(QL, L_old, transpose_a=True), QL))
    DR_old = tf.linalg.diag_part(tf.matmul(tf.matmul(QR, R_old, transpose_a=True), QR))
    zeros = tf.zeros_like(G)

    result = opt._matrix_step_kernel(
        G,
        L_old,
        R_old,
        QL,
        QR,
        DL_old,
        DR_old,
        zeros,
        zeros,
        zeros,
        zeros,
        tf.constant(1e-3, dtype),
        tf.constant(True),
        tf.constant(False),
        tf.constant(0.1, dtype),
        tf.constant(0.05, dtype),
    )
    _, L, R, QL_new, QR_new, DL, DR, *_ = result
    DL_direct = tf.linalg.diag_part(
        tf.matmul(tf.matmul(QL_new, L, transpose_a=True), QL_new)
    )
    DR_direct = tf.linalg.diag_part(
        tf.matmul(tf.matmul(QR_new, R, transpose_a=True), QR_new)
    )

    np.testing.assert_allclose(DL, DL_direct, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(DR, DR_direct, rtol=1e-12, atol=1e-12)


def test_first_step_fast_path_and_shape_cache_trace_once():
    opt = _optimizer(tau_trigger=2.0)
    state = _SSESOAPLayerState([4, 4, 32, 32], tf.float64)
    grad = tf.random.stateless_normal(state.shape, seed=[11, 13], dtype=tf.float64)

    update = opt._matrix_step(
        grad,
        state,
        should_check=tf.constant(True),
        first_step=tf.constant(True),
        bc1=tf.constant(0.1, tf.float64),
        bc2=tf.constant(0.05, tf.float64),
        lr=tf.constant(3e-4, tf.float64),
    )
    update_2 = opt._matrix_step(
        grad,
        state,
        should_check=tf.constant(True),
        first_step=tf.constant(False),
        bc1=tf.constant(0.19, tf.float64),
        bc2=tf.constant(0.0975, tf.float64),
        lr=tf.constant(3e-4, tf.float64),
    )

    assert np.isfinite(update.numpy()).all()
    assert np.isfinite(update_2.numpy()).all()
    assert opt._get_matrix_step_fn(128, 128) is opt._get_matrix_step_fn(128, 128)
    assert opt._get_matrix_step_fn(128, 128).experimental_get_tracing_count() == 1


def test_nondefault_trigger_schedule_keeps_conditional_path():
    opt = _optimizer(check_freq=5, warmup=2)

    assert opt._check_every_step is False
    assert int(opt.check_freq) == 5
    assert int(opt.warmup) == 2


def test_learning_rate_drop_retains_base_rate_before_milestone():
    opt = _optimizer(lr=1e-3, lr_drop_iter=1000, lr_drop_factor=0.3)

    np.testing.assert_allclose(opt._learning_rate_at(tf.constant(999)), 1e-3)
    np.testing.assert_allclose(opt._learning_rate_at(tf.constant(1000)), 3e-4)


def test_automatic_drop_uses_windowed_relative_cost_improvement():
    opt = _optimizer(
        lr=1e-3,
        lr_drop_factor=0.2,
        lr_auto_drop_patience=10,
        lr_auto_drop_warmup=5,
        lr_auto_drop_rel_improvement=0.01,
    )
    lr = tf.constant(1e-3, tf.float64)
    reference = tf.constant(float("inf"), tf.float64)
    dropped = tf.constant(False)

    lr, reference, dropped, fired = opt._automatic_drop_update(
        tf.constant(5), tf.constant(100.0), lr, reference, dropped
    )
    assert not bool(fired)
    lr, reference, dropped, fired = opt._automatic_drop_update(
        tf.constant(15), tf.constant(99.5), lr, reference, dropped
    )

    assert bool(fired)
    assert bool(dropped)
    np.testing.assert_allclose(lr, 2e-4)

    lr, reference, dropped, fired = opt._automatic_drop_update(
        tf.constant(25), tf.constant(99.4), lr, reference, dropped
    )
    assert not bool(fired)
    np.testing.assert_allclose(lr, 2e-4)


def test_automatic_drop_ignores_nonfinite_cost_samples():
    opt = _optimizer(
        lr=1e-3,
        lr_drop_factor=0.2,
        lr_auto_drop_patience=10,
        lr_auto_drop_warmup=5,
        lr_auto_drop_rel_improvement=0.01,
    )
    lr = tf.constant(1e-3, tf.float64)
    reference = tf.constant(float("inf"), tf.float64)
    dropped = tf.constant(False)

    lr, reference, dropped, fired = opt._automatic_drop_update(
        tf.constant(5), tf.constant(float("nan"), tf.float64), lr, reference, dropped
    )
    assert not bool(fired)
    assert np.isinf(reference)

    lr, reference, dropped, fired = opt._automatic_drop_update(
        tf.constant(15), tf.constant(100.0), lr, reference, dropped
    )
    assert not bool(fired)
    np.testing.assert_allclose(reference, 100.0)


def test_learning_rate_drop_configuration_rejects_noops_and_conflicts():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _optimizer(lr_drop_iter=10, lr_auto_drop_patience=10)

    with pytest.raises(ValueError, match="smaller than 1"):
        _optimizer(lr_auto_drop_patience=10, lr_drop_factor=1.0)
