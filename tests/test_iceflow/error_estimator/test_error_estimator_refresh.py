"""Adaptive refresh of the frozen Hessian: count cap and change trigger."""

import numpy as np
import tensorflow as tf

from test_error_estimator_nonlinear import _estimator, _make_energy


def _uv(array):
    return tf.constant(array[:, :2], tf.float64), tf.constant(array[:, 2:], tf.float64)


def test_refresh_triggers_on_change_and_on_count():
    rng = np.random.default_rng(8)
    target = tf.constant(rng.normal(size=(1, 4, 5, 7)), tf.float64)
    energy = _make_energy(target)
    inputs = tf.ones((1, 5, 7, 1), tf.float64)
    base = target.numpy() + 0.1 * rng.normal(size=(1, 4, 5, 7))
    est = _estimator(
        energy, cg_iters=[30], operator_update_freq=4, operator_refresh_rel_change=0.05
    )

    # First call always prepares.
    assert est.estimate(*_uv(base), inputs)[2]
    # Tiny move (1e-3 relative): reuse.
    small = base * (1.0 + 1e-3)
    assert not est.estimate(*_uv(small), inputs)[2]
    # Large move (20 % relative): refresh although the count cap is not reached.
    large = base * 1.2
    assert est.estimate(*_uv(large), inputs)[2]
    # Then three reuses and a forced refresh on the fourth call (count cap).
    flags = [est.estimate(*_uv(large * (1.0 + 1e-4)), inputs)[2] for _ in range(4)]
    assert flags == [False, False, False, True]


def test_update_freq_one_always_prepares():
    rng = np.random.default_rng(9)
    target = tf.constant(rng.normal(size=(1, 4, 5, 7)), tf.float64)
    energy = _make_energy(target)
    inputs = tf.ones((1, 5, 7, 1), tf.float64)
    base = target.numpy() + 0.1 * rng.normal(size=(1, 4, 5, 7))
    est = _estimator(energy, cg_iters=[30], operator_update_freq=1)
    assert all(est.estimate(*_uv(base), inputs)[2] for _ in range(3))


def test_update_freq_zero_has_no_count_cap_but_tracks_velocity():
    rng = np.random.default_rng(10)
    target = tf.constant(rng.normal(size=(1, 4, 5, 7)), tf.float64)
    energy = _make_energy(target)
    inputs = tf.ones((1, 5, 7, 1), tf.float64)
    base = target.numpy() + 0.1 * rng.normal(size=(1, 4, 5, 7))
    est = _estimator(
        energy,
        cg_iters=[1],
        operator_update_freq=0,
        operator_refresh_rel_change=0.05,
    )

    assert est.estimate(*_uv(base), inputs)[2]
    assert not any(est.estimate(*_uv(base), inputs)[2] for _ in range(6))
    # Compare against the last refresh, not against the preceding estimate.
    assert not est.estimate(*_uv(base * 1.03), inputs)[2]
    assert est.estimate(*_uv(base * 1.06), inputs)[2]
    assert not est.estimate(*_uv(base * 1.06), inputs)[2]


def test_changed_inputs_force_refresh_without_a_count_cap():
    rng = np.random.default_rng(11)
    target = tf.constant(rng.normal(size=(1, 4, 5, 7)), tf.float64)
    energy = _make_energy(target)
    inputs = tf.ones((1, 5, 7, 1), tf.float64)
    base = target.numpy() + 0.1 * rng.normal(size=(1, 4, 5, 7))
    est = _estimator(energy, cg_iters=[1], operator_update_freq=0)

    assert est.estimate(*_uv(base), inputs)[2]
    assert not est.estimate(*_uv(base), inputs)[2]
    changed_inputs = inputs * 1.001
    assert est.estimate(*_uv(base), changed_inputs)[2]
    assert not est.estimate(*_uv(base), tf.identity(changed_inputs))[2]
    assert est.estimate(*_uv(base), inputs)[2]


def test_nonfinite_velocity_change_does_not_reuse_operator():
    target = tf.ones((1, 4, 5, 7), tf.float64)
    inputs = tf.ones((1, 5, 7, 1), tf.float64)
    est = _estimator(_make_energy(target), cg_iters=[1], operator_update_freq=0)
    U, V = _uv(target.numpy())

    assert est.estimate(U, V, inputs)[2]
    assert not est._needs_prepare(U, V, inputs)
    assert est._needs_prepare(U * np.nan, V, inputs)
