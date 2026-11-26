"""A second chord step tightens the estimate on a nonlinear energy."""

import numpy as np
import tensorflow as tf

from test_error_estimator_nonlinear import _estimator, _make_energy


def test_second_step_is_noop_for_quadratic_energy():
    rng = np.random.default_rng(3)
    target = tf.constant(rng.normal(size=(1, 4, 5, 7)), tf.float64)
    energy = _make_energy(target, quartic=0.0)
    inputs = tf.ones((1, 5, 7, 1), tf.float64)
    start = target.numpy() + 0.3 * rng.normal(size=(1, 4, 5, 7))
    U0, V0 = tf.constant(start[:, :2], tf.float64), tf.constant(start[:, 2:], tf.float64)
    est = _estimator(energy, cg_iters=[60], newton_steps=2)
    scalars, *_ = est.estimate(U0, V0, inputs)
    np.testing.assert_allclose(scalars["step2_delta_l2"], scalars["k60_delta_l2"], rtol=1e-8)
    assert scalars["step2_grad_l2"] < 1e-8 * max(scalars["grad_l2"], 1e-30)
    assert scalars["est_delta_l2"] == scalars["step2_delta_l2"]


def test_second_step_reduces_nonlinear_residual():
    rng = np.random.default_rng(4)
    target = tf.constant(rng.normal(size=(1, 4, 5, 7)), tf.float64)
    energy = _make_energy(target, quartic=0.3)
    inputs = tf.ones((1, 5, 7, 1), tf.float64)
    direction = rng.normal(size=(1, 4, 5, 7))
    direction /= np.linalg.norm(direction)
    eps = 0.4
    start = target.numpy() + eps * direction
    U0, V0 = tf.constant(start[:, :2], tf.float64), tf.constant(start[:, 2:], tf.float64)

    one = _estimator(energy, cg_iters=[60], newton_steps=1)
    two = _estimator(energy, cg_iters=[60], newton_steps=2)
    s1, *_ = one.estimate(U0, V0, inputs)
    s2, *_ = two.estimate(U0, V0, inputs)
    err1 = abs(s1["est_delta_l2"] - eps)
    err2 = abs(s2["est_delta_l2"] - eps)
    assert err2 < 0.5 * err1
    # The first-step tiers are unchanged by the extra step.
    np.testing.assert_allclose(s2["k60_delta_l2"], s1["k60_delta_l2"], rtol=1e-10)
