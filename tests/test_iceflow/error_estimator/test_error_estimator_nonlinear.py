"""Nonlinear energy on a MOLHO grid with the banded operator and multigrid."""

import numpy as np
import pytest
import tensorflow as tf

from igm.processes.iceflow.error_estimator import ErrorEstimator

SHAPE = (1, 2, 5, 7)


def _make_energy(target: tf.Tensor, quartic: float = 0.3):
    def energy(U, V, inputs):
        del inputs
        c = tf.concat([U, V], axis=1) - target
        point = 0.5 * tf.reduce_sum(c * c)
        point += 0.1 * tf.reduce_sum(tf.square(tf.reduce_sum(c, axis=1)))
        dx = c[..., 1:] - c[..., :-1]
        dy = c[..., 1:, :] - c[..., :-1, :]
        smooth = 0.25 * (tf.reduce_sum(dx * dx) + tf.reduce_sum(dy * dy))
        return point + smooth + quartic * tf.reduce_sum(c**4)

    return energy


def _estimator(energy, **overrides):
    kwargs = dict(
        cost_fn=energy,
        bcs=[],
        field_shape=SHAPE,
        precision="double",
        basis_vertical="molho",
        V_s=tf.constant([1.0, 1.0], tf.float64),
        idx_thk=0,
        cg_iters=[2, 5, 10, 60],
        cg_tol=1e-12,
        hvp_mode="banded",
        probe_mode="autodiff",
        preconditioner="barotropic_multigrid",
        preconditioner_options={"coarse_size": 4},
        damping=1e-12,
        # These tests characterise a single Newton step with a fresh Hessian;
        # the production defaults (two steps, adaptive reuse) are overridden.
        newton_steps=1,
        operator_update_freq=1,
        record_path=None,
        verbose=False,
    )
    kwargs.update(overrides)
    return ErrorEstimator(**kwargs)


@pytest.mark.parametrize(
    "preconditioner", ["barotropic_multigrid", "block_jacobi", "none"]
)
def test_newton_estimate_error_is_quadratic_in_true_error(preconditioner):
    rng = np.random.default_rng(21)
    target = tf.constant(rng.normal(size=(1, 4, 5, 7)), tf.float64)
    energy = _make_energy(target)
    inputs = tf.ones((1, 5, 7, 1), tf.float64)
    direction = rng.normal(size=(1, 4, 5, 7))
    direction /= np.linalg.norm(direction)

    est = _estimator(
        energy,
        preconditioner=preconditioner,
        hvp_mode="banded" if preconditioner != "none" else "autodiff",
    )

    ratios = []
    for eps in (0.4, 0.2, 0.1):
        start = target.numpy() + eps * direction
        U0 = tf.constant(start[:, :2], tf.float64)
        V0 = tf.constant(start[:, 2:], tf.float64)
        scalars, *_ = est.estimate(U0, V0, inputs)
        assert scalars["k60_relres"] < 1e-8
        true_norm = eps
        est_norm = scalars["est_delta_l2"]
        ratios.append(abs(est_norm - true_norm) / true_norm)
    # Relative estimate error decays at least linearly with eps (quadratic absolute).
    assert ratios[1] < 0.6 * ratios[0]
    assert ratios[2] < 0.6 * ratios[1]


def test_truncation_levels_are_monotone_and_converge():
    rng = np.random.default_rng(5)
    target = tf.constant(rng.normal(size=(1, 4, 5, 7)), tf.float64)
    energy = _make_energy(target)
    inputs = tf.ones((1, 5, 7, 1), tf.float64)
    start = target.numpy() + 0.1 * rng.normal(size=(1, 4, 5, 7))
    est = _estimator(energy)
    scalars, *_ = est.estimate(
        tf.constant(start[:, :2], tf.float64),
        tf.constant(start[:, 2:], tf.float64),
        inputs,
    )
    ks = (2, 5, 10, 60)
    decrements = [scalars[f"k{k}_decrement"] for k in ks]
    assert all(b >= a - 1e-12 for a, b in zip(decrements, decrements[1:]))
    relres = [scalars[f"k{k}_relres"] for k in ks]
    assert relres[0] > relres[1] > relres[-1]
    assert scalars["k2_decrement"] <= scalars["k60_decrement"] + 1e-12
    # Multigrid makes even a few iterations accurate on this small grid.
    assert (
        abs(scalars["k10_delta_l2"] - scalars["k60_delta_l2"])
        < 1e-2 * scalars["k60_delta_l2"]
    )


def test_operator_update_freq_reuses_stencil():
    rng = np.random.default_rng(9)
    target = tf.constant(rng.normal(size=(1, 4, 5, 7)), tf.float64)
    energy = _make_energy(target)
    inputs = tf.ones((1, 5, 7, 1), tf.float64)
    est = _estimator(energy, operator_update_freq=2, cg_iters=[30])
    start = target.numpy() + 0.1 * rng.normal(size=(1, 4, 5, 7))
    U0 = tf.constant(start[:, :2], tf.float64)
    V0 = tf.constant(start[:, 2:], tf.float64)
    _, _, prepared_first = est.estimate(U0, V0, inputs)
    _, _, prepared_second = est.estimate(U0, V0, inputs)
    _, _, prepared_third = est.estimate(U0, V0, inputs)
    assert prepared_first and not prepared_second and prepared_third
