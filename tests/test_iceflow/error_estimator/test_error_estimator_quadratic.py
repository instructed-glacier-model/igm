"""One Newton step recovers the exact error of a quadratic energy."""

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.error_estimator import (
    ErrorEstimator,
    masked_median,
    masked_rmse,
)

SHAPE = (1, 1, 3, 4)  # (B, Nz, Ny, Nx): 12 nodes per component, 24 DOF
N = 2 * int(np.prod(SHAPE))


def _problem(seed=3):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(N, N))
    hessian = factors @ factors.T + N * np.eye(N)
    target = rng.normal(size=N)
    start = target + rng.normal(size=N)
    H = tf.constant(hessian, tf.float64)
    t = tf.constant(target, tf.float64)

    def energy(U, V, inputs):
        del inputs
        x = tf.concat([tf.reshape(U, [-1]), tf.reshape(V, [-1])], axis=0)
        e = x - t
        return 0.5 * tf.tensordot(e, tf.linalg.matvec(H, e), axes=1)

    return energy, target, start


def _split(flat):
    n = int(np.prod(SHAPE))
    U = tf.reshape(tf.constant(flat[:n], tf.float64), SHAPE)
    V = tf.reshape(tf.constant(flat[n:], tf.float64), SHAPE)
    return U, V


def _estimator(energy, reference=None, **overrides):
    kwargs = dict(
        cost_fn=energy,
        bcs=[],
        field_shape=SHAPE,
        precision="double",
        basis_vertical="lagrange",
        V_s=tf.constant([1.0], tf.float64),
        idx_thk=0,
        cg_iters=[2, N],
        cg_tol=1e-14,
        hvp_mode="autodiff",
        preconditioner="none",
        damping=0.0,
        newton_steps=1,
        operator_update_freq=1,
        record_path=None,
        verbose=False,
        reference=reference,
    )
    kwargs.update(overrides)
    return ErrorEstimator(**kwargs)


def test_full_newton_step_equals_exact_error():
    energy, target, start = _problem()
    inputs = tf.ones((1, SHAPE[2], SHAPE[3], 1), tf.float64)  # thk = 1 everywhere
    U0, V0 = _split(start)
    n = int(np.prod(SHAPE))
    u_ref, v_ref = target[:n].reshape(SHAPE[2:]), target[n:].reshape(SHAPE[2:])

    est = _estimator(energy, reference={"u_ref": u_ref, "v_ref": v_ref})
    scalars, field, refreshed = est.estimate(U0, V0, inputs)

    error = start - target
    np.testing.assert_allclose(
        scalars["est_delta_l2"], np.linalg.norm(error), rtol=1e-9
    )
    assert scalars[f"k{N}_relres"] < 1e-10
    # Newton decrement equals the energy gap E(u) - E(u*) = 1/2 e^T H e.
    E0 = float(energy(U0, V0, inputs))
    np.testing.assert_allclose(scalars["est_decrement"], E0, rtol=1e-9)

    # Estimated statistics equal the true ones (denominator = u + delta = u*).
    for key in ("median_rel_all", "rmse_all"):
        np.testing.assert_allclose(
            scalars["est_" + key], scalars["true_" + key], rtol=1e-8
        )
    # The corrected velocity u + delta is the reference: zero remaining error.
    assert scalars["corr_median_rel_all"] < 1e-8
    assert scalars["corr_rmse_all"] < 1e-8

    # RMSE by hand over the (full-ice) grid.
    du = error[:n].reshape(SHAPE[2:])
    dv = error[n:].reshape(SHAPE[2:])
    np.testing.assert_allclose(
        scalars["est_rmse_all"], np.sqrt(np.mean(du**2 + dv**2)), rtol=1e-9
    )
    assert field.shape == (SHAPE[2], SHAPE[3])
    np.testing.assert_allclose(field.numpy(), np.sqrt(du**2 + dv**2), rtol=1e-6)
    assert refreshed and scalars["wall_s"] > 0.0


def test_truncated_estimate_is_monotone_lower_bound():
    energy, target, start = _problem(seed=7)
    inputs = tf.ones((1, SHAPE[2], SHAPE[3], 1), tf.float64)
    U0, V0 = _split(start)
    est = _estimator(energy, cg_iters=[1, 3, 6, N])
    scalars, *_ = est.estimate(U0, V0, inputs)
    decrements = [scalars[f"k{k}_decrement"] for k in (1, 3, 6, N)]
    assert all(b >= a - 1e-12 for a, b in zip(decrements, decrements[1:]))
    relres = [scalars[f"k{k}_relres"] for k in (1, 3, 6, N)]
    assert relres[-1] < relres[0]
    # Truncated corrections have a smaller A-norm than the exact Newton step.
    assert scalars["k1_decrement"] < scalars[f"k{N}_decrement"]


def test_ice_mask_restricts_statistics():
    energy, target, start = _problem(seed=11)
    thk = np.ones(SHAPE[2:])
    thk[0, :] = 0.0  # first row ice-free: nodes there have no active cell
    inputs = tf.constant(thk[None, :, :, None], tf.float64)
    U0, V0 = _split(start)
    est = _estimator(energy)
    scalars, field, *_ = est.estimate(U0, V0, inputs)
    assert scalars["est_cells_all"] == (SHAPE[2] - 1) * SHAPE[3]
    assert np.all(field.numpy()[0, :] == 0.0)


def test_masked_median_and_rmse():
    values = tf.constant(np.arange(1.0, 10.0).reshape(3, 3))
    mask = tf.constant(np.array([[1, 1, 1], [0, 0, 0], [1, 1, 0]], dtype=bool))
    assert float(masked_median(values, mask)) == 3.0  # values {1,2,3,7,8}
    np.testing.assert_allclose(
        float(masked_rmse(values, mask)), np.sqrt(np.mean([1, 4, 9, 49, 64]))
    )
    assert np.isnan(float(masked_median(values, tf.zeros_like(mask))))
