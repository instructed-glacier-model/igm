#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import pytest

from igm.processes.iceflow.unified.optimizers.lbfgs import OptimizerLBFGS
from igm.processes.iceflow.unified.optimizers.lbfgs_bounds import OptimizerLBFGSBounds
from .vector_mapping import VectorMapping, BoundedVectorMapping, IdentitySampler

# Mark the whole module as slow
pytestmark = pytest.mark.slow


# ---------- ID helpers (pretty failure names) ----------
def id_n(n):
    return f"n={n}"


def id_mem(m):
    return f"mem={m}"


def id_alpha(a):
    return "alpha=0" if float(a) == 0.0 else f"alpha={a:.0e}"


def id_ls(s):
    return f"ls={s}"


# --- Utilities ---------------------------------------------------------------


def _dummy_inputs_5d(dtype=tf.float32):
    return tf.zeros([1, 1, 1, 1, 1], dtype=dtype)


# --- Tests -------------------------------------------------------------------


@pytest.mark.skip
@pytest.mark.parametrize("n", [10, 100], ids=id_n)
@pytest.mark.parametrize("memory", [1, 10], ids=id_mem)  # stress different histories
@pytest.mark.parametrize("alpha_min", [0.0, 1e-6], ids=id_alpha)
@pytest.mark.parametrize("ls_method", ["hager-zhang", "armijo", "wolfe"], ids=id_ls)
def test_rosenbrock_unconstrained_converges_parametric(
    request, n, memory, alpha_min, ls_method
):
    # HZ enforces strong Wolfe conditions; with memory=1 the L-BFGS directions
    # are too poor for the Rosenbrock valley — the curvature condition causes
    # zigzagging and the cost stagnates well above 1e-6.
    if ls_method == "hager-zhang" and memory == 1:
        request.applymarker(
            pytest.mark.xfail(
                reason="Hager-Zhang stagnates with memory=1 on Rosenbrock",
                strict=True,
            )
        )

    mapping = VectorMapping(n=n, dtype=tf.float32)

    # Difficult start: (-1.2, 1.0, -1.2, 1.0, ...)
    full = np.tile(np.array([-1.2, 1.0], dtype=np.float32), n)[:n]
    mapping.theta.assign(full)

    def cost_fn(U, V, inputs):
        x = mapping.theta
        x1, x2 = x[:-1], x[1:]
        return tf.reduce_sum(100.0 * (x2 - x1 * x1) ** 2 + (1.0 - x1) ** 2)

    opt = OptimizerLBFGS(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=False,
        print_cost_freq=1,
        line_search_method=ls_method,
        iter_max=2000,
        alpha_min=alpha_min,
        memory=memory,
        precision=tf.float32,
    )
    opt.sampler = IdentitySampler()

    costs = opt.minimize(_dummy_inputs_5d())
    final = float(costs[-1].numpy())
    assert (
        final <= 1e-6
    ), f"Did not converge: cost={final} (n={n}, mem={memory}, alpha_min={alpha_min}, ls={ls_method})"


@pytest.mark.skip
def test_rosenbrock_float64_support():
    """Covers your 64-bit reduction path with a 64-bit parameter vector."""
    n = 20
    mapping = VectorMapping(n=n, dtype=tf.float64)
    init = np.tile(np.array([-1.2, 1.0], dtype=np.float64), n)[:n]
    mapping.theta.assign(init)

    def cost_fn(U, V, inputs):
        x = mapping.theta
        x1, x2 = x[:-1], x[1:]
        return tf.reduce_sum(100.0 * (x2 - x1 * x1) ** 2 + (1.0 - x1) ** 2)

    opt = OptimizerLBFGS(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=False,
        print_cost_freq=1,
        line_search_method="hager-zhang",
        iter_max=1500,
        memory=10,
        precision=tf.float64,
    )
    opt.sampler = IdentitySampler()
    costs = opt.minimize(_dummy_inputs_5d(tf.float64))
    assert float(costs[-1].numpy()) <= 1e-6


# --- Convex quadratic: exact solution checks ---------------------------------


def _quadratic_cost(mapping, d_vec, b_vec):
    """
    f(x) = 0.5 * sum_i d_i x_i^2 - sum_i b_i x_i
    Minimizer (unconstrained): x* = b_i / d_i
    """
    d = tf.convert_to_tensor(d_vec, dtype=mapping.theta.dtype)
    b = tf.convert_to_tensor(b_vec, dtype=mapping.theta.dtype)

    def cost_fn(U, V, inputs):
        x = mapping.theta
        return 0.5 * tf.reduce_sum(d * x * x) - tf.reduce_sum(b * x)

    return cost_fn


def _quadratic_grad(x, d_vec, b_vec):
    # ∇f = d ⊙ x - b
    return d_vec * x - b_vec


@pytest.mark.parametrize("n", [8, 32], ids=id_n)
def test_quadratic_unconstrained_matches_closed_form(n):
    mapping = VectorMapping(n=n, dtype=tf.float64)

    # Ill-conditioned diagonal (spans 1..1e3) to exercise H0 scaling/τ tempering
    d = np.geomspace(1.0, 1e3, num=n).astype(np.float64)
    b = np.ones(n, dtype=np.float64) * 3.0
    x_star = b / d

    mapping.theta.assign(tf.zeros([n], tf.float64))

    cost_fn = _quadratic_cost(mapping, d, b)

    opt = OptimizerLBFGS(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=False,
        print_cost_freq=20,
        line_search_method="hager-zhang",
        iter_max=1000,
        memory=10,
        precision=tf.float64,
    )
    opt.sampler = IdentitySampler()
    opt.minimize(_dummy_inputs_5d(tf.float64))

    x_final = mapping.theta.numpy()
    rel_err = np.linalg.norm(x_final - x_star) / (np.linalg.norm(x_star) + 1e-12)
    assert rel_err < 1e-6, f"rel err {rel_err} too large"


# --- Box-constrained quadratic: interior and active-set ----------------------


@pytest.mark.parametrize("n", [16, 128], ids=id_n)
def test_quadratic_box_interior_solution(n):
    """Optimum is strictly inside bounds; should match unconstrained solution."""
    L, U = -5.0, 5.0
    mapping = BoundedVectorMapping(n=n, L=L, U=U, dtype=tf.float64)

    d = np.linspace(1.0, 10.0, num=n).astype(np.float64)
    b = np.linspace(0.5, 5.0, num=n).astype(
        np.float64
    )  # x* = b/d ∈ (0.05..5) → interior
    x_star = b / d

    mapping.theta.assign(tf.zeros([n], tf.float64))
    cost_fn = _quadratic_cost(mapping, d, b)

    opt = OptimizerLBFGS(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=False,
        print_cost_freq=50,
        line_search_method="hager-zhang",
        iter_max=700,
        memory=10,
        precision=tf.float64,
    )
    opt.sampler = IdentitySampler()
    opt.minimize(_dummy_inputs_5d(tf.float64))

    x_final = mapping.theta.numpy()
    assert np.all(x_final > L - 1e-6) and np.all(x_final < U + 1e-6)
    rel_err = np.linalg.norm(x_final - x_star) / (np.linalg.norm(x_star) + 1e-12)
    assert rel_err < 1e-6


def test_quadratic_box_boundary_active_set_and_KKT():
    """
    Force the solution to lie on the upper bound and check KKT-like conditions:
    - interior coords: grad ≈ 0
    - at lower bound:   grad >= 0
    - at upper bound:   grad <= 0
    """
    n = 50
    L, U = -0.5, 0.5
    mapping = BoundedVectorMapping(n=n, L=L, U=U, dtype=tf.float64)

    d = np.ones(n, dtype=np.float64) * 2.0
    b = np.ones(n, dtype=np.float64) * 2.0  # unconstrained x* = b/d = 1.0 → hits U=0.5
    x_star_proj = np.full(n, U, dtype=np.float64)

    mapping.theta.assign(tf.zeros([n], tf.float64))
    cost_fn = _quadratic_cost(mapping, d, b)

    opt = OptimizerLBFGSBounds(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=False,
        print_cost_freq=20,
        line_search_method="hager-zhang",
        iter_max=500,
        memory=10,
        precision=tf.float64,
    )
    opt.sampler = IdentitySampler()
    opt.minimize(_dummy_inputs_5d(tf.float64))

    x = mapping.theta.numpy()
    assert np.all(x <= U + 1e-6) and np.all(x >= L - 1e-6)
    assert np.max(np.abs(x - x_star_proj)) < 1e-5

    g = _quadratic_grad(x, d, b)  # ∇f = d⊙x - b
    tol = 1e-4
    atL = x <= (L + 1e-6)
    atU = x >= (U - 1e-6)
    free = (~atL) & (~atU)
    assert np.all(g[free] == pytest.approx(0.0, abs=1e-4))
    assert np.all(g[atL] >= -tol)
    assert np.all(g[atU] <= tol)
