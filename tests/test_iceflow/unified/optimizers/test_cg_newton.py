import numpy as np
import pytest
import tensorflow as tf

from igm.processes.iceflow.unified.bcs.periodic_ns import PeriodicNS
from igm.processes.iceflow.unified.bcs.periodic_we import PeriodicWE
from igm.processes.iceflow.unified.mappings.identity import MappingIdentity
from igm.processes.iceflow.unified.optimizers.cg_newton import OptimizerCGNewton
from igm.processes.iceflow.unified.optimizers.energy_operator import (
    ADOperator,
    BandedADOperator,
)
from igm.processes.iceflow.unified.optimizers.preconditioner import (
    ComponentBlockJacobiPreconditioner,
    build_preconditioner,
)


@pytest.mark.parametrize("line_search_compile", [True, False])
@pytest.mark.parametrize("preconditioner", ["none", "block_jacobi"])
def test_cg_newton_solves_coupled_quadratic(
    line_search_compile,
    preconditioner,
):
    """A Newton step should solve a positive-definite quadratic with CG."""
    dtype = tf.float64
    shape = (1, 1, 1, 3)
    initial = tf.constant([-1.0, 0.5, 2.0, -2.0, 1.0, 0.0], dtype=dtype)
    target = tf.constant([1.5, -2.0, 0.75, 3.0, -1.0, 2.5], dtype=dtype)

    # Symmetric strict diagonal dominance makes this Hessian positive definite.
    hessian = tf.constant(
        [
            [4.0, 1.0, 0.0, 0.5, 0.0, 0.0],
            [1.0, 5.0, 1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 4.0, 0.0, 0.0, 0.5],
            [0.5, 0.0, 0.0, 3.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5, 4.0, 0.5],
            [0.0, 0.0, 0.5, 0.0, 0.5, 3.0],
        ],
        dtype=dtype,
    )

    mapping = MappingIdentity(
        [],
        tf.reshape(initial[:3], shape),
        tf.reshape(initial[3:], shape),
        precision="double",
    )

    def quadratic_energy(U, V, inputs):
        del inputs
        x = tf.concat([tf.reshape(U, [-1]), tf.reshape(V, [-1])], axis=0)
        error = x - target
        return 0.5 * tf.tensordot(error, tf.linalg.matvec(hessian, error), axes=1)

    optimizer = OptimizerCGNewton(
        cost_fn=quadratic_energy,
        map=mapping,
        print_cost=False,
        precision="double",
        line_search_method="armijo",
        line_search_compile=line_search_compile,
        iter_max=1,
        damping=0.0,
        cg_max_iter=6,
        cg_tol=1.0e-12,
        warm_start=False,
        preconditioner=preconditioner,
    )

    inputs = tf.zeros([1, 1, 1, 1, 1], dtype=dtype)
    initial_cost = float(quadratic_energy(*mapping.get_UV(inputs), inputs))
    optimizer.minimize(inputs)

    solution = mapping.flatten_theta(mapping.get_theta())
    final_cost = float(quadratic_energy(*mapping.get_UV(inputs), inputs))

    np.testing.assert_allclose(
        solution.numpy(), target.numpy(), rtol=1.0e-9, atol=1.0e-9
    )
    assert final_cost < 1.0e-18
    assert final_cost < initial_cost
    assert 1 < int(optimizer.last_cg_iterations.numpy()) <= 6
    assert float(optimizer.last_cg_relative_residual.numpy()) < 1.0e-10


@pytest.mark.parametrize("nz", [1, 2], ids=["ssa", "molho"])
def test_banded_hvp_and_block_jacobi_match_periodic_quadratic(nz):
    dtype = tf.float64
    shape = (1, nz, 5, 5)
    rng = np.random.default_rng(12)
    u0 = tf.constant(rng.normal(size=shape), dtype=dtype)
    v0 = tf.constant(rng.normal(size=shape), dtype=dtype)
    mapping = MappingIdentity(
        [PeriodicNS(), PeriodicWE()], u0, v0, precision="double"
    )

    def local_quadratic(U, V, inputs):
        del inputs
        components = tf.concat([U, V], axis=1)
        point_energy = 0.5 * tf.reduce_sum(components * components)
        point_energy += 0.1 * tf.reduce_sum(tf.square(tf.reduce_sum(components, axis=1)))
        dx = components[:, :, :, 1:] - components[:, :, :, :-1]
        dy = components[:, :, 1:, :] - components[:, :, :-1, :]
        return point_energy + 0.25 * (
            tf.reduce_sum(dx * dx) + tf.reduce_sum(dy * dy)
        )

    inputs = tf.zeros([1, 1, 1, 1, 1], dtype=dtype)
    damping = tf.constant(1.0e-15, dtype=dtype)
    exact = ADOperator(local_quadratic, mapping, precision="double")
    banded = BandedADOperator(
        local_quadratic,
        mapping,
        precision="double",
        probe_mode="fd",
        verify_stencil=False,
    )
    banded.prepare(inputs, damping)

    vector = tf.constant(rng.normal(size=2 * np.prod(shape)), dtype=dtype)
    exact_hvp = exact.hvp(inputs, vector, damping)
    banded_hvp = banded.hvp(inputs, vector, damping)
    relative_error = tf.norm(exact_hvp - banded_hvp) / tf.norm(exact_hvp)

    assert float(relative_error.numpy()) < 1.0e-8

    preconditioner = build_preconditioner(
        "block_jacobi", mapping, precision="double"
    )
    assert isinstance(preconditioner, ComponentBlockJacobiPreconditioner)
    preconditioner.set_operator(banded)
    preconditioner.update(inputs, damping)
    preconditioned = preconditioner.apply(vector)
    assert bool(tf.reduce_all(tf.math.is_finite(preconditioned)))
    assert preconditioned.shape == vector.shape
    assert float(tf.tensordot(vector, preconditioned, axes=1)) > 0.0

    exact_preconditioner = build_preconditioner(
        "block_jacobi", mapping, precision="double"
    )
    exact_preconditioner.set_operator(exact)
    exact_preconditioner.update(inputs, damping)
    np.testing.assert_allclose(
        exact_preconditioner.inverse_center.numpy(),
        preconditioner.inverse_center.numpy(),
        rtol=1.0e-7,
        atol=1.0e-9,
    )


@pytest.mark.parametrize(
    "removed_kind",
    ["exact_line_x", "ssa_line_x", "ssa_line_x_secant", "multigrid"],
)
def test_removed_preconditioners_fail_loudly(removed_kind):
    dtype = tf.float64
    shape = (1, 1, 2, 3)
    mapping = MappingIdentity(
        [],
        tf.zeros(shape, dtype=dtype),
        tf.zeros(shape, dtype=dtype),
        precision="double",
    )

    with pytest.raises(ValueError, match="none.*block_jacobi"):
        build_preconditioner(removed_kind, mapping, precision="double")
