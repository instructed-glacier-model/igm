import re

import numpy as np
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from igm.processes.iceflow.unified.bcs.periodic_ns import PeriodicNS
from igm.processes.iceflow.unified.bcs.periodic_we import PeriodicWE
from igm.processes.iceflow.unified.bcs.dirichlet import DirichletBoundary
from igm.processes.iceflow.unified.bcs.frozen_bed import FrozenBed
from igm.processes.iceflow.unified.mappings.identity import MappingIdentity
from igm.processes.iceflow.unified.optimizers.cg_newton import OptimizerCGNewton
from igm.processes.iceflow.unified.optimizers.banded import (
    COMPONENT_CENTER_KEY,
)
from igm.processes.iceflow.unified.optimizers.energy_operator import (
    ADOperator,
    BandedADOperator,
    MOLHOBandedADOperator,
    SSABandedADOperator,
)
from igm.processes.iceflow.unified.optimizers.preconditioner import (
    ComponentBlockJacobiPreconditioner,
    SSABlockJacobiPreconditioner,
    build_preconditioner,
)
from igm.processes.iceflow.unified.optimizers.line_searches import LineSearches
from igm.processes.iceflow.unified.optimizers.interfaces.cg_newton import (
    InterfaceCGNewton,
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


@pytest.mark.parametrize("periodic", [False, True], ids=["local", "periodic"])
@pytest.mark.parametrize("nz", [1, 2], ids=["ssa", "molho"])
def test_banded_hvp_and_block_jacobi_match_quadratic(nz, periodic):
    dtype = tf.float64
    shape = (1, nz, 5, 5)
    rng = np.random.default_rng(12)
    u0 = tf.constant(rng.normal(size=shape), dtype=dtype)
    v0 = tf.constant(rng.normal(size=shape), dtype=dtype)
    bcs = [PeriodicNS(), PeriodicWE()] if periodic else []
    mapping = MappingIdentity(bcs, u0, v0, precision="double")

    def local_quadratic(U, V, inputs):
        del inputs
        components = tf.concat([U, V], axis=1)
        point_energy = 0.5 * tf.reduce_sum(components * components)
        point_energy += 0.1 * tf.reduce_sum(
            tf.square(tf.reduce_sum(components, axis=1))
        )
        dx = components[:, :, :, 1:] - components[:, :, :, :-1]
        dy = components[:, :, 1:, :] - components[:, :, :-1, :]
        return point_energy + 0.25 * (
            tf.reduce_sum(dx * dx) + tf.reduce_sum(dy * dy)
        )

    inputs = tf.zeros([1, 1, 1, 1, 1], dtype=dtype)
    damping = tf.constant(1.0e-15, dtype=dtype)
    exact = ADOperator(local_quadratic, mapping, precision="double")
    operator_cls = (
        SSABandedADOperator if nz == 1 and not periodic else BandedADOperator
    )
    banded = operator_cls(
        local_quadratic,
        mapping,
        precision="double",
        probe_mode="fd",
        verify_stencil=False,
    )
    banded.prepare(inputs, damping)
    assembled = banded.assemble_bands(inputs, damping)
    if nz == 1 and not periodic:
        assert set(assembled) == {
            ("u", "u"),
            ("u", "v"),
            ("v", "u"),
            ("v", "v"),
        }
    else:
        assert set(assembled) == {COMPONENT_CENTER_KEY}
        assert assembled[COMPONENT_CENTER_KEY].shape == (
            1,
            2 * nz,
            2 * nz,
            5,
            5,
        )

    vector = tf.constant(rng.normal(size=2 * np.prod(shape)), dtype=dtype)
    exact_hvp = exact.hvp(inputs, vector, damping)
    banded_hvp = banded.hvp(inputs, vector, damping)
    relative_error = tf.norm(exact_hvp - banded_hvp) / tf.norm(exact_hvp)

    assert float(relative_error.numpy()) < 1.0e-8

    preconditioner = build_preconditioner(
        "block_jacobi",
        mapping,
        precision="double",
        layout=banded.preconditioner_layout,
    )
    if nz == 1 and not periodic:
        assert isinstance(preconditioner, SSABlockJacobiPreconditioner)
    else:
        assert isinstance(preconditioner, ComponentBlockJacobiPreconditioner)
    preconditioner.set_operator(banded)
    preconditioner.update(inputs, damping)
    if nz == 1 and not periodic:
        assert preconditioner.inverse_center.shape == (1, 2, 2, 5, 5)
        assert not hasattr(preconditioner, "_fine_vars")
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
        exact_preconditioner.apply(vector).numpy(),
        preconditioner.apply(vector).numpy(),
        rtol=1.0e-7,
        atol=1.0e-9,
    )


@pytest.mark.parametrize(
    ("nz", "periodic", "expected_type"),
    [
        (1, False, SSABandedADOperator),
        (1, True, BandedADOperator),
        (2, False, BandedADOperator),
    ],
)
def test_banded_operator_dispatch(nz, periodic, expected_type):
    shape = (1, nz, 3, 3)
    bcs = [PeriodicNS(), PeriodicWE()] if periodic else []
    mapping = MappingIdentity(
        bcs,
        tf.zeros(shape, tf.float64),
        tf.zeros(shape, tf.float64),
        precision="double",
    )
    cfg = OmegaConf.create(
        {
            "processes": {
                "iceflow": {
                    "numerics": {"precision": "double"},
                    "unified": {
                        "cg_newton": {
                            "hvp_mode": "banded",
                            "hvp_verify": False,
                            "probe_mode": "fd",
                        }
                    },
                }
            }
        }
    )

    def cost(U, V, inputs):
        del inputs
        return tf.reduce_sum(U * U + V * V)

    operator = InterfaceCGNewton._build_operator(cfg, cost, mapping)
    assert isinstance(operator, expected_type)

    optimizer = OptimizerCGNewton(
        cost_fn=cost,
        map=mapping,
        operator=operator,
        preconditioner="block_jacobi",
        precision="double",
        print_cost=False,
    )
    expected_preconditioner = (
        SSABlockJacobiPreconditioner
        if expected_type is SSABandedADOperator
        else ComponentBlockJacobiPreconditioner
    )
    assert isinstance(optimizer.preconditioner, expected_preconditioner)


def test_molho_banded_operator_dispatch():
    shape = (1, 2, 3, 5)
    mapping = MappingIdentity(
        [],
        tf.zeros(shape, tf.float64),
        tf.zeros(shape, tf.float64),
        precision="double",
    )
    cfg = OmegaConf.create(
        {
            "processes": {
                "iceflow": {
                    "numerics": {
                        "precision": "double",
                        "basis_vertical": "molho",
                    },
                    "unified": {
                        "cg_newton": {
                            "hvp_mode": "banded",
                            "hvp_verify": False,
                            "probe_mode": "autodiff",
                        }
                    },
                }
            }
        }
    )

    def cost(U, V, inputs):
        del inputs
        return tf.reduce_sum(U * U + V * V)

    assert isinstance(
        InterfaceCGNewton._build_operator(cfg, cost, mapping),
        MOLHOBandedADOperator,
    )


def test_two_component_block_jacobi_uses_elementwise_inverse(monkeypatch):
    dtype = tf.float64
    shape = (1, 1, 2, 2)
    mapping = MappingIdentity(
        [],
        tf.zeros(shape, dtype=dtype),
        tf.zeros(shape, dtype=dtype),
        precision="double",
    )
    center = np.array(
        [
            [
                [[4.0, 5.0], [6.0, 7.0]],
                [[1.0, 0.5], [0.25, 0.75]],
            ],
            [
                [[0.5, 1.5], [0.75, 0.25]],
                [[3.0, 4.0], [5.0, 6.0]],
            ],
        ],
        dtype=np.float64,
    )[np.newaxis]
    bands = np.zeros((9, 1, 2, 2, 2, 2), dtype=np.float64)
    bands[0] = center

    class CenterOperator:
        def hvp(self, inputs, vector, damping):
            del inputs, damping
            return vector

        def assemble_bands(self, inputs, damping):
            del inputs, damping
            return {"component_blocks": tf.constant(bands)}

    def fail_eigh(*args, **kwargs):
        del args, kwargs
        raise AssertionError("SSA block inversion must not call tf.linalg.eigh")

    preconditioner = build_preconditioner(
        "block_jacobi", mapping, precision="double"
    )
    preconditioner.set_operator(CenterOperator())
    monkeypatch.setattr(tf.linalg, "eigh", fail_eigh)
    preconditioner.update(
        tf.zeros([1, 1, 1, 1, 1], dtype=dtype),
        tf.constant(0.0, dtype=dtype),
    )

    matrices = np.moveaxis(center[0], (0, 1), (-2, -1))
    expected = np.moveaxis(np.linalg.inv(matrices), (-2, -1), (0, 1))[np.newaxis]
    np.testing.assert_allclose(
        preconditioner.inverse_center.numpy(),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_non_armijo_line_search_remains_compiled():
    dtype = tf.float64
    shape = (1, 1, 1, 1)
    mapping = MappingIdentity(
        [],
        tf.zeros(shape, dtype=dtype),
        tf.zeros(shape, dtype=dtype),
        precision="double",
    )

    def quadratic_energy(U, V, inputs):
        del inputs
        return tf.reduce_sum(tf.square(U) + tf.square(V))

    line_search_name = next(name for name in LineSearches if name != "armijo")
    optimizer = OptimizerCGNewton(
        cost_fn=quadratic_energy,
        map=mapping,
        print_cost=False,
        precision="double",
        line_search_method=line_search_name,
        line_search_compile=False,
        preconditioner="none",
    )

    assert optimizer.line_search_compile


def test_timing_output_matches_convergence_parser(capsys):
    dtype = tf.float64
    shape = (1, 1, 1, 1)
    mapping = MappingIdentity(
        [],
        tf.zeros(shape, dtype=dtype),
        tf.zeros(shape, dtype=dtype),
        precision="double",
    )

    def quadratic_energy(U, V, inputs):
        del inputs
        return 0.5 * tf.reduce_sum(tf.square(U - 1.0) + tf.square(V + 1.0))

    optimizer = OptimizerCGNewton(
        cost_fn=quadratic_energy,
        map=mapping,
        print_cost=False,
        print_timing=True,
        precision="double",
        iter_max=1,
        damping=0.0,
        cg_max_iter=2,
        cg_tol=1.0e-12,
        warm_start=False,
        preconditioner="none",
    )
    optimizer.minimize(tf.zeros([1, 1, 1, 1, 1], dtype=dtype))

    assert re.search(
        r"\[timing\] iter=\s*0 .*?cg=[\d.]+s\(\d+it.*?total=[\d.]+s",
        capsys.readouterr().out,
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
