import numpy as np
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from igm.processes.iceflow.unified.bcs.periodic_ns import PeriodicNS
from igm.processes.iceflow.unified.bcs.periodic_we import PeriodicWE
from igm.processes.iceflow.unified.bcs.dirichlet import DirichletBoundary
from igm.processes.iceflow.unified.mappings.identity import MappingIdentity
from igm.processes.iceflow.unified.operators import ADOperator
from igm.processes.iceflow.unified.operators.tridiag1d import (
    Tridiag1DADOperator,
    supports_tridiag1d,
)
from igm.processes.iceflow.unified.operators.tridiag1d_analytic import (
    Tridiag1DAnalyticOperator,
)
from igm.processes.iceflow.unified import utils as unified_utils
from igm.processes.iceflow.energy.utils import get_energy_components
from igm.processes.iceflow.horizontal import Q1Discr
from igm.processes.iceflow.vertical import SSADiscr
from igm.processes.iceflow.unified.optimizers.tridiag_newton import (
    OptimizerTridiagNewton,
)
from igm.processes.iceflow.unified.optimizers.interfaces.tridiag_newton import (
    InterfaceTridiagNewton,
)


def test_supports_tridiag1d():
    dtype = tf.float64
    shape_ok = (1, 1, 2, 5)
    mapping_ok = MappingIdentity(
        [DirichletBoundary(left=0.0), PeriodicNS()],
        tf.zeros(shape_ok, dtype),
        tf.zeros(shape_ok, dtype),
        precision="double",
    )
    assert supports_tridiag1d(mapping_ok)

    shape_2d = (1, 1, 5, 5)
    mapping_2d = MappingIdentity(
        [DirichletBoundary(left=0.0), PeriodicNS()],
        tf.zeros(shape_2d, dtype),
        tf.zeros(shape_2d, dtype),
        precision="double",
    )
    assert not supports_tridiag1d(mapping_2d)

    mapping_periodic_x = MappingIdentity(
        [PeriodicNS(), PeriodicWE()],
        tf.zeros(shape_ok, dtype),
        tf.zeros(shape_ok, dtype),
        precision="double",
    )
    assert not supports_tridiag1d(mapping_periodic_x)

    with pytest.raises(ValueError):
        Tridiag1DADOperator(
            lambda U, V, inputs: tf.reduce_sum(U * U + V * V),
            mapping_2d,
            precision="double",
        )


def test_tridiag1d_hvp_matches_exact_autodiff():
    dtype = tf.float64
    nx = 9
    shape = (1, 1, 2, nx)
    rng = np.random.default_rng(7)
    u0 = tf.constant(rng.normal(size=shape), dtype=dtype)
    v0 = tf.constant(rng.normal(size=shape), dtype=dtype)
    bcs = [DirichletBoundary(left=0.0), PeriodicNS()]
    mapping = MappingIdentity(bcs, u0, v0, precision="double")

    def local_quadratic(U, V, inputs):
        del inputs
        components = tf.concat([U, V], axis=1)
        reduce_axes = tuple(range(1, components.shape.rank))
        point_energy = 0.5 * tf.reduce_sum(components * components, axis=reduce_axes)
        point_energy += 0.1 * tf.reduce_sum(
            tf.square(tf.reduce_sum(components, axis=1)),
            axis=tuple(range(1, components.shape.rank - 1)),
        )
        dx = components[:, :, :, 1:] - components[:, :, :, :-1]
        dy = components[:, :, 1:, :] - components[:, :, :-1, :]
        batch_energy = point_energy + 0.25 * (
            tf.reduce_sum(dx * dx, axis=reduce_axes)
            + tf.reduce_sum(dy * dy, axis=reduce_axes)
        )
        return tf.reduce_mean(batch_energy)

    inputs = tf.zeros([1, 1, 1, 1, 1], dtype=dtype)
    damping = tf.constant(1.0e-10, dtype=dtype)

    exact = ADOperator(local_quadratic, mapping, precision="double")
    tridiag = Tridiag1DADOperator(
        local_quadratic,
        mapping,
        precision="double",
        probe_mode="fd",
    )
    tridiag.prepare(inputs, damping)

    vector = tf.constant(rng.normal(size=2 * np.prod(shape)), dtype=dtype)
    exact_hvp = exact.hvp(inputs, vector, damping)
    tridiag_hvp = tridiag.hvp(inputs, vector, damping)

    relative_error = tf.norm(exact_hvp - tridiag_hvp) / tf.norm(exact_hvp)
    assert float(relative_error.numpy()) < 1.0e-6


def test_tridiag1d_analytic_q1_ssa_hessian_matches_exact_autodiff():
    """Direct element assembly must reproduce the real viscosity+sliding
    Hessian, including Q1 quadrature, cell masking, periodic row tying, bed
    slope correction, and Dirichlet elimination."""
    dtype = tf.float64
    nx = 8
    shape = (1, 1, 2, nx)
    cfg = OmegaConf.create(
        {
            "processes": {
                "iceflow": {
                    "method": "unified",
                    "physics": {
                        "energy_components": ["viscosity", "gravity", "sliding"],
                        "ice_density": 910.0,
                        "water_density": 1028.0,
                        "gravity_cst": 9.81,
                        "thr_ice_thk": 0.1,
                        "max_sr": 1.0e20,
                        "force_negative_gravitational_energy": False,
                        "viscosity": {
                            "exponent": 3.0,
                            "regularization": 1.0e-5,
                        },
                        "sliding": {
                            "law": "weertman",
                            "regularization": 1.0,
                            "exponent": 3.0,
                            "u_ref": 1.0,
                            "rho_ratio": 1028.0 / 910.0,
                            "use_mask_gr": True,
                        },
                    },
                    "numerics": {
                        "precision": "double",
                        "basis_horizontal": "q1",
                        "basis_vertical": "ssa",
                        "Nz": 1,
                    },
                    "unified": {
                        "inputs": ["thk", "usurf", "arrhenius", "tau_ref", "dX"]
                    },
                }
            }
        }
    )
    # get_sliding_params_args computes rho_ratio from the density fields; the
    # explicit value above simply documents the intended test geometry.
    rng = np.random.default_rng(29)
    x = np.linspace(0.0, 1.0, nx)[None, None, :]
    thickness = 300.0 + 25.0 * rng.random((1, 2, nx))
    surface = thickness - 150.0 - 40.0 * x
    arrhenius = 60.0 + 20.0 * rng.random((1, 2, nx))
    tau_ref = 0.03 + 0.01 * rng.random((1, 2, nx))
    spacing = np.full((1, 2, nx), 1800.0)
    inputs = tf.constant(
        np.stack([thickness, surface, arrhenius, tau_ref, spacing], axis=-1),
        dtype,
    )

    u0 = tf.constant(20.0 + 100.0 * rng.random(shape), dtype)
    v0 = tf.constant(-10.0 + 20.0 * rng.random(shape), dtype)
    mapping = MappingIdentity(
        [DirichletBoundary(left=0.0), PeriodicNS()],
        u0,
        v0,
        precision="double",
    )
    state = type("State", (), {})()
    state.iceflow = type("Iceflow", (), {})()
    state.iceflow.discr_h = Q1Discr(cfg)
    state.iceflow.discr_v = SSADiscr(cfg)
    cost_fn = unified_utils.get_cost_fn(cfg, state)
    exact = ADOperator(cost_fn, mapping, precision="double")
    components = get_energy_components(cfg)
    analytic = Tridiag1DAnalyticOperator(
        cost_fn,
        mapping,
        cfg,
        components,
        precision="double",
    )
    damping = tf.constant(0.0, dtype)
    analytic.prepare(inputs, damping)
    vector = tf.constant(rng.normal(size=2 * np.prod(shape)), dtype)
    exact_hvp = exact.hvp(inputs, vector, damping)
    analytic_hvp = analytic.hvp(inputs, vector, damping)

    relative_error = tf.norm(exact_hvp - analytic_hvp) / tf.norm(exact_hvp)
    # Nested TensorFlow/XLA reductions introduce a small, uniform reduction
    # rounding difference even in float64; the hand-derived curvature itself
    # agrees much more closely than the old finite-difference probe.
    assert float(relative_error.numpy()) < 1.0e-7


def test_scalar_q1_ssa_system_matches_full_igm_energy_and_bands():
    """The flowline specialization must preserve every configured energy."""
    dtype = tf.float64
    nx = 9
    shape = (1, 1, 2, nx)
    cfg = OmegaConf.create(
        {
            "processes": {
                "iceflow": {
                    "method": "unified",
                    "physics": {
                        "energy_components": [
                            "viscosity",
                            "gravity",
                            "sliding",
                            "floating",
                        ],
                        "ice_density": 910.0,
                        "water_density": 1028.0,
                        "gravity_cst": 9.81,
                        "cf_eswn": ["E"],
                        "thr_ice_thk": 0.1,
                        "max_sr": 1.0e20,
                        "force_negative_gravitational_energy": False,
                        "viscosity": {
                            "exponent": 3.0,
                            "regularization": 1.0e-5,
                        },
                        "sliding": {
                            "law": "weertman",
                            "regularization": 1.0,
                            "exponent": 3.0,
                            "u_ref": 1.0,
                            "use_mask_gr": True,
                        },
                    },
                    "numerics": {
                        "precision": "double",
                        "basis_horizontal": "q1",
                        "basis_vertical": "ssa",
                        "Nz": 1,
                    },
                    "unified": {
                        "inputs": [
                            "thk",
                            "usurf",
                            "arrhenius",
                            "tau_ref",
                            "dX",
                            "water_level",
                        ]
                    },
                }
            }
        }
    )
    x = np.linspace(0.0, 1.0, nx)
    thickness_1d = np.array([320, 315, 300, 280, 250, 210, 160, 0, 0.0])
    bed_1d = 80.0 - 260.0 * x
    surface_1d = bed_1d + thickness_1d

    def duplicate(values):
        return np.broadcast_to(values[None, None, :], (1, 2, nx)).copy()

    inputs = tf.constant(
        np.stack(
            [
                duplicate(thickness_1d),
                duplicate(surface_1d),
                duplicate(np.linspace(65.0, 75.0, nx)),
                duplicate(np.full(nx, 0.03)),
                duplicate(np.full(nx, 1800.0)),
                duplicate(np.zeros(nx)),
            ],
            axis=-1,
        ),
        dtype,
    )
    u_values = tf.constant(np.linspace(0.0, 450.0, nx)[None, :], dtype=dtype)
    U0 = tf.concat(
        [u_values[:, None, None, :], tf.zeros_like(u_values)[:, None, None, :]],
        axis=2,
    )
    V0 = tf.zeros(shape, dtype)
    mapping = MappingIdentity(
        [DirichletBoundary(left=0.0), PeriodicNS()],
        U0,
        V0,
        precision="double",
    )
    state = type("State", (), {})()
    state.iceflow = type("Iceflow", (), {})()
    state.iceflow.discr_h = Q1Discr(cfg)
    state.iceflow.discr_v = SSADiscr(cfg)
    cost_fn = unified_utils.get_cost_fn(cfg, state)
    components = get_energy_components(cfg)
    analytic = Tridiag1DAnalyticOperator(
        cost_fn,
        mapping,
        cfg,
        components,
        precision="double",
    )
    damping = tf.constant(1.0e-16, dtype)
    exact_cost, exact_gradient = analytic.cost_grad_u_at(inputs, u_values)
    cost, gradient, lower, diagonal, upper = analytic.scalar_system_at(
        inputs, u_values, damping
    )
    # The scalar reduction changes nested Q1/XLA reduction order slightly.
    np.testing.assert_allclose(cost.numpy(), exact_cost.numpy(), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        gradient.numpy(), exact_gradient.numpy(), rtol=1e-8, atol=1e-10
    )

    theta = analytic.join_row0(u_values, tf.zeros_like(u_values))
    full = analytic.assemble_bands_at(inputs, theta, damping)
    np.testing.assert_allclose(
        lower.numpy(), full["west"][:, 0, 0].numpy(), rtol=2e-12, atol=2e-14
    )
    np.testing.assert_allclose(
        diagonal.numpy(), full["center"][:, 0, 0].numpy(), rtol=2e-12, atol=2e-14
    )
    np.testing.assert_allclose(
        upper.numpy(), full["east"][:, 0, 0].numpy(), rtol=2e-12, atol=2e-14
    )


def test_tridiag_newton_solves_coupled_quadratic_in_one_step():
    """A single Newton step should exactly solve a quadratic on a y-invariant grid.

    The Dirichlet-fixed x=0 neighbour couples into the dx penalty term, so the
    constrained minimizer over the live degrees of freedom isn't simply
    ``target`` -- the real invariant of an exact single-step Newton solve of a
    quadratic is that the *gradient* vanishes afterwards, checked here with an
    independent exact-autodiff operator (not the code under test).
    """
    dtype = tf.float64
    nx = 4
    rng = np.random.default_rng(3)

    def embed_row0(flat_half):
        row0 = tf.reshape(flat_half, (1, 1, 1, nx))
        return tf.concat([row0, row0], axis=2)

    initial = tf.constant(rng.normal(size=2 * nx), dtype=dtype)
    target = tf.constant(rng.normal(size=2 * nx), dtype=dtype)
    u_target = embed_row0(target[:nx])
    v_target = embed_row0(target[nx:])

    u0 = embed_row0(initial[:nx])
    v0 = embed_row0(initial[nx:])
    bcs = [DirichletBoundary(left=0.0), PeriodicNS()]
    mapping = MappingIdentity(bcs, u0, v0, precision="double")

    def quadratic_energy(U, V, inputs):
        del inputs
        # Same functional form validated against exact autodiff HVP in
        # test_tridiag1d_hvp_matches_exact_autodiff: point energy + coupling
        # between components + nearest-neighbour differences in x and y.
        components = tf.concat([U - u_target, V - v_target], axis=1)
        point_energy = 0.5 * tf.reduce_sum(components * components)
        point_energy += 0.1 * tf.reduce_sum(
            tf.square(tf.reduce_sum(components, axis=1))
        )
        dx = components[:, :, :, 1:] - components[:, :, :, :-1]
        dy = components[:, :, 1:, :] - components[:, :, :-1, :]
        return point_energy + 0.25 * (tf.reduce_sum(dx * dx) + tf.reduce_sum(dy * dy))

    optimizer = OptimizerTridiagNewton(
        cost_fn=quadratic_energy,
        map=mapping,
        print_cost=False,
        precision="double",
        iter_max=1,
        damping=0.0,
    )

    inputs = tf.zeros([1, 1, 1, 1, 1], dtype=dtype)
    initial_cost = float(quadratic_energy(*mapping.get_UV(inputs), inputs))
    optimizer.minimize(inputs)

    final_cost = float(quadratic_energy(*mapping.get_UV(inputs), inputs))

    exact = ADOperator(quadratic_energy, mapping, precision="double")
    theta_flat = mapping.flatten_theta(mapping.get_theta())
    _, exact_grad = exact.cost_grad_at(inputs, theta_flat)

    assert float(tf.norm(exact_grad).numpy()) < 1.0e-8
    assert final_cost < initial_cost
    assert final_cost < initial_cost


def test_tridiag_newton_operator_dispatch_rejects_2d_mapping():
    shape = (1, 1, 5, 5)
    mapping = MappingIdentity(
        [DirichletBoundary(left=0.0), PeriodicNS()],
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
                        "tridiag_newton": {
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

    with pytest.raises(ValueError, match="tridiag_newton"):
        InterfaceTridiagNewton._build_operator(cfg, cost, mapping)
