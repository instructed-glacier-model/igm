import numpy as np
import pytest
import tensorflow as tf

from igm.processes.iceflow.hessian.operators import GaussNewtonOperator, Operator
from igm.processes.iceflow.hessian.preconditioners import (
    IdentityPreconditioner,
    NystromPreconditioner,
)
from igm.processes.iceflow.mappings import MappingNetwork
from igm.processes.iceflow.optimizers.gauss_newton import (
    OptimizerGaussNewton,
)

from .vector_mapping import VectorMapping


@pytest.fixture
def linear_network_problem():
    old_policy = tf.keras.mixed_precision.global_policy().name
    tf.keras.mixed_precision.set_global_policy("float64")
    inputs_layer = tf.keras.Input(shape=(None, None, 1), dtype=tf.float64)
    outputs = tf.keras.layers.Conv2D(2, 1, use_bias=True, dtype=tf.float64)(
        inputs_layer
    )
    network = tf.keras.Model(inputs_layer, outputs)
    network.nb_inputs = 1
    mapping = MappingNetwork([], network, Nz=1, precision="double")
    first_patch = tf.reshape(
        tf.constant([1.0, 2.0, 3.0, 4.0], tf.float64), [1, 2, 2, 1]
    )
    # Distinct patches catch accidental first-patch-only objectives.
    inputs = tf.concat([first_patch, 2.0 * first_patch], axis=0)
    target_u = tf.fill([1, 1, 2, 2], tf.constant(2.0, tf.float64))
    target_v = tf.fill([1, 1, 2, 2], tf.constant(-1.0, tf.float64))

    def cost_fn(U, V, unused_inputs):
        del unused_inputs
        return 0.5 * tf.reduce_sum(tf.square(U - target_u) + tf.square(V - target_v))

    yield mapping, inputs, cost_fn
    tf.keras.mixed_precision.set_global_policy(old_policy)


def test_network_stateless_evaluation_matches_and_does_not_mutate(
    linear_network_problem,
):
    mapping, inputs, _ = linear_network_problem
    theta = mapping.get_theta()
    before = [value.numpy().copy() for value in theta]
    operator = GaussNewtonOperator(
        lambda U, V, inputs: tf.reduce_sum(U + V + 0.0 * inputs),
        mapping,
        precision="double",
        hu_mode="autodiff",
    )

    expected = mapping.get_UV(inputs)
    theta_flat = mapping.flatten_theta([tf.identity(value) for value in theta])
    actual = operator.velocities_at(inputs, theta_flat)

    for expected_component, actual_component in zip(expected, actual):
        np.testing.assert_allclose(actual_component, expected_component)
    for variable, snapshot in zip(mapping.get_theta(), before):
        np.testing.assert_array_equal(variable.numpy(), snapshot)


def test_network_stateless_evaluation_freezes_normalization_state():
    inputs_layer = tf.keras.Input(shape=(2, 2, 1), dtype=tf.float64)
    outputs = tf.keras.layers.Conv2D(2, 1, dtype=tf.float64)(inputs_layer)
    outputs = tf.keras.layers.BatchNormalization(dtype=tf.float64)(outputs)
    network = tf.keras.Model(inputs_layer, outputs)
    network.nb_inputs = 1
    mapping = MappingNetwork([], network, Nz=1, precision="double")
    operator = GaussNewtonOperator(
        lambda U, V, inputs: tf.reduce_sum(U + V + 0.0 * inputs),
        mapping,
        precision="double",
        hu_mode="autodiff",
    )
    inputs = tf.reshape(tf.range(8, dtype=tf.float64), [2, 2, 2, 1])
    trainable_before = [value.numpy().copy() for value in mapping.get_theta()]
    state_before = [
        value.numpy().copy()
        for value in [
            getattr(variable, "_value", variable)
            for variable in network.non_trainable_variables
        ]
    ]

    expected = mapping.get_UV(inputs)
    actual = operator.velocities_at(
        inputs,
        mapping.flatten_theta([tf.identity(value) for value in mapping.get_theta()]),
    )

    for expected_component, actual_component in zip(expected, actual):
        np.testing.assert_allclose(actual_component, expected_component)
    for variable, snapshot in zip(mapping.get_theta(), trainable_before):
        np.testing.assert_array_equal(variable.numpy(), snapshot)
    for variable, snapshot in zip(
        [
            getattr(variable, "_value", variable)
            for variable in network.non_trainable_variables
        ],
        state_before,
    ):
        np.testing.assert_array_equal(variable.numpy(), snapshot)


def test_gauss_newton_matches_exact_hessian_for_linear_network(
    linear_network_problem,
):
    mapping, inputs, cost_fn = linear_network_problem
    operator = GaussNewtonOperator(
        cost_fn, mapping, precision="double", hu_mode="autodiff"
    )
    zero = tf.constant(0.0, tf.float64)
    operator.prepare(inputs, zero)
    theta_flat = mapping.flatten_theta(mapping.get_theta())
    n = int(theta_flat.shape[0])
    columns = [
        operator.hvp(inputs, tf.one_hot(i, n, dtype=tf.float64), zero) for i in range(n)
    ]
    gauss_newton = tf.stack(columns, axis=1)

    with tf.GradientTape() as outer:
        outer.watch(theta_flat)
        with tf.GradientTape() as inner:
            inner.watch(theta_flat)
            exact_cost = operator.cost_at(inputs, theta_flat)
        exact_gradient = inner.gradient(exact_cost, theta_flat)
    exact_hessian = outer.jacobian(exact_gradient, theta_flat)

    np.testing.assert_allclose(gauss_newton, exact_hessian, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        gauss_newton,
        tf.transpose(gauss_newton),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.linalg.eigvalsh(gauss_newton.numpy()).min() >= -1e-11


def test_gauss_newton_optimizer_reduces_linear_network_quadratic(
    linear_network_problem,
):
    mapping, inputs, cost_fn = linear_network_problem
    operator = GaussNewtonOperator(
        cost_fn, mapping, precision="double", hu_mode="autodiff"
    )
    optimizer = OptimizerGaussNewton(
        cost_fn=cost_fn,
        map=mapping,
        operator=operator,
        preconditioner_obj=IdentityPreconditioner(),
        preconditioner="identity",
        precision="double",
        print_cost=False,
        iter_max=3,
        cg_max_iter=10,
        forcing_eta_min=1e-10,
        forcing_eta_max=1e-8,
        damping_rel_init=1e-4,
    )
    initial = float(cost_fn(*mapping.get_UV(inputs), inputs))
    history = optimizer.minimize(inputs).numpy()
    final = float(cost_fn(*mapping.get_UV(inputs), inputs))

    assert np.all(np.diff(history) <= 0.0)
    assert final < initial * 1e-10
    assert 0 < int(optimizer.last_cg_iterations.numpy()) <= 10
    assert np.isfinite(float(optimizer.last_lm_ratio.numpy()))
    for name, count in optimizer.tracing_counts().items():
        assert count <= 1, (name, count)


class _DenseOperator:
    def __init__(self, matrix):
        self.matrix = tf.convert_to_tensor(matrix)

    def hvp(self, inputs, vector, damping):
        del inputs
        return tf.linalg.matvec(self.matrix, vector) + damping * vector


def test_nystrom_update_is_stable_and_damping_does_not_resketch():
    mapping = VectorMapping(6, dtype=tf.float64)
    matrix = tf.linalg.diag(tf.constant([9.0, 7.0, 5.0, 3.0, 2.0, 1.0], tf.float64))
    preconditioner = NystromPreconditioner(
        mapping, rank=4, minimum_rank=1, precision="double", seed=3
    )
    preconditioner.set_operator(_DenseOperator(matrix))
    inputs = tf.zeros([1, 1, 1, 1], tf.float64)
    preconditioner.update(inputs, tf.constant(0.1, tf.float64))
    refresh_index = preconditioner._refresh_index
    eigenvectors = preconditioner._eigenvectors.numpy().copy()

    preconditioner.set_damping(tf.constant(0.2, tf.float64))
    result = preconditioner.apply(tf.ones([6], tf.float64))

    assert preconditioner._refresh_index == refresh_index
    np.testing.assert_array_equal(preconditioner._eigenvectors.numpy(), eigenvectors)
    np.testing.assert_allclose(
        eigenvectors.T @ eigenvectors,
        np.eye(eigenvectors.shape[1]),
        rtol=1e-11,
        atol=1e-11,
    )
    assert np.all(np.isfinite(result.numpy()))
    assert float(preconditioner.lambda_max) > 0.0


def test_nystrom_memory_cap_falls_back_to_identity():
    mapping = VectorMapping(100, dtype=tf.float32)
    preconditioner = NystromPreconditioner(
        mapping,
        rank=32,
        minimum_rank=8,
        sketch_memory_gb=1e-12,
    )
    assert not preconditioner.active
    assert preconditioner.effective_rank == 0
    preconditioner.set_operator(_DenseOperator(tf.eye(100, dtype=tf.float32)))
    with pytest.warns(RuntimeWarning, match="using identity"):
        preconditioner.update(
            tf.zeros([1, 1, 1, 1], tf.float32),
            tf.constant(0.0, tf.float32),
        )
    vector = tf.range(100, dtype=tf.float32)
    np.testing.assert_array_equal(preconditioner.apply(vector), vector)


def test_nystrom_memory_cap_reduces_rank_deterministically():
    mapping = VectorMapping(100, dtype=tf.float32)
    bytes_for_ten_vectors = 3 * 100 * 10 * tf.float32.size
    with pytest.warns(RuntimeWarning, match="rank reduced"):
        preconditioner = NystromPreconditioner(
            mapping,
            rank=32,
            minimum_rank=8,
            sketch_memory_gb=bytes_for_ten_vectors / float(1024**3),
        )
    assert preconditioner.active
    assert preconditioner.effective_rank == 10
    assert preconditioner.estimated_sketch_bytes <= bytes_for_ten_vectors


class _InputScaledOperator(Operator):
    def cost_and_grad(self, inputs):
        raise NotImplementedError

    def cost_grad_at(self, inputs, theta_flat):
        raise NotImplementedError

    def hvp(self, inputs, vector, damping):
        scale = tf.reshape(inputs, [-1])[0]
        return (scale + damping) * vector


def test_gauss_newton_pcg_stops_on_negative_curvature():
    mapping = VectorMapping(3, dtype=tf.float64)
    optimizer = OptimizerGaussNewton(
        cost_fn=lambda U, V, inputs: tf.reduce_sum(U + V + inputs * 0.0),
        map=mapping,
        operator=_InputScaledOperator(),
        preconditioner_obj=IdentityPreconditioner(),
        preconditioner="identity",
        precision="double",
        print_cost=False,
        iter_max=1,
        cg_max_iter=3,
    )
    direction, iterations, _, _, negative = optimizer._solve(
        tf.fill([1, 1, 1, 1], tf.constant(-1.0, tf.float64)),
        tf.ones([3], tf.float64),
        tf.constant(0.0, tf.float64),
        tf.constant(1e-6, tf.float64),
    )
    np.testing.assert_array_equal(direction, tf.zeros([3], tf.float64))
    assert int(iterations) == 1
    assert bool(negative)
