"""The optimizer hook: start estimate in ``minimize`` and periodic in-loop calls."""

import json
import types

import numpy as np
import tensorflow as tf

from igm.processes.iceflow.error_estimator import ErrorEstimator
from igm.processes.iceflow.mappings.identity import MappingIdentity
from igm.processes.iceflow.optimizers.adam import OptimizerAdam

SHAPE = (1, 1, 3, 4)
N = 2 * int(np.prod(SHAPE))


def test_start_estimate_and_periodic_hook(tmp_path):
    rng = np.random.default_rng(1)
    factors = rng.normal(size=(N, N))
    H = tf.constant(factors @ factors.T + N * np.eye(N), tf.float64)
    target = tf.constant(rng.normal(size=N), tf.float64)

    def energy(U, V, inputs):
        del inputs
        x = tf.concat([tf.reshape(U, [-1]), tf.reshape(V, [-1])], axis=0)
        e = x - target
        return 0.5 * tf.tensordot(e, tf.linalg.matvec(H, e), axes=1)

    n = int(np.prod(SHAPE))
    start = target.numpy() + rng.normal(size=N)
    mapping = MappingIdentity(
        [],
        tf.reshape(tf.constant(start[:n], tf.float64), SHAPE),
        tf.reshape(tf.constant(start[n:], tf.float64), SHAPE),
        precision="double",
    )
    state = types.SimpleNamespace(it=0)
    record_path = tmp_path / "error_estimate.jsonl"
    estimator = ErrorEstimator(
        cost_fn=energy,
        bcs=[],
        field_shape=SHAPE,
        precision="double",
        basis_vertical="lagrange",
        V_s=tf.constant([1.0], tf.float64),
        idx_thk=0,
        freq=2,
        estimate_at_start=True,
        cg_iters=[N],
        cg_tol=1e-14,
        hvp_mode="autodiff",
        preconditioner="none",
        damping=0.0,
        record_path=str(record_path),
        state=state,
        verbose=False,
    )

    optimizer = OptimizerAdam(
        cost_fn=energy,
        map=mapping,
        print_cost=False,
        precision="double",
        lr=1e-2,
        iter_max=6,
    )
    optimizer.attach_error_estimator(estimator)

    inputs = tf.ones((1, SHAPE[2], SHAPE[3], 1), tf.float64)
    costs = optimizer.minimize(inputs)
    assert int(costs.shape[0]) == 6

    # `minimize` makes the start estimate (iteration -1) on the initial iterate.
    records = [json.loads(line) for line in record_path.read_text().splitlines()]
    assert [r["iter"] for r in records] == [-1]

    # The in-loop hook, traced like an optimizer loop would trace it, fires
    # every `freq` parameter updates: after updates 2, 4 and 6.
    @tf.function
    def loop():
        U, V = mapping.get_UV(inputs)
        for iteration in tf.range(6):
            optimizer._estimate_error(iteration, U, V, inputs)

    loop()
    records = [json.loads(line) for line in record_path.read_text().splitlines()]
    assert [r["iter"] for r in records] == [-1, 1, 3, 5]
    for rec in records:
        assert np.isfinite(rec["est_delta_l2"])
        assert rec["k%d_relres" % N] < 1e-10
    # Adam moved the iterate towards the target before the in-loop estimates.
    assert records[-1]["est_delta_l2"] < records[0]["est_delta_l2"]
    # State publication of the latest estimate.
    assert state.err_est_iter == 5
    assert state.err_est_surf.shape == (SHAPE[2], SHAPE[3])
    assert np.isfinite(state.err_est_rmse_all)


def test_hook_is_inert_without_estimator():
    optimizer = OptimizerAdam(
        cost_fn=lambda U, V, inputs: tf.reduce_sum(U + V),
        map=MappingIdentity(
            [], tf.zeros(SHAPE, tf.float64), tf.zeros(SHAPE, tf.float64), "double"
        ),
        print_cost=False,
        precision="double",
        iter_max=1,
    )
    inputs = tf.ones((1, SHAPE[2], SHAPE[3], 1), tf.float64)

    @tf.function
    def loop():
        U, V = optimizer.map.get_UV(inputs)
        optimizer._estimate_error(tf.constant(0), U, V, inputs)

    graph = loop.get_concrete_function().graph
    assert not any(
        op.type in ("PyFunc", "EagerPyFunc", "StatelessIf", "If")
        for op in graph.get_operations()
    )
