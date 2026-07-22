import pytest
import tensorflow as tf

from igm.processes.iceflow.unified.optimizers.line_searches import (
    LineSearchArmijo,
    LineSearchHagerZhang,
    LineSearchWolfe,
    ValueAndGradient,
)


DTYPE = tf.float64
W = tf.zeros([8], dtype=DTYPE)
P = tf.ones([8], dtype=DTYPE)


def quadratic(target, counter):
    target = tf.constant(target, dtype=DTYPE)

    def evaluate(alpha):
        counter.assign_add(1)
        error = alpha - target
        return ValueAndGradient(alpha, 0.5 * error * error, error)

    val_0 = ValueAndGradient(
        x=tf.constant(0.0, dtype=DTYPE),
        f=0.5 * target * target,
        df=-target,
    )
    return evaluate, val_0


@pytest.mark.parametrize(
    "line_search,expected_evals",
    [
        (LineSearchArmijo(), 1),
        (LineSearchWolfe(), 1),
        (LineSearchHagerZhang(), 2),
    ],
)
def test_full_step_avoids_origin_evaluation(line_search, expected_evals):
    counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    evaluate, val_0 = quadratic(1.0, counter)

    alpha = line_search.search(W, P, evaluate, val_0=val_0)

    assert float(alpha.numpy()) == pytest.approx(1.0)
    assert int(counter.numpy()) == expected_evals


def test_armijo_uses_cost_only_trials_and_known_origin():
    grad_counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    value_counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    evaluate, val_0 = quadratic(0.05, grad_counter)
    target = tf.constant(0.05, dtype=DTYPE)

    def value(alpha):
        value_counter.assign_add(1)
        error = alpha - target
        return 0.5 * error * error

    line_search = LineSearchArmijo()
    alpha = line_search.search(
        W,
        P,
        evaluate,
        val_0=val_0,
        value_fn=value,
    )

    assert float(alpha.numpy()) == pytest.approx(0.0625)
    assert int(grad_counter.numpy()) == 0
    assert int(value_counter.numpy()) == 5


@pytest.mark.parametrize(
    "line_search",
    [LineSearchWolfe(), LineSearchHagerZhang()],
)
def test_derivative_searches_ignore_cost_only_callback(line_search):
    grad_counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    value_counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    evaluate, val_0 = quadratic(1.0, grad_counter)

    def value(alpha):
        value_counter.assign_add(1)
        return 0.5 * tf.square(alpha - tf.constant(1.0, dtype=DTYPE))

    alpha = line_search.search(
        W,
        P,
        evaluate,
        val_0=val_0,
        value_fn=value,
    )

    assert float(alpha.numpy()) == pytest.approx(1.0)
    assert int(grad_counter.numpy()) > 0
    assert int(value_counter.numpy()) == 0


def test_line_search_selects_supported_trial_callback():
    def value(alpha):
        return alpha

    assert LineSearchArmijo().select_value_fn(value) is value
    assert LineSearchWolfe().select_value_fn(value) is None
    assert LineSearchHagerZhang().select_value_fn(value) is None


def test_wolfe_caches_zoom_endpoint():
    counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    evaluate, val_0 = quadratic(0.05, counter)

    line_search = LineSearchWolfe()
    alpha = line_search.search(W, P, evaluate, val_0=val_0)
    value = evaluate(alpha)

    assert float(alpha.numpy()) == pytest.approx(0.0625)
    assert int(counter.numpy()) - 1 == 5
    assert bool(value.f <= val_0.f + 1.0e-4 * alpha * val_0.df)
    assert bool(tf.abs(value.df) <= -0.9 * val_0.df)


@pytest.mark.parametrize(
    "initial_step,expected_evals",
    [(1.0, 1), (0.1, 3)],
)
def test_hager_zhang_preserves_projected_boundary(initial_step, expected_evals):
    counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    target = tf.constant(5.0, dtype=DTYPE)
    bound = tf.constant(0.2, dtype=DTYPE)

    def evaluate(alpha):
        counter.assign_add(1)
        alpha_effective = tf.minimum(alpha, bound)
        error = alpha_effective - target
        return ValueAndGradient(
            alpha_effective,
            0.5 * error * error,
            error,
        )

    val_0 = ValueAndGradient(
        tf.constant(0.0, dtype=DTYPE),
        0.5 * target * target,
        -target,
    )
    line_search = LineSearchHagerZhang(step_size_initial=initial_step)

    result = line_search.search_result(W, P, evaluate, val_0=val_0)

    assert float(result.alpha.numpy()) == pytest.approx(0.2)
    assert bool(result.converged)
    assert not bool(result.failed)
    assert int(counter.numpy()) == expected_evals
    assert int(result.func_evals.numpy()) == expected_evals


@pytest.mark.parametrize(
    "line_search,compiled_method",
    [
        (LineSearchArmijo(), "search"),
        (LineSearchWolfe(), "search"),
        (LineSearchHagerZhang(), "search_result"),
    ],
)
def test_stable_callback_reuses_trace(line_search, compiled_method):
    counter = tf.Variable(0, dtype=tf.int32, trainable=False)
    evaluate, val_0 = quadratic(1.0, counter)

    def value(alpha):
        return 0.5 * tf.square(alpha - tf.constant(1.0, dtype=DTYPE))

    for _ in range(3):
        line_search.search(
            W,
            P,
            evaluate,
            val_0=val_0,
            value_fn=value,
        ).numpy()

    compiled = getattr(line_search, compiled_method)
    assert len(compiled._list_all_concrete_functions()) == 1
