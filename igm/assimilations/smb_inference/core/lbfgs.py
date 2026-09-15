"""L-BFGS built on IGM's own line searches.

This replaces ``tfp.optimizer.lbfgs_minimize``, which was the module's only
reason to depend on ``tensorflow_probability`` — a package IGM does not list in
its requirements and imports lazily where it uses it at all.

IGM's :class:`OptimizerLBFGS` is not reusable here: it optimises the network
parameters of the unified ice-flow problem and is written against the
``Mapping``/``Halt`` abstractions, taking ``cost_fn(U, V, inputs)`` rather than
a plain objective. The SMB inversion instead minimises a small profile vector
under ``f(x) -> (value, gradient)``. IGM's line-search package *is* reusable,
so the step-acceptance logic stays shared with the rest of IGM and only the
two-loop recursion lives here.

Those line searches are ``@tf.function``-decorated, which does not survive this
objective: the forward run is a gradient-checkpointed ``tf.while_loop``, and
TensorFlow refuses to nest its gradient inside a further ``tf.function``
("... is in list of internal_captures but not in internal_capture_to_output").
The search is therefore driven through its undecorated ``python_function`` so
IGM's step-acceptance logic runs eagerly, unchanged.

Only ``armijo`` is reliable on that eager path. IGM's ``wolfe`` and
``hager-zhang`` searches are written against autograph control flow that holds
only inside ``tf.function``; run eagerly they fail to converge (verified on a
Rosenbrock problem), so ``armijo`` is the default here.
"""

from collections import namedtuple

import tensorflow as tf

from igm.processes.iceflow.unified.optimizers.line_searches import (
    LineSearches,
    ValueAndGradient,
)

LBFGSResults = namedtuple(
    "LBFGSResults",
    [
        "position",
        "objective_value",
        "converged",
        "failed",
        "num_iterations",
        "num_objective_evaluations",
    ],
)


def _two_loop_direction(grad, s_list, y_list, gamma_min=1e-6, gamma_max=1e6):
    """Standard L-BFGS two-loop recursion returning the direction -H*grad."""
    if not s_list:
        return -grad

    q = tf.identity(grad)
    alphas = []
    rhos = []
    for s, y in zip(reversed(s_list), reversed(y_list)):
        rho = tf.math.divide_no_nan(
            tf.constant(1.0, dtype=grad.dtype), tf.tensordot(s, y, 1)
        )
        a = rho * tf.tensordot(s, q, 1)
        q = q - a * y
        alphas.append(a)
        rhos.append(rho)

    # Initial Hessian scaling from the newest pair, clipped as IGM's does.
    s_last, y_last = s_list[-1], y_list[-1]
    gamma = tf.math.divide_no_nan(
        tf.tensordot(s_last, y_last, 1), tf.tensordot(y_last, y_last, 1)
    )
    r = tf.clip_by_value(gamma, gamma_min, gamma_max) * q

    for (s, y), a, rho in zip(zip(s_list, y_list), reversed(alphas), reversed(rhos)):
        b = rho * tf.tensordot(y, r, 1)
        r = r + s * (a - b)

    return -r


def lbfgs_minimize(
    value_and_gradients_function,
    initial_position,
    max_iterations=50,
    num_correction_pairs=10,
    tolerance=1e-6,
    x_tolerance=0.0,
    f_relative_tolerance=0.0,
    line_search_method="armijo",
    callback=None,
):
    """Minimise *value_and_gradients_function* from *initial_position*.

    The objective is called repeatedly per iteration, at trial points the line
    search may reject, so per-iteration reporting belongs in *callback* rather
    than in the objective. *callback* receives ``(iteration, position, value)``
    once per accepted step. The accepted point is always the most recent
    evaluation, so a callback may reuse whatever the objective last computed.

    The returned fields mirror the subset of ``tfp``'s result object that this
    module used, so call sites read unchanged.
    """
    x0 = tf.convert_to_tensor(initial_position)
    dtype = x0.dtype

    line_search = LineSearches[line_search_method]()
    # Drive IGM's search eagerly; see the module docstring for why the
    # tf.function wrapper cannot be used with this objective.
    search = getattr(type(line_search).search, "python_function", None)

    state = {"x": x0, "p": tf.zeros_like(x0), "n": 0}

    def eval_fn(alpha):
        state["n"] += 1
        pos = state["x"] + tf.cast(alpha, dtype) * state["p"]
        fv, gv = value_and_gradients_function(pos)
        return ValueAndGradient(x=alpha, f=fv, df=tf.tensordot(gv, state["p"], 1))

    def evaluate(pos):
        state["n"] += 1
        return value_and_gradients_function(pos)

    x = x0
    f, g = evaluate(x)
    s_list, y_list = [], []
    converged = False
    failed = False
    iterations = 0

    for i in range(int(max_iterations)):
        iterations = i + 1

        if not bool(tf.math.is_finite(f)):
            failed = True
            break
        if float(tf.norm(g)) <= tolerance:
            converged = True
            break

        p = _two_loop_direction(g, s_list, y_list)
        # A non-descent direction means the curvature history is unusable;
        # reset to steepest descent rather than stepping the wrong way.
        if float(tf.tensordot(p, g, 1)) >= 0.0:
            s_list, y_list = [], []
            p = -g

        state["x"], state["p"] = x, p
        if search is not None:
            alpha = search(line_search, x, p, eval_fn)
        else:
            alpha = line_search.search(x, p, eval_fn)
        alpha = tf.cast(alpha, dtype)

        x_new = x + alpha * p
        f_new, g_new = evaluate(x_new)

        if not bool(tf.math.is_finite(f_new)):
            failed = True
            break

        # Keep only curvature pairs that preserve positive definiteness.
        s = x_new - x
        y = g_new - g
        if float(tf.tensordot(s, y, 1)) > 1e-12:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > int(num_correction_pairs):
                s_list.pop(0)
                y_list.pop(0)

        f_prev, x_prev = f, x
        x, f, g = x_new, f_new, g_new

        if callback is not None:
            callback(i, x, f)

        if x_tolerance > 0.0 and float(tf.norm(x - x_prev)) <= x_tolerance:
            converged = True
            break
        if f_relative_tolerance > 0.0:
            denom = max(abs(float(f_prev)), 1e-30)
            if abs(float(f_prev) - float(f)) / denom <= f_relative_tolerance:
                converged = True
                break

    return LBFGSResults(
        position=x,
        objective_value=f,
        converged=tf.constant(converged),
        failed=tf.constant(failed),
        num_iterations=tf.constant(iterations, dtype=tf.int32),
        num_objective_evaluations=tf.constant(state["n"], dtype=tf.int32),
    )
