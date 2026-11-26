#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

import collections
from typing import Callable, Optional

import tensorflow as tf

from .line_search import LineSearch, ValueAndGradient


LineSearchResult = collections.namedtuple(
    "LineSearchResult",
    ["alpha", "converged", "failed", "func_evals", "iterations", "left", "right"],
)

HagerZhangLineSearchResult = collections.namedtuple(
    "HagerZhangLineSearchResult",
    ["converged", "failed", "func_evals", "iterations", "left", "right"],
)

_IntermediateResult = collections.namedtuple(
    "_IntermediateResult",
    ["iteration", "stopped", "failed", "num_evals", "left", "right"],
)

_Secant2Result = collections.namedtuple(
    "_Secant2Result",
    ["active", "converged", "failed", "num_evals", "left", "right"],
)


def _is_finite(val: ValueAndGradient) -> tf.Tensor:
    return (
        tf.math.is_finite(val.x)
        & tf.math.is_finite(val.f)
        & tf.math.is_finite(val.df)
    )


def _bad_nan(val: ValueAndGradient) -> tf.Tensor:
    # Match TFP logic: if f is +/-inf, df is ignored because the decision is already known.
    bad_nan_df = tf.math.is_finite(val.f) & tf.math.is_nan(val.df)
    return tf.math.is_nan(val.f) | bad_nan_df | tf.math.is_nan(val.x)


def _is_negative_inf(x: tf.Tensor) -> tf.Tensor:
    return x <= tf.constant(float("-inf"), dtype=x.dtype)


def _is_rising(val: ValueAndGradient) -> tf.Tensor:
    # A suitable right endpoint has finite value and nonnegative slope.
    return tf.math.is_finite(val.f) & (val.df >= tf.zeros_like(val.df))


def _needs_bisect(val: ValueAndGradient, f_lim: tf.Tensor) -> tf.Tensor:
    # Either +inf, or a negative slope with function value still too high.
    return (val.f >= tf.constant(float("inf"), dtype=val.f.dtype)) | (
        (val.df < 0) & (val.f > f_lim)
    )


def _very_close(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    # Same test TFP uses.
    return tf.math.nextafter(x, y) >= y


def _secant(val_a: ValueAndGradient, val_b: ValueAndGradient) -> tf.Tensor:
    # Secant interpolation on the directional derivative.
    return tf.math.divide_no_nan(
        val_a.x * val_b.df - val_b.x * val_a.df,
        val_b.df - val_a.df,
    )


def _satisfies_wolfe(
    val_0: ValueAndGradient,
    val_c: ValueAndGradient,
    f_lim: tf.Tensor,
    sufficient_decrease_param: tf.Tensor,
    curvature_param: tf.Tensor,
) -> tf.Tensor:
    """
    Check exact Wolfe OR approximate Wolfe, closely following TFP.
    """

    # Exact Wolfe:
    #   f(c) <= f(0) + delta * c * f'(0)
    #   f'(c) >= sigma * f'(0)
    exact_wolfe_suff_dec = (
        sufficient_decrease_param * val_0.df
        >= tf.math.divide_no_nan(val_c.f - val_0.f, val_c.x)
    )
    wolfe_curvature = val_c.df >= curvature_param * val_0.df
    exact_wolfe = exact_wolfe_suff_dec & wolfe_curvature

    # Approximate Wolfe near a minimum:
    #   f(c) <= f_lim = f(0) + eps * |f(0)|
    #   f'(c) <= (2*delta - 1) * f'(0)
    #   f'(c) >= sigma * f'(0)
    approx_wolfe_applies = val_c.f <= f_lim
    approx_wolfe_suff_dec = (
        (2.0 * sufficient_decrease_param - 1.0) * val_0.df >= val_c.df
    )
    approx_wolfe = approx_wolfe_applies & approx_wolfe_suff_dec & wolfe_curvature

    return exact_wolfe | approx_wolfe


def _bisect(
    value_and_gradients_function: Callable[[tf.Tensor], ValueAndGradient],
    initial_args: _IntermediateResult,
    f_lim: tf.Tensor,
) -> _IntermediateResult:
    """
    Scalar version of TFP's internal bisection helper.
    """

    def _loop_cond(curr: _IntermediateResult) -> tf.Tensor:
        return tf.logical_not(curr.stopped)

    def _loop_body(curr: _IntermediateResult):
        mid = value_and_gradients_function((curr.left.x + curr.right.x) / 2.0)

        finished = _is_negative_inf(mid.f)
        failed = curr.failed | (
            tf.logical_not(finished)
            & tf.logical_not(curr.stopped)
            & (_bad_nan(mid) | tf.equal(mid.x, curr.left.x) | tf.equal(mid.x, curr.right.x))
        )

        to_update = tf.logical_not(curr.stopped | failed)
        update_left = (mid.df < 0) & (mid.f <= f_lim)

        left = tf.cond(
            to_update & (update_left | finished),
            lambda: mid,
            lambda: curr.left,
        )
        right = tf.cond(
            to_update & (tf.logical_not(update_left) | finished),
            lambda: mid,
            lambda: curr.right,
        )

        stopped = curr.stopped | failed | finished | _is_rising(right)

        return (
            _IntermediateResult(
                iteration=curr.iteration,
                stopped=stopped,
                failed=failed,
                num_evals=curr.num_evals + tf.constant(1, dtype=curr.num_evals.dtype),
                left=left,
                right=right,
            ),
        )

    return tf.while_loop(
        cond=_loop_cond,
        body=_loop_body,
        loop_vars=(initial_args,),
        parallel_iterations=1,
    )[0]


def _update(
    value_and_gradients_function: Callable[[tf.Tensor], ValueAndGradient],
    val_left: ValueAndGradient,
    val_right: ValueAndGradient,
    val_trial: ValueAndGradient,
    f_lim: tf.Tensor,
    active: tf.Tensor | None = None,
) -> _IntermediateResult:
    """
    Scalar version of TFP's interval update routine.
    """

    within_range = (val_left.x < val_trial.x) & (val_trial.x < val_right.x)
    if active is not None:
        within_range = within_range & active

    valid_left = (val_trial.df < 0) & (val_trial.f <= f_lim)
    needs_bisect = within_range & (val_trial.df < 0) & (val_trial.f > f_lim)

    left = tf.cond(within_range & valid_left, lambda: val_trial, lambda: val_left)
    right = tf.cond(
        within_range & tf.logical_not(valid_left),
        lambda: val_trial,
        lambda: val_right,
    )

    bisect_args = _IntermediateResult(
        iteration=tf.constant(0, dtype=tf.int32),
        stopped=tf.logical_not(needs_bisect),
        failed=tf.constant(False),
        num_evals=tf.constant(0, dtype=tf.int32),
        left=left,
        right=right,
    )
    return _bisect(value_and_gradients_function, bisect_args, f_lim)


def _bracket(
    value_and_gradients_function: Callable[[tf.Tensor], ValueAndGradient],
    search_interval: HagerZhangLineSearchResult,
    f_lim: tf.Tensor,
    max_iterations: tf.Tensor,
    expansion_param: tf.Tensor,
) -> _IntermediateResult:
    """
    Scalar version of TFP's bracketing phase.
    """

    already_stopped = search_interval.failed | search_interval.converged
    failed = _bad_nan(search_interval.right)
    finished = _is_negative_inf(search_interval.right.f)
    bracketed = _is_rising(search_interval.right)
    needs_bisect = _needs_bisect(search_interval.right, f_lim)

    initial_args = _IntermediateResult(
        iteration=search_interval.iterations,
        stopped=already_stopped | failed | finished | bracketed | needs_bisect,
        failed=search_interval.failed | failed,
        num_evals=search_interval.func_evals,
        left=search_interval.left,
        right=search_interval.right,
    )

    def _loop_cond(curr: _IntermediateResult) -> tf.Tensor:
        return (curr.iteration < max_iterations) & tf.logical_not(curr.stopped)

    def _loop_body(curr: _IntermediateResult):
        new_right = value_and_gradients_function(expansion_param * curr.right.x)

        # If the evaluation is clamped (e.g. a box-bounded eval_fn that caps the
        # step at alpha_max), the right endpoint cannot advance and expansion
        # would spin until max_iterations doing one full evaluation per round.
        # Collapse the interval onto the clamped point and stop; the degenerate
        # interval is treated as converged-at-the-boundary downstream.
        clamped = tf.logical_not(curr.stopped) & (new_right.x <= curr.right.x)

        left = tf.cond(
            curr.stopped,
            lambda: curr.left,
            lambda: tf.cond(clamped, lambda: new_right, lambda: curr.right),
        )
        right = tf.cond(curr.stopped, lambda: curr.right, lambda: new_right)

        failed = curr.failed | _bad_nan(right)
        finished = _is_negative_inf(right.f)
        bracketed = _is_rising(right)
        needs_bisect = _needs_bisect(right, f_lim)

        return (
            _IntermediateResult(
                iteration=curr.iteration + tf.cast(tf.logical_not(curr.stopped), tf.int32),
                stopped=curr.stopped | failed | finished | bracketed | needs_bisect | clamped,
                failed=failed,
                num_evals=curr.num_evals + tf.constant(1, dtype=curr.num_evals.dtype),
                left=left,
                right=right,
            ),
        )

    bracket_result = tf.while_loop(
        cond=_loop_cond,
        body=_loop_body,
        loop_vars=(initial_args,),
        parallel_iterations=1,
    )[0]

    finished = _is_negative_inf(bracket_result.right.f)
    bracketed = _is_rising(bracket_result.right)
    needs_bisect = _needs_bisect(bracket_result.right, f_lim)
    # A degenerate interval (left == right, e.g. after a clamped expansion) has
    # nothing left to bisect — running _bisect on it just burns an evaluation
    # and flags a spurious failure.
    degenerate = tf.equal(bracket_result.left.x, bracket_result.right.x)
    stopped = (
        already_stopped
        | bracket_result.failed
        | finished
        | bracketed
        | (degenerate & tf.logical_not(needs_bisect))
    )

    left = tf.cond(
        finished,
        lambda: bracket_result.right,
        lambda: tf.cond(
            needs_bisect,
            lambda: search_interval.left,
            lambda: bracket_result.left,
        ),
    )

    bisect_args = bracket_result._replace(stopped=stopped, left=left)
    return _bisect(value_and_gradients_function, bisect_args, f_lim)


def _secant2_inner_update(
    value_and_gradients_function: Callable[[tf.Tensor], ValueAndGradient],
    initial_args: _Secant2Result,
    val_0: ValueAndGradient,
    val_c: ValueAndGradient,
    f_lim: tf.Tensor,
    sufficient_decrease_param: tf.Tensor,
    curvature_param: tf.Tensor,
) -> _Secant2Result:
    new_failed = initial_args.active & tf.logical_not(_is_finite(val_c))
    active = initial_args.active & tf.logical_not(new_failed)
    failed = initial_args.failed | new_failed

    found_wolfe = active & _satisfies_wolfe(
        val_0, val_c, f_lim, sufficient_decrease_param, curvature_param
    )

    val_left = tf.cond(found_wolfe, lambda: val_c, lambda: initial_args.left)
    val_right = tf.cond(found_wolfe, lambda: val_c, lambda: initial_args.right)
    converged = initial_args.converged | found_wolfe
    active = active & tf.logical_not(found_wolfe)

    def _apply_update() -> _Secant2Result:
        update_result = _update(
            value_and_gradients_function,
            val_left,
            val_right,
            val_c,
            f_lim,
            active=active,
        )
        return _Secant2Result(
            active=tf.constant(False),
            converged=converged,
            failed=failed | update_result.failed,
            num_evals=initial_args.num_evals + update_result.num_evals,
            left=update_result.left,
            right=update_result.right,
        )

    def _default() -> _Secant2Result:
        return _Secant2Result(
            active=active,
            converged=converged,
            failed=failed,
            num_evals=initial_args.num_evals,
            left=val_left,
            right=val_right,
        )

    return tf.cond(active, _apply_update, _default)


def _secant2_inner(
    value_and_gradients_function: Callable[[tf.Tensor], ValueAndGradient],
    initial_args: _Secant2Result,
    val_0: ValueAndGradient,
    val_c: ValueAndGradient,
    f_lim: tf.Tensor,
    sufficient_decrease_param: tf.Tensor,
    curvature_param: tf.Tensor,
) -> _Secant2Result:
    update_result = _update(
        value_and_gradients_function,
        initial_args.left,
        initial_args.right,
        val_c,
        f_lim,
        active=initial_args.active,
    )

    active = initial_args.active & tf.logical_not(update_result.failed)
    failed = initial_args.failed | update_result.failed

    val_left = tf.cond(active, lambda: update_result.left, lambda: initial_args.left)
    val_right = tf.cond(active, lambda: update_result.right, lambda: initial_args.right)

    updated_left = active & tf.equal(val_left.x, val_c.x)
    updated_right = active & tf.equal(val_right.x, val_c.x)
    is_new = updated_left | updated_right

    next_c = tf.cond(
        updated_left,
        lambda: _secant(initial_args.left, val_left),
        lambda: val_c.x,
    )
    next_c = tf.cond(
        updated_right,
        lambda: _secant(initial_args.right, val_right),
        lambda: next_c,
    )

    in_range = (val_left.x <= next_c) & (next_c <= val_right.x)
    needs_extra_eval = in_range & is_new

    num_evals = initial_args.num_evals + update_result.num_evals
    num_evals = num_evals + tf.cast(needs_extra_eval, num_evals.dtype)

    next_args = _Secant2Result(
        active=active & in_range,
        converged=initial_args.converged,
        failed=failed,
        num_evals=num_evals,
        left=val_left,
        right=val_right,
    )

    def _apply_inner_update() -> _Secant2Result:
        next_val_c = tf.cond(
            needs_extra_eval,
            lambda: value_and_gradients_function(next_c),
            lambda: val_c,
        )
        return _secant2_inner_update(
            value_and_gradients_function,
            next_args,
            val_0,
            next_val_c,
            f_lim,
            sufficient_decrease_param,
            curvature_param,
        )

    return tf.cond(next_args.active, _apply_inner_update, lambda: next_args)


def _secant2(
    value_and_gradients_function: Callable[[tf.Tensor], ValueAndGradient],
    val_0: ValueAndGradient,
    search_interval: HagerZhangLineSearchResult,
    f_lim: tf.Tensor,
    sufficient_decrease_param: tf.Tensor,
    curvature_param: tf.Tensor,
) -> _Secant2Result:
    val_c = value_and_gradients_function(_secant(search_interval.left, search_interval.right))

    finished = _is_negative_inf(val_c.f)
    failed = tf.logical_not(search_interval.converged) & (
        search_interval.failed
        | (tf.logical_not(finished) & tf.logical_not(_is_finite(val_c)))
    )

    converged = search_interval.converged | finished | (
        tf.logical_not(failed)
        & _satisfies_wolfe(
            val_0, val_c, f_lim, sufficient_decrease_param, curvature_param
        )
    )

    new_converged = converged & tf.logical_not(search_interval.converged)
    val_left = tf.cond(new_converged, lambda: val_c, lambda: search_interval.left)
    val_right = tf.cond(new_converged, lambda: val_c, lambda: search_interval.right)

    initial_args = _Secant2Result(
        active=tf.logical_not(failed) & tf.logical_not(converged),
        converged=converged,
        failed=failed,
        num_evals=search_interval.func_evals + tf.constant(1, dtype=search_interval.func_evals.dtype),
        left=val_left,
        right=val_right,
    )

    return tf.cond(
        initial_args.active,
        lambda: _secant2_inner(
            value_and_gradients_function,
            initial_args,
            val_0,
            val_c,
            f_lim,
            sufficient_decrease_param,
            curvature_param,
        ),
        lambda: initial_args,
    )


def _line_search_inner_bisection(
    value_and_gradients_function: Callable[[tf.Tensor], ValueAndGradient],
    search_interval: HagerZhangLineSearchResult,
    active: tf.Tensor,
    f_lim: tf.Tensor,
) -> HagerZhangLineSearchResult:
    midpoint = (search_interval.left.x + search_interval.right.x) / 2.0
    val_mid = value_and_gradients_function(midpoint)
    is_valid_mid = _is_finite(val_mid)

    still_active = active & is_valid_mid
    new_failed = active & tf.logical_not(is_valid_mid)

    next_interval = search_interval._replace(
        failed=search_interval.failed | new_failed,
        func_evals=search_interval.func_evals + tf.constant(1, dtype=search_interval.func_evals.dtype),
    )

    def _apply_update() -> HagerZhangLineSearchResult:
        update_result = _update(
            value_and_gradients_function,
            next_interval.left,
            next_interval.right,
            val_mid,
            f_lim,
            active=still_active,
        )
        return HagerZhangLineSearchResult(
            converged=next_interval.converged,
            failed=next_interval.failed | update_result.failed,
            iterations=next_interval.iterations + update_result.iteration,
            func_evals=next_interval.func_evals + update_result.num_evals,
            left=update_result.left,
            right=update_result.right,
        )

    return tf.cond(still_active, _apply_update, lambda: next_interval)


def _line_search_after_bracketing(
    value_and_gradients_function: Callable[[tf.Tensor], ValueAndGradient],
    search_interval: HagerZhangLineSearchResult,
    val_0: ValueAndGradient,
    f_lim: tf.Tensor,
    max_iterations: tf.Tensor,
    sufficient_decrease_param: tf.Tensor,
    curvature_param: tf.Tensor,
    shrinkage_param: tf.Tensor,
) -> HagerZhangLineSearchResult:
    def _loop_cond(curr_interval: HagerZhangLineSearchResult) -> tf.Tensor:
        active = tf.logical_not(curr_interval.converged | curr_interval.failed)
        return (curr_interval.iterations < max_iterations) & active

    def _loop_body(curr_interval: HagerZhangLineSearchResult):
        active = tf.logical_not(curr_interval.converged | curr_interval.failed)

        secant2_raw_result = _secant2(
            value_and_gradients_function,
            val_0,
            curr_interval,
            f_lim,
            sufficient_decrease_param,
            curvature_param,
        )

        secant2_result = HagerZhangLineSearchResult(
            converged=secant2_raw_result.converged & tf.logical_not(curr_interval.failed),
            failed=secant2_raw_result.failed | curr_interval.failed,
            iterations=curr_interval.iterations + tf.cast(active, tf.int32),
            func_evals=secant2_raw_result.num_evals,
            left=secant2_raw_result.left,
            right=secant2_raw_result.right,
        )

        should_check_shrinkage = tf.logical_not(secant2_result.converged | secant2_result.failed)

        def _do_check_shrinkage() -> HagerZhangLineSearchResult:
            old_width = curr_interval.right.x - curr_interval.left.x
            new_width = secant2_result.right.x - secant2_result.left.x
            sufficient_shrinkage = new_width < old_width * shrinkage_param

            func_is_flat = _very_close(curr_interval.left.f, curr_interval.right.f) & _very_close(
                secant2_result.left.f, secant2_result.right.f
            )

            new_converged = should_check_shrinkage & sufficient_shrinkage & func_is_flat
            needs_inner_bisect = should_check_shrinkage & tf.logical_not(sufficient_shrinkage)

            inner_bisect_args = secant2_result._replace(
                converged=secant2_result.converged | new_converged
            )

            return tf.cond(
                needs_inner_bisect,
                lambda: _line_search_inner_bisection(
                    value_and_gradients_function,
                    inner_bisect_args,
                    needs_inner_bisect,
                    f_lim,
                ),
                lambda: inner_bisect_args,
            )

        next_args = tf.cond(
            should_check_shrinkage,
            _do_check_shrinkage,
            lambda: secant2_result,
        )

        interval_shrunk = tf.logical_not(next_args.failed) & _very_close(
            next_args.left.x, next_args.right.x
        )

        return (
            next_args._replace(converged=next_args.converged | interval_shrunk),
        )

    return tf.while_loop(
        cond=_loop_cond,
        body=_loop_body,
        loop_vars=(search_interval,),
        parallel_iterations=1,
    )[0]


def _prepare_args(
    value_and_gradients_function: Callable[[tf.Tensor], ValueAndGradient],
    initial_step_size: tf.Tensor,
    approximate_wolfe_threshold: tf.Tensor,
    val_0: ValueAndGradient | None = None,
) -> tuple[ValueAndGradient, ValueAndGradient, tf.Tensor, tf.Tensor]:
    val_initial = value_and_gradients_function(initial_step_size)

    # The value/slope at alpha=0 are usually already known by the caller
    # (current cost and directional derivative); accept them to avoid a full
    # re-evaluation per line search.
    if val_0 is None:
        val_0 = value_and_gradients_function(tf.zeros_like(val_initial.x))
        num_evals = tf.constant(2, dtype=tf.int32)
    else:
        num_evals = tf.constant(1, dtype=tf.int32)

    # Same threshold used by TFP.
    f_lim = val_0.f + approximate_wolfe_threshold * tf.math.abs(val_0.f)

    return val_0, val_initial, f_lim, num_evals


def _select_output_alpha(result: HagerZhangLineSearchResult, dtype: tf.dtypes.DType) -> tf.Tensor:
    zero = tf.constant(0.0, dtype=dtype)
    left_x = tf.cast(result.left.x, dtype)
    left_valid = tf.math.is_finite(left_x) & (left_x > zero)

    return tf.cond(
        result.converged,
        lambda: left_x,
        lambda: tf.cond(
            tf.logical_not(result.failed) & left_valid,
            lambda: left_x,
            lambda: zero,
        ),
    )


class LineSearchHagerZhang(LineSearch):
    def __init__(
        self,
        step_size_initial: float = 1.0,
        threshold_use_approximate_wolfe_condition: float = 1e-6,
        shrinkage_param: float = 0.66,
        expansion_param: float = 5.0,
        sufficient_decrease_param: float = 0.1,
        curvature_param: float = 0.9,
        max_iterations: int = 50,
    ):
        super().__init__(step_size_initial)
        self.name = "hager_zhang"

        # These match the TFP public API, with your local step_size_initial name
        # corresponding to TFP's initial_step_size.
        self.threshold_use_approximate_wolfe_condition = threshold_use_approximate_wolfe_condition
        self.shrinkage_param = shrinkage_param
        self.expansion_param = expansion_param
        self.sufficient_decrease_param = sufficient_decrease_param
        self.curvature_param = curvature_param
        self.max_iterations = max_iterations

    @tf.function(autograph=False, reduce_retracing=True)
    def search_result(
        self,
        w: tf.Tensor,
        p: tf.Tensor,
        value_and_grad_fn: Callable[[tf.Tensor], ValueAndGradient],
        val_0: ValueAndGradient | None = None,
        value_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    ) -> LineSearchResult:
        del p, value_fn  # unused by the scalar line-search routine itself

        dtype = w.dtype

        initial_step_size = tf.cast(self.step_size_initial, dtype)
        threshold_use_approximate_wolfe_condition = tf.cast(
            self.threshold_use_approximate_wolfe_condition, dtype
        )
        shrinkage_param = tf.cast(self.shrinkage_param, dtype)
        expansion_param = tf.cast(self.expansion_param, dtype)
        sufficient_decrease_param = tf.cast(self.sufficient_decrease_param, dtype)
        curvature_param = tf.cast(self.curvature_param, dtype)
        max_iterations = tf.cast(self.max_iterations, tf.int32)

        val_0, val_initial, f_lim, prepare_evals = _prepare_args(
            value_and_grad_fn,
            initial_step_size,
            threshold_use_approximate_wolfe_condition,
            val_0=val_0,
        )

        valid_inputs = (
            _is_finite(val_0)
            & (val_0.df < tf.zeros_like(val_0.df))
            & tf.math.is_finite(val_initial.x)
            & (val_initial.x > tf.zeros_like(val_initial.x))
        )

        # A bounded eval_fn may clamp the requested step (returned x smaller
        # than asked). If the clamped point still has negative slope and an
        # acceptable value, the constrained minimizer along this direction is
        # the clamp boundary itself: accept it immediately instead of letting
        # bracketing/bisection spin against the bound.
        clamped_initial = (
            valid_inputs
            & _is_finite(val_initial)
            & (val_initial.x < initial_step_size)
            & (val_initial.df < tf.zeros_like(val_initial.df))
            & (val_initial.f <= f_lim)
        )
        init_interval = HagerZhangLineSearchResult(
            converged=clamped_initial,
            failed=tf.logical_not(valid_inputs),
            func_evals=prepare_evals,
            iterations=tf.constant(0, dtype=tf.int32),
            left=tf.cond(clamped_initial, lambda: val_initial, lambda: val_0),
            right=val_initial,
        )

        def _apply_bracket_and_search() -> HagerZhangLineSearchResult:
            bracket_result = _bracket(
                value_and_grad_fn,
                init_interval,
                f_lim,
                max_iterations,
                expansion_param,
            )

            converged = init_interval.converged | _very_close(
                bracket_result.left.x,
                bracket_result.right.x,
            )

            exhausted_iterations = tf.logical_not(converged) & (
                bracket_result.iteration >= max_iterations
            )

            line_search_args = HagerZhangLineSearchResult(
                converged=converged,
                failed=bracket_result.failed | exhausted_iterations,
                iterations=bracket_result.iteration,
                func_evals=bracket_result.num_evals,
                left=bracket_result.left,
                right=bracket_result.right,
            )

            return _line_search_after_bracketing(
                value_and_grad_fn,
                line_search_args,
                val_0,
                f_lim,
                max_iterations,
                sufficient_decrease_param,
                curvature_param,
                shrinkage_param,
            )

        result = tf.cond(
            tf.logical_not(init_interval.failed) & tf.logical_not(init_interval.converged),
            _apply_bracket_and_search,
            lambda: init_interval,
        )

        return LineSearchResult(
            alpha=tf.cast(_select_output_alpha(result, dtype), dtype),
            converged=result.converged,
            failed=result.failed,
            func_evals=result.func_evals,
            iterations=result.iterations,
            left=result.left,
            right=result.right,
        )

    def search(
        self,
        w: tf.Tensor,
        p: tf.Tensor,
        value_and_grad_fn: Callable[[tf.Tensor], ValueAndGradient],
        val_0: ValueAndGradient | None = None,
        value_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
    ) -> tf.Tensor:
        return self.search_result(
            w,
            p,
            value_and_grad_fn,
            val_0=val_0,
            value_fn=value_fn,
        ).alpha
