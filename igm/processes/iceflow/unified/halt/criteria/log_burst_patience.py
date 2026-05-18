import tensorflow as tf
from typing import Tuple

from .criterion import Criterion
from ..metrics import Metric
from ..step_state import StepState

class CriterionLogBurstPatience(Criterion):
    """
    Stop when the cost has not achieved a meaningful log-scale improvement
    for an adaptive patience period.

    A small log improvement resets patience. A larger log improvement is counted
    as a burst and increases the patience budget for later plateau phases.
    """

    def __init__(
        self,
        metric: Metric,
        dtype: str,
        patience: int,
        log_tol: float = 1.0e-3,
        burst_log_tol: float = 4.879016416943205e-2,  # log(1.05)
        patience_growth: float = 1.5,
        max_patience: int | None = None,
        min_iter: int = 0,
        cost_floor: float = 1.0e-30,
    ) -> None:
        super().__init__(metric=metric, dtype=dtype)

        self.name = "log_burst_patience"
        self.base_patience = tf.constant(float(patience), dtype=self.dtype)
        self.max_patience = tf.constant(
            float(max_patience if max_patience is not None else patience),
            dtype=self.dtype,
        )
        self.min_iter = tf.constant(int(min_iter), dtype=tf.int32)
        self.log_tol = tf.constant(float(log_tol), dtype=self.dtype)
        self.burst_log_tol = tf.constant(float(burst_log_tol), dtype=self.dtype)
        self.patience_growth = tf.constant(float(patience_growth), dtype=self.dtype)
        self.cost_floor = tf.constant(float(cost_floor), dtype=self.dtype)

        self.ref_cost = tf.Variable(0.0, dtype=self.dtype, trainable=False)
        self.iter_last_progress = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.n_bursts = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.init = tf.Variable(False, dtype=tf.bool, trainable=False)

    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        cost = tf.cast(self.metric.compute(step_state), self.dtype)
        cost = tf.reshape(cost, [])
        iter_ = tf.cast(step_state.iter, tf.int32)

        def initialize():
            self.ref_cost.assign(cost)
            self.iter_last_progress.assign(iter_)
            self.n_bursts.assign(0)
            self.init.assign(True)
            return tf.constant(False), tf.constant(0.0, dtype=self.dtype)

        def compute():
            ref = tf.maximum(self.ref_cost, self.cost_floor)
            cur = tf.maximum(cost, self.cost_floor)
            log_gain = tf.math.log(ref / cur)

            is_progress = log_gain >= self.log_tol
            is_burst = log_gain >= self.burst_log_tol

            def on_progress():
                self.ref_cost.assign(cost)
                self.iter_last_progress.assign(iter_)
                self.n_bursts.assign_add(tf.cast(is_burst, tf.int32))
                return tf.constant(False), tf.constant(0.0, dtype=self.dtype)

            def on_plateau():
                effective_patience = self.base_patience * tf.pow(
                    self.patience_growth,
                    tf.cast(self.n_bursts, self.dtype),
                )
                effective_patience = tf.minimum(effective_patience, self.max_patience)

                stale_iters = iter_ - self.iter_last_progress
                stale_iters_f = tf.cast(stale_iters, self.dtype)
                enough_iters = iter_ >= self.min_iter
                patience_exceeded = stale_iters_f >= effective_patience
                is_satisfied = tf.logical_and(enough_iters, patience_exceeded)

                return is_satisfied, stale_iters_f

            return tf.cond(is_progress, on_progress, on_plateau)

        return tf.cond(self.init, compute, initialize)

    def reset(self) -> None:
        self.init.assign(False)