#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Stop the initial training of the network on its error against a direct solve.

At initialisation, the ice flow of the initial state is solved directly (identity mapping
and cg_newton, see ``reference.py``). Every ``freq`` iterations of the initial training,
the network's surface speed ``s`` is compared with the direct one ``s_ref``:

    e = 100 * median(|s - s_ref| / (s_ref + speed_floor)),

over grounded and over floating ice. The training stops when both errors meet their
targets at ``consecutive`` checks in a row, or when they no longer improve (``patience``);
``nbit_init`` is the cap.

The check is a success criterion of the network optimizer's halt, so it works with every
optimizer without touching their loops. It is active during the initial training only, and
it runs eagerly through ``tf.py_function``, like the error estimator's hook: the compiled
training step stays exactly the same, and so does the training.
"""

import csv
import time
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State
from igm.processes.thk.masks import compute_grounded_mask
from ..error_estimator.metrics import masked_median
from ..evaluator.evaluator import get_evaluator_inputs_from_state
from ..halt.criteria import Criterion
from ..halt.step_state import StepState
from ..mappings import Mapping
from .display import InitStopDisplay
from .reference import ReferenceSolve, solve_reference, surface_speed


class StopReason(IntEnum):
    CAP = 0
    TARGETS = 1
    PLATEAU = 2


def ratio_to_target(error: float, target: float) -> float:
    """error / target (inf for a zero target), and 0 for an empty region (NaN error)."""
    if np.isnan(error):
        return 0.0
    if target > 0.0:
        return error / target
    if error > 0.0:
        return np.inf
    return 0.0


class InitStopCriterion(Criterion):
    """The network's error against the direct solve meets its targets, or stops improving."""

    def __init__(
        self,
        mapping: Mapping,
        inputs: tf.Tensor,
        V_s: tf.Tensor,
        speed_ref: tf.Tensor,
        grounded: tf.Tensor,
        floating: tf.Tensor,
        target_grounded: float,
        target_floating: float,
        consecutive: int,
        speed_floor: float,
        freq: int,
        patience: int,
        min_gain: float,
        dtype: str,
        display: Optional[InitStopDisplay] = None,
    ):
        super().__init__(metric=None, dtype=dtype)
        self.name = "init_stop"

        self.mapping = mapping
        self.inputs = inputs
        self.V_s = V_s
        self.speed_ref = speed_ref
        self.grounded = grounded
        self.floating = floating

        self.target_grounded = float(target_grounded)
        self.target_floating = float(target_floating)
        self.consecutive = int(consecutive)
        self.speed_floor = float(speed_floor)
        self.freq = int(freq)
        self.patience = int(patience)
        self.min_gain = float(min_gain)
        self.display = display

        # Read inside the optimizer's graph: a variable, so that disarming needs no retrace
        self.active = tf.Variable(False, trainable=False)

        # Checked eagerly: plain Python state
        self.history: List[Tuple[int, float, float]] = []
        self.reason = StopReason.CAP
        self.best = np.inf
        self.iter_best = 0
        self.n_met = 0

    def arm(self) -> None:
        """Start a fresh record; the criterion is active until ``disarm``."""
        self.history = []
        self.reason = StopReason.CAP
        self.reset()
        self.active.assign(True)

    def disarm(self) -> None:
        self.active.assign(False)

    def reset(self) -> None:
        """Reset the trackers only: the record must survive the halt's resets."""
        self.best = np.inf
        self.iter_best = 0
        self.n_met = 0

    @tf.function
    def errors(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Floored median errors (%) of the network's surface speed: grounded, floating."""
        U, V = self.mapping.get_UV(self.inputs)
        speed = surface_speed(U[0], V[0], self.V_s)
        error = tf.abs(speed - self.speed_ref) / (self.speed_ref + self.speed_floor)
        error_grounded = 100.0 * masked_median(error, self.grounded)
        error_floating = 100.0 * masked_median(error, self.floating)
        return error_grounded, error_floating

    def check(self, step_state: StepState) -> Tuple[tf.Tensor, tf.Tensor]:
        """Checked every ``freq`` iterations (the updates applied so far) while active."""
        iterations = tf.cast(step_state.iter, tf.int32) + 1
        is_due = tf.logical_and(
            self.active, tf.equal(tf.math.mod(iterations, self.freq), 0)
        )
        return tf.cond(is_due, lambda: self.evaluate_eagerly(iterations), self.skip)

    def skip(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.constant(False), tf.constant(np.nan, self.dtype)

    def evaluate_eagerly(self, iterations: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # The check reads the network's weights outside the graph, where nothing orders it
        # after this iteration's update: read them in the graph first, and wait for that.
        weights = [tf.identity(weight) for weight in self.mapping.get_theta()]
        with tf.control_dependencies(weights):
            is_satisfied, ratio = tf.py_function(
                self.evaluate, [iterations], [tf.bool, self.dtype]
            )
        is_satisfied.set_shape([])
        ratio.set_shape([])
        return is_satisfied, ratio

    def evaluate(self, iterations) -> Tuple[tf.Tensor, tf.Tensor]:
        """One check: record the errors, then decide whether to stop."""
        iterations = int(iterations)
        error_grounded, error_floating = [float(e) for e in self.errors()]
        self.history.append((iterations, error_grounded, error_floating))

        ratio = max(
            ratio_to_target(error_grounded, self.target_grounded),
            ratio_to_target(error_floating, self.target_floating),
        )

        # Plateau tracker: a gain is a relative decrease larger than min_gain
        is_best = ratio < self.best * (1.0 - self.min_gain)
        if is_best:
            self.best = ratio
            self.iter_best = iterations

        # Targets: met at this check, and for how many checks in a row
        if ratio <= 1.0:
            self.n_met += 1
        else:
            self.n_met = 0

        if self.n_met >= self.consecutive:
            self.reason = StopReason.TARGETS
        elif self.patience > 0 and iterations - self.iter_best >= self.patience:
            self.reason = StopReason.PLATEAU
        else:
            self.reason = StopReason.CAP

        if self.display is not None and self.display.enabled:
            self.display.row(iterations, error_grounded, error_floating, is_best)

        is_satisfied = self.reason != StopReason.CAP
        return tf.constant(is_satisfied), tf.constant(ratio, self.dtype)


class InitStop:
    """Stops the initial training on its error against a direct solve (``init_stop``)."""

    def __init__(
        self,
        cfg: DictConfig,
        criterion: InitStopCriterion,
        reference: ReferenceSolve,
        display: InitStopDisplay,
    ):
        self.cfg_init_stop = cfg.processes.iceflow.unified.init_stop
        self.nbit_init = int(cfg.processes.iceflow.unified.nbit_init)
        self.criterion = criterion
        self.reference = reference
        self.display = display
        self.training_start = time.perf_counter()

    @staticmethod
    def check_cfg(cfg: DictConfig, state: State) -> None:
        """Raise a ValueError when init_stop cannot apply to this run."""
        cfg_unified = cfg.processes.iceflow.unified
        cfg_init_stop = cfg_unified.init_stop
        optimizer = state.iceflow.optimizer

        if cfg_unified.mapping != "network":
            raise ValueError(
                "❌ init_stop needs mapping=network: with the identity mapping, the "
                "initial solve already is the direct solve."
            )
        if optimizer.name == "sequential" or optimizer.halt is None:
            raise ValueError(
                f"❌ init_stop does not support the {optimizer.name!r} optimizer "
                "(it needs a single optimizer with a halt)."
            )
        if bool(cfg_unified.adaptive_patching.enabled):
            raise ValueError(
                "❌ init_stop does not support adaptive_patching: the initial training "
                "would restart its iteration count on every batch of patches."
            )
        freq = int(cfg_init_stop.freq)
        if freq < 1 or freq % int(optimizer.halt.freq):
            raise ValueError(
                f"❌ init_stop.freq ({freq}) must be a positive multiple of "
                f"unified.halt.freq ({optimizer.halt.freq})."
            )
        if int(cfg_init_stop.patience) < 0 or float(cfg_init_stop.speed_floor) <= 0.0:
            raise ValueError("❌ init_stop needs patience >= 0 and speed_floor > 0.")
        if int(cfg_init_stop.consecutive) < 1:
            raise ValueError("❌ init_stop.consecutive must be at least 1.")
        if int(cfg_init_stop.reference.nbit) < 1:
            raise ValueError("❌ init_stop.reference.nbit must be at least 1.")

    @classmethod
    def from_cfg(cls, cfg: DictConfig, state: State) -> "InitStop":
        """Solve the reference and attach the criterion to the network's optimizer."""
        InitStop.check_cfg(cfg, state)

        cfg_unified = cfg.processes.iceflow.unified
        cfg_init_stop = cfg_unified.init_stop
        cfg_physics = cfg.processes.iceflow.physics
        optimizer = state.iceflow.optimizer

        display = InitStopDisplay(
            bool(cfg_init_stop.display), progress=optimizer.display
        )
        display.setup(cfg)

        inputs = get_evaluator_inputs_from_state(cfg, state)
        with display.solving():
            reference = solve_reference(cfg, state, inputs)

        dtype = reference.speed.dtype
        thk = tf.cast(state.thk, dtype)
        ice = thk > 0.0
        grounded = ice & compute_grounded_mask(
            thk,
            tf.cast(state.topg, dtype),
            tf.cast(state.water_level, dtype),
            cfg_physics.water_density / cfg_physics.ice_density,
        )
        floating = ice & tf.logical_not(grounded)
        display.reference(reference, ice, grounded)

        criterion = InitStopCriterion(
            mapping=state.iceflow.mapping,
            inputs=inputs,
            V_s=state.iceflow.discr_v.V_s,
            speed_ref=reference.speed,
            grounded=grounded,
            floating=floating,
            target_grounded=float(cfg_init_stop.target_grounded),
            target_floating=float(cfg_init_stop.target_floating),
            consecutive=int(cfg_init_stop.consecutive),
            speed_floor=float(cfg_init_stop.speed_floor),
            freq=int(cfg_init_stop.freq),
            patience=int(cfg_init_stop.patience),
            min_gain=float(cfg_init_stop.min_gain),
            dtype=cfg.processes.iceflow.numerics.precision,
            display=display,
        )
        optimizer.halt.add_success(criterion)
        criterion.arm()

        return cls(cfg, criterion, reference, display)

    def finish(self) -> None:
        """After the initial training: deactivate the check, write the log, report."""
        self.criterion.disarm()
        training_seconds = time.perf_counter() - self.training_start
        history = self.criterion.history
        reason = self.criterion.reason

        log = str(self.cfg_init_stop.log)
        if log:
            with open(log, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["iterations", "error_grounded", "error_floating"])
                writer.writerows(history)

        if history:
            iterations, error_grounded, error_floating = history[-1]
        else:
            iterations, error_grounded, error_floating = self.nbit_init, np.nan, np.nan
        if reason == StopReason.CAP:
            iterations = self.nbit_init

        self.display.summary(
            reason=reason.name.lower(),
            iterations=iterations,
            nbit_init=self.nbit_init,
            error_grounded=error_grounded,
            error_floating=error_floating,
            target_grounded=float(self.cfg_init_stop.target_grounded),
            target_floating=float(self.cfg_init_stop.target_floating),
            reference_seconds=self.reference.seconds,
            training_seconds=training_seconds,
            log=log,
        )
