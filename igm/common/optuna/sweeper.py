# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Custom Hydra sweeper plugin using Optuna directly.

Replaces the upstream hydra-optuna-sweeper (which pins optuna<3.0) with a
lightweight sweeper that:
  - Uses Optuna directly (no version ceiling)
  - Exposes the full Optuna API (any sampler, pruner, storage)
  - Runs each trial in a subprocess (clean TF state per trial)
  - Supports parallel trials with GPU distribution
  - Preserves multirun, multi-GPU, profiling, and all other Hydra features
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Optional

from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class IGMOptunaSweeper(Sweeper):
    """Custom Hydra sweeper that uses Optuna directly.

    Each trial is launched as a subprocess (``igm_run`` with overrides)
    so that TensorFlow graph state is fully reset between trials.

    When ``n_jobs > 1``, trials within a batch run in parallel.
    Use ``gpu_ids`` to distribute trials across GPUs.
    """

    def __init__(self, optuna_config: str = "optuna_params.yaml"):
        self.optuna_config = optuna_config
        self.config: Optional[DictConfig] = None
        self.hydra_context: Optional[HydraContext] = None
        self.task_function = None

    def setup(
        self,
        *,
        config: DictConfig,
        hydra_context: HydraContext,
        task_function,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def _build_env(self, optimize_cfg, trial_index_in_batch):
        """Build environment variables for a trial subprocess."""
        env = os.environ.copy()

        gpu_ids = optimize_cfg.get("gpu_ids")
        if gpu_ids:
            gpu_id = gpu_ids[trial_index_in_batch % len(gpu_ids)]
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        if optimize_cfg.get("gpu_allow_growth", False):
            env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        return env

    def _extract_values(self, score_raw, optimize_cfg):
        """Extract objective values from the score (dict, list, or scalar).

        When ``objectives`` is configured, extracts the named keys from
        a score dict.  Falls back to positional or scalar for legacy format.
        """
        obj_names = self._obj_names(optimize_cfg)
        is_multi = isinstance(optimize_cfg["direction"], list)

        if isinstance(score_raw, dict) and obj_names:
            missing = [n for n in obj_names if n not in score_raw]
            if missing:
                raise KeyError(
                    f"Score dict is missing objectives: {missing}. "
                    f"Available keys: {list(score_raw.keys())}"
                )
            return [score_raw[n] for n in obj_names]

        # Legacy: score is already a list or scalar
        return score_raw

    @staticmethod
    def _obj_names(optimize_cfg):
        """Return objective names or None."""
        objs = optimize_cfg.get("objectives")
        if objs:
            return [o["name"] for o in objs]
        return None

    def _collect_result(self, trial, process, score_file, optimize_cfg):
        """Wait for a trial process to finish and report result to Optuna."""
        import optuna

        is_multi = isinstance(optimize_cfg["direction"], list)

        try:
            stdout, stderr = process.communicate(
                timeout=optimize_cfg.get("trial_timeout")
            )
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            self.study.tell(
                trial.number, state=optuna.trial.TrialState.FAIL
            )
            logger.warning(f"Trial {trial.number} timed out")
            return

        if score_file.exists():
            with open(score_file) as f:
                score_raw = json.load(f)
            value = self._extract_values(score_raw, optimize_cfg)
            self.study.tell(trial.number, value)
            logger.info(f"Trial {trial.number} completed: score={value}")
        elif process.returncode == 0:
            fallback = (
                [float("inf")] * len(optimize_cfg["direction"])
                if is_multi
                else float("inf")
            )
            self.study.tell(trial.number, fallback)
            logger.warning(
                f"Trial {trial.number}: no score file, using inf"
            )
        else:
            self.study.tell(
                trial.number, state=optuna.trial.TrialState.FAIL
            )
            stderr_tail = "\n".join(
                stderr.strip().split("\n")[-5:]
            )
            logger.warning(
                f"Trial {trial.number} failed "
                f"(exit {process.returncode}):\n{stderr_tail}"
            )

    def sweep(self, arguments: List[str]) -> Any:
        import optuna

        from igm.common.optuna.config import load_optimize_config
        from igm.common.optuna.core import create_study, report_results

        assert self.config is not None

        cwd = Path(self.config.hydra.runtime.cwd)
        sweep_dir = Path(self.config.hydra.sweep.dir)

        optimize_cfg = load_optimize_config(cwd, self.optuna_config)
        self.study = create_study(optimize_cfg)

        n_trials = optimize_cfg["n_trials"]
        n_jobs = optimize_cfg.get("n_jobs", 1)
        is_multi = isinstance(optimize_cfg["direction"], list)

        # Collect the base overrides from the original command line
        base_overrides = list(arguments)

        # Add fixed overrides (e.g. target values) from optimize config
        for key, val in optimize_cfg.get("overrides", {}).items():
            base_overrides.append(f"{key}={val}")

        obj_names = self._obj_names(optimize_cfg)

        for batch_start in range(0, n_trials, n_jobs):
            batch_end = min(batch_start + n_jobs, n_trials)
            batch_count = batch_end - batch_start

            batch_trials = []
            batch_overrides_list = []

            for _ in range(batch_count):
                trial = self.study.ask()
                batch_trials.append(trial)

                overrides = list(base_overrides)
                for param in optimize_cfg["parameters"]:
                    name = param["name"]
                    if param["type"] == "float":
                        value = trial.suggest_float(
                            name,
                            param["low"],
                            param["high"],
                            log=param.get("log", False),
                        )
                    elif param["type"] == "int":
                        value = trial.suggest_int(
                            name,
                            param["low"],
                            param["high"],
                            log=param.get("log", False),
                        )
                    elif param["type"] == "categorical":
                        value = trial.suggest_categorical(name, param["choices"])
                    else:
                        raise ValueError(
                            f"Unsupported param type: {param['type']}"
                        )
                    overrides.append(f"{name}={value}")

                batch_overrides_list.append(overrides)

            if n_jobs == 1:
                # Sequential: simpler code path
                trial = batch_trials[0]
                overrides = batch_overrides_list[0]
                trial_dir = sweep_dir / str(trial.number)

                cmd = [
                    "igm_run",
                    *overrides,
                    f"hydra.run.dir={trial_dir}",
                ]

                logger.info(
                    f"Trial {trial.number}: "
                    + " ".join(overrides)
                )

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=self._build_env(optimize_cfg, 0),
                )

                score_file = cwd / str(trial_dir) / "_igm_score.json"
                self._collect_result(trial, proc, score_file, optimize_cfg)

            else:
                # Parallel: launch all trials in the batch, then wait
                running = []
                for idx, (trial, overrides) in enumerate(
                    zip(batch_trials, batch_overrides_list)
                ):
                    trial_dir = sweep_dir / str(trial.number)
                    cmd = [
                        "igm_run",
                        *overrides,
                        f"hydra.run.dir={trial_dir}",
                    ]

                    logger.info(
                        f"Trial {trial.number} (parallel {idx+1}/{batch_count}): "
                        + " ".join(overrides)
                    )

                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(cwd),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=self._build_env(optimize_cfg, idx),
                    )

                    score_file = cwd / str(trial_dir) / "_igm_score.json"
                    running.append((trial, proc, score_file))

                # Wait for all to finish
                for trial, proc, score_file in running:
                    self._collect_result(
                        trial, proc, score_file, optimize_cfg
                    )

        report_results(self.study, cwd, obj_names)

        if is_multi:
            return self.study.best_trials
        return self.study.best_trial
