# igmcfg/optuna_hooks.py
from typing import Any
from omegaconf import DictConfig
from optuna.trial import Trial


def configure(cfg: DictConfig, trial: Trial) -> None:
    # Add the Optuna trial number as a *fixed* param in the Hydra overrides
    trial.suggest_int("+optuna_trial_number", trial.number, trial.number)
