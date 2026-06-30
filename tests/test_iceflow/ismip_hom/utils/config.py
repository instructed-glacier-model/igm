#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
from typing import Any, Dict, List
import yaml


def load_test_config() -> Dict[str, Any]:
    """Load test config. Override with IGM_TEST_CONFIG env var (e.g. test_config_full.yaml)."""
    config_name = os.environ.get("IGM_TEST_CONFIG", "test_config.yaml")
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_name)
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_test_parameters() -> List[tuple]:
    """Generate test parameters from config."""
    config = load_test_config()
    params = []

    no_length_experiments = {"exp_e_1", "exp_e_2"}

    for exp in config["experiments"]:
        if exp in no_length_experiments:
            continue

        for length in config["lengths"]:
            if config["methods"]["unified"]["enabled"]:
                for mapping in config["methods"]["unified"]["mappings"]:
                    for optimizer in config["methods"]["unified"]["optimizers"]:
                        test_case = {
                            "experiment": exp,
                            "length": length,
                            "method": "unified",
                            "mapping": mapping,
                            "optimizer": optimizer,
                        }
                        if not _should_skip(test_case, config.get("skip", [])):
                            params.append((exp, length, "unified", mapping, optimizer))

            if config["methods"]["emulated"]["enabled"]:
                test_case = {
                    "experiment": exp,
                    "length": length,
                    "method": "emulated",
                }
                if not _should_skip(test_case, config.get("skip", [])):
                    params.append((exp, length, "emulated", None, None))

            if config["methods"]["solved"]["enabled"]:
                test_case = {
                    "experiment": exp,
                    "length": length,
                    "method": "solved",
                }
                if not _should_skip(test_case, config.get("skip", [])):
                    params.append((exp, length, "solved", None, None))

    return params


def _is_experiment_enabled(experiment: str, config: Dict[str, Any]) -> bool:
    """Check if experiment is in config."""
    return experiment in config["experiments"]


def get_unified_parameters(experiment: str) -> List[tuple]:
    """Get unified method parameters for experiment (with length scales)."""
    config = load_test_config()

    if not _is_experiment_enabled(experiment, config):
        return []

    if not config["methods"]["unified"]["enabled"]:
        return []

    params = []
    for length in config["lengths"]:
        for mapping in config["methods"]["unified"]["mappings"]:
            for optimizer in config["methods"]["unified"]["optimizers"]:
                params.append((length, mapping, optimizer))

    return params


def get_unified_parameters_no_length(experiment: str) -> List[tuple]:
    """Get unified method parameters for experiment (without length scales)."""
    config = load_test_config()

    if not _is_experiment_enabled(experiment, config):
        return []

    if not config["methods"]["unified"]["enabled"]:
        return []

    params = []
    for mapping in config["methods"]["unified"]["mappings"]:
        for optimizer in config["methods"]["unified"]["optimizers"]:
            params.append((mapping, optimizer))

    return params


def get_tolerance(experiment: str) -> float:
    """Get tolerance value for experiment."""
    config = load_test_config()
    tol_config = config["validation"]["tolerance"]

    # Check for experiment-specific tolerance
    exp_tolerances = tol_config.get("experiments", {})
    if experiment in exp_tolerances:
        return exp_tolerances[experiment]

    # Use default (backward compatible with old config format)
    return tol_config.get("default", tol_config.get("value", 0.10))


def _should_skip(test_case: Dict, skip_list: List[Dict]) -> bool:
    """Check if test case should be skipped."""
    for skip_item in skip_list:
        if all(test_case.get(k) == v for k, v in skip_item.items()):
            return True
    return False
