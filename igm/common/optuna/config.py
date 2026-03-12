# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Load and validate optimize.yaml configuration."""

from pathlib import Path
import yaml


def load_optimize_config(cwd, filename="optimize.yaml"):
    """Load optimize.yaml from *cwd* (or an absolute path).

    Returns a plain dict with validated required keys.

    Supports two formats for specifying objectives:

    New format (recommended)::

        objectives:
          - name: cost_volume
            direction: minimize
          - name: cost_speed
            direction: minimize

    Legacy format::

        direction:
          - minimize
          - minimize
    """
    path = Path(cwd) / filename if not Path(filename).is_absolute() else Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"Optimization config not found: {path}\n"
            f"Create an optimize.yaml in your working directory."
        )

    with open(path) as f:
        cfg = yaml.safe_load(f)

    # --- Normalize objectives ---
    if "objectives" in cfg:
        objs = cfg["objectives"]
        if not isinstance(objs, list) or len(objs) == 0:
            raise ValueError("'objectives' must be a non-empty list")
        for i, obj in enumerate(objs):
            if "name" not in obj or "direction" not in obj:
                raise ValueError(
                    f"Objective #{i} must have 'name' and 'direction'"
                )
        # Derive direction list from objectives
        cfg["direction"] = [o["direction"] for o in objs]
    elif "direction" not in cfg:
        raise KeyError(
            "optimize.yaml must have either 'objectives' or 'direction'"
        )

    # Validate other required keys
    for key in ("n_trials", "parameters"):
        if key not in cfg:
            raise KeyError(f"optimize.yaml is missing required key: '{key}'")

    if not isinstance(cfg["parameters"], list) or len(cfg["parameters"]) == 0:
        raise ValueError("optimize.yaml 'parameters' must be a non-empty list")

    for i, param in enumerate(cfg["parameters"]):
        if "name" not in param or "type" not in param:
            raise ValueError(
                f"Parameter #{i} in optimize.yaml must have 'name' and 'type'"
            )
        if param["type"] in ("float", "int"):
            if "low" not in param or "high" not in param:
                raise ValueError(
                    f"Parameter '{param['name']}' (type={param['type']}) "
                    f"must have 'low' and 'high'"
                )
        elif param["type"] == "categorical":
            if "choices" not in param:
                raise ValueError(
                    f"Parameter '{param['name']}' (type=categorical) must have 'choices'"
                )
        else:
            raise ValueError(
                f"Parameter '{param['name']}' has unsupported type: '{param['type']}'. "
                f"Supported: float, int, categorical"
            )

    return cfg
