#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Hard-fail check for legacy parameter names.

Step 2 of the iceflow parameter migration renamed several yaml keys (see
PARAM_CHANGE.md at the repo root). Rather than silently mapping old keys
to new ones, we abort at startup with a clear migration table whenever a
user's yaml still carries any legacy key. This avoids silent
behaviour changes between releases.
"""

from typing import List, Tuple
from omegaconf import OmegaConf, DictConfig


# (old dotted path, new dotted path or human-readable target)
_LEGACY_KEYS: List[Tuple[str, str]] = [
    # init_<x> moved into sliding: / viscosity:
    ("processes.iceflow.physics.init_slidingco",
     "processes.iceflow.physics.sliding.slidingco"),
    ("processes.iceflow.physics.init_tau_ref",
     "processes.iceflow.physics.sliding.tau_ref"),
    ("processes.iceflow.physics.init_arrhenius",
     "processes.iceflow.physics.viscosity.arrhenius"),
    ("processes.iceflow.physics.enhancement_factor",
     "processes.iceflow.physics.viscosity.enhancement_factor"),
    ("processes.iceflow.physics.exp_glen",
     "processes.iceflow.physics.viscosity.exponent"),
    ("processes.iceflow.physics.regu_glen",
     "processes.iceflow.physics.viscosity.regularization"),
]

# Per-law sliding sub-blocks were flattened; presence of any of these
# indicates the user is still on the pre-step-2 yaml shape.
_SLIDING_LAWS = ("weertman", "coulomb", "budd", "mohr_coulomb")
for _law in _SLIDING_LAWS:
    _LEGACY_KEYS.append((
        f"processes.iceflow.physics.sliding.{_law}",
        "processes.iceflow.physics.sliding (flat keys: regularization, "
        "exponent, u_ref, plus law-specific mu/N_ref/q_exponent/phi/...)",
    ))


def check_legacy_keys(cfg: DictConfig) -> None:
    """Abort with a clear migration error if any legacy key is present.

    Called from `igm_run.py` right after Hydra composes the config so the
    user sees the message before anything else runs.
    """
    found = [
        (old, new) for old, new in _LEGACY_KEYS
        if OmegaConf.select(cfg, old, default=None) is not None
    ]
    if not found:
        return
    lines = [f"  - '{old}'  →  '{new}'" for old, new in found]
    raise ValueError(
        "❌ Your config uses parameter names that have been renamed. "
        "Please update your yaml file(s):\n"
        + "\n".join(lines)
        + "\nSee PARAM_CHANGE.md for the full migration table."
    )
