# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Optuna integration for IGM (Tier 2: Hydra sweeper plugin).

Requires: pip install igm-model[optuna]
"""

try:
    import optuna  # noqa: F401
except ImportError:
    raise ImportError(
        "The IGM Optuna sweeper requires the 'optuna' package.\n"
        "Install it with:  pip install igm-model[optuna]"
    )

from .register import register_sweeper

register_sweeper()
