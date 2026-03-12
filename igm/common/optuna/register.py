# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Register the IGM Optuna sweeper in Hydra's ConfigStore and plugin system.

Hydra requires sweeper plugin classes to have __module__ starting with
"hydra_plugins." (hardcoded check in Plugins._instantiate).  Rather than
maintaining a physical hydra_plugins/ directory, we create a thin subclass
dynamically with the correct __module__ and register it directly.
"""

from dataclasses import dataclass


@dataclass
class IGMOptunaSweeperConf:
    _target_: str = "hydra_plugins.igm_optuna.IGMOptunaSweeper"
    optuna_config: str = "optuna_params.yaml"


def register_sweeper():
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(
        group="hydra/sweeper",
        name="igm_optuna",
        node=IGMOptunaSweeperConf,
        provider="igm",
    )

    # Hydra's plugin check requires classname.startswith("hydra_plugins.")
    # AND its instantiate() resolves _target_ by importing the module.
    # We satisfy both without a physical directory by:
    #   1. Creating a subclass with __module__ = "hydra_plugins.igm_optuna"
    #   2. Injecting a virtual module into sys.modules so import resolves
    import sys
    import types

    from hydra.core.plugins import Plugins
    from igm.common.optuna.sweeper import IGMOptunaSweeper as _Base

    _HydraClass = type(
        "IGMOptunaSweeper",
        (_Base,),
        {"__module__": "hydra_plugins.igm_optuna"},
    )

    # Make "hydra_plugins.igm_optuna" importable
    # Also ensure the parent "hydra_plugins" namespace exists (needed when
    # IGM is pip-installed normally rather than in editable mode).
    # Preserve existing __path__ so other Hydra plugins (e.g. joblib
    # launcher) remain discoverable.
    if "hydra_plugins" not in sys.modules:
        import importlib
        try:
            ns = importlib.import_module("hydra_plugins")
        except ImportError:
            ns = types.ModuleType("hydra_plugins")
            ns.__path__ = []
            sys.modules["hydra_plugins"] = ns

    mod = types.ModuleType("hydra_plugins.igm_optuna")
    mod.IGMOptunaSweeper = _HydraClass
    sys.modules["hydra_plugins.igm_optuna"] = mod

    Plugins.instance().register(_HydraClass)
