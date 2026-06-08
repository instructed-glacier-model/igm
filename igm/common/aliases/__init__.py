#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""State-variable alias loader.

Alias sets live in YAML files next to this package (one file per naming
convention).  Each file maps  alias_name: canonical_name  where the canonical
name is the IGM canonical (internal implementation) name stored in State.__dict__.

Built-in sets
-------------
  descriptive  –  long English names (bed_elevation, surface_elevation, …)
  pism         –  PISM variable names (temp → T, climatic_mass_balance → smb, …)

Custom sets
-----------
Call load_aliases_from_yaml(path) with any YAML file that follows the same
alias: canonical  format, then pass the result to State.register_aliases().
"""

from pathlib import Path
import yaml

_ALIASES_DIR = Path(__file__).parent


def load_aliases_from_yaml(path) -> dict[str, str]:
    """Return alias → canonical mapping loaded from *path*."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_builtin_aliases(*names: str) -> dict[str, str]:
    """Merge one or more built-in alias sets by name (omit the .yaml extension).

    Example::

        aliases = load_builtin_aliases("descriptive", "my_custom_set")
        State.register_aliases(aliases)
    """
    result: dict[str, str] = {}
    for name in names:
        result.update(load_aliases_from_yaml(_ALIASES_DIR / f"{name}.yaml"))
    return result


def load_aliases_from_core_cfg(aliases_cfg) -> dict[str, str]:
    """Build the alias mapping from cfg.core.aliases (OmegaConf DictConfig or plain dict)."""
    result: dict[str, str] = {}
    for name in (aliases_cfg.builtin or []):
        result.update(load_builtin_aliases(name))
    for path in (aliases_cfg.extra_files or []):
        result.update(load_aliases_from_yaml(path))
    return result
