"""Sub-shelf melt laws of the bmb process, their dispatch table, and selection.

A law is a module exposing ``melt_rate(cfg, state, geometry)``, which returns
the sub-shelf melt rate (m ice eq. yr-1, positive for melt) on the shelf
nodes of ``geometry``, and optionally ``initialize(cfg, state)``, which checks
the configuration and the processes it relies on. Its ``<name>.yaml`` lists
the state variables it reads, for the runner's needs check.
"""

from types import ModuleType
from typing import Tuple

from omegaconf import DictConfig

from . import pico, picop, plume, prescribed, quadratic

MeltLaws = {
    "pico": pico,
    "picop": picop,
    "plume": plume,
    "prescribed": prescribed,
    "quadratic": quadratic,
}


def available_melt_laws() -> Tuple[str, ...]:
    """Return the melt-law names in deterministic order."""
    return tuple(sorted(MeltLaws))


def get_melt_law(cfg: DictConfig) -> Tuple[str, ModuleType]:
    """Resolve ``cfg.processes.bmb.method`` into ``(name, module)``."""
    name = str(cfg.processes.bmb.method).strip().lower()
    try:
        return name, MeltLaws[name]
    except KeyError:
        available = ", ".join(available_melt_laws())
        raise ValueError(
            "cfg.processes.bmb.method must name an available melt law; "
            f"available methods: {available}. Got {name!r}."
        ) from None
