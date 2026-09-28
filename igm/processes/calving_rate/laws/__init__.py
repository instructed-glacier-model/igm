"""Calving laws of the calving_rate process, their dispatch table, and selection.

A law is a package ``laws/<name>/`` whose ``__init__`` exposes:

``calving_rate(cfg, state, geometry)``
    The calving rate (m/yr, >= 0). By default it is needed only on the ice
    nodes that carry velocity (``geometry.supported``). The process then
    carries it to the rest of the front band by the mean over the
    neighbours (:func:`..calving_rate.spread`), as PISM does.
``ON_BAND`` (required)
    With ``True``, the law returns the rate on the whole front band
    (``geometry.band``) itself, and it is not carried. A law that prescribes
    the front velocity relative to the ice at the front needs this: the
    front moves with the ice speed at the front (``geometry.front_speed``),
    which exceeds the speed of the last ice node on a spreading shelf.
    Carrying such a law from that node would bias the front velocity by the
    difference (``ice_speed``: 7-9 m/yr, 3 %, on the Albrecht et al. (2011)
    shelf at 5 km).

Its ``<name>.yaml`` lists the state variables it reads, for the runner's
needs check.
"""

from types import ModuleType
from typing import Tuple

from omegaconf import DictConfig

from . import constant, eigen, ice_speed, von_mises, water_depth, zero

CalvingLaws = {
    "constant": constant,
    "eigen": eigen,
    "ice_speed": ice_speed,
    "von_mises": von_mises,
    "water_depth": water_depth,
    "zero": zero,
}


def available_calving_laws() -> Tuple[str, ...]:
    """Return the calving-law names in deterministic order."""
    return tuple(sorted(CalvingLaws))


def get_calving_law(cfg: DictConfig) -> Tuple[str, ModuleType]:
    """Resolve ``cfg.processes.calving_rate.law`` into ``(name, module)``."""
    name = str(cfg.processes.calving_rate.law).strip().lower()
    if name == "thickness_threshold":
        raise ValueError(
            "calving_rate.law 'thickness_threshold' was removed: the minimum "
            "front thickness is now cfg.processes.thk.front.min_thickness."
        )
    try:
        law = CalvingLaws[name]
    except KeyError:
        available = ", ".join(available_calving_laws())
        raise ValueError(
            "cfg.processes.calving_rate.law must name an available calving "
            f"law; available laws: {available}. Got {name!r}."
        ) from None
    missing = [a for a in ("calving_rate", "ON_BAND") if not hasattr(law, a)]
    if missing:
        raise TypeError(
            f"Calving law {name!r} is missing attribute(s): {', '.join(missing)}."
        )
    return name, law
