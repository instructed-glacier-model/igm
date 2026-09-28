"""Calving-front methods, their dispatch table, and selection.

``cfg.processes.thk.front.method`` selects how a marine ice front moves:
``none`` leaves it to the thickness transport, ``sub_grid`` is the
volume-based sub-grid front of Albrecht et al. (2011) as in PISM, and
``level_set`` a mass-consistent level-set front (after Bondzio et al., 2016).
Both front methods share the representation and transport step of
:mod:`.common`.
"""

from types import ModuleType
from typing import Optional, Tuple

from omegaconf import DictConfig

from . import level_set, sub_grid

#: How a front composes with the configured transport scheme.
#: ``replace_transport`` means the front owns mass transport itself;
#: ``after_transport`` means it runs immediately after the transport step.
UPDATE_MODES = ("after_transport", "replace_transport")

FrontMethods = {
    "level_set": level_set,
    "sub_grid": sub_grid,
}

#: Keys of the former front configuration, directly under ``thk``.
_REMOVED_KEYS = (
    "calving_front",
    "method",
    "front_slope_type",
    "interior_slope_type",
    "only_marine",
    "extend_halo",
    "extend_thresh",
    "sub_grid",
    "level_set",
)


def available_front_methods() -> Tuple[str, ...]:
    """Return production-available front-method names in deterministic order."""
    return tuple(
        sorted(
            name
            for name, backend in FrontMethods.items()
            if bool(getattr(backend, "AVAILABLE", False))
        )
    )


def get_front_method(cfg: DictConfig) -> Optional[str]:
    """Return the configured front method, or ``None`` for ``none``."""
    p = cfg.processes.thk
    removed = [key for key in _REMOVED_KEYS if key in p]
    if removed:
        raise ValueError(
            "The calving-front options moved to cfg.processes.thk.front "
            f"(method: none | sub_grid | level_set); remove {', '.join(removed)} "
            "from cfg.processes.thk (see igm/processes/thk/DESIGN.md)."
        )
    front = p.get("front", None)
    name = "none" if front is None else str(front.get("method", "none"))
    name = name.strip().lower()
    return None if name == "none" else name


def get_front(
    cfg: DictConfig, transport_name: str
) -> Tuple[Optional[str], Optional[ModuleType]]:
    """Resolve the optional front into ``(name, backend module)``.

    Returns ``(None, None)`` when no calving front is configured. Otherwise
    the method is checked against the selected transport scheme, so a chosen
    transport backend is never silently ignored.
    """
    name = get_front_method(cfg)
    if name is None:
        return None, None
    try:
        backend = FrontMethods[name]
    except KeyError:
        available = ", ".join(("none",) + available_front_methods())
        raise ValueError(
            "cfg.processes.thk.front.method must name an available front "
            f"method; available methods: {available}. Got {name!r}."
        ) from None

    required = (
        "UPDATE_MODE",
        "COMPATIBLE_TRANSPORTS",
        "AVAILABLE",
        "UNAVAILABLE_REASON",
    )
    missing = [constant for constant in required if not hasattr(backend, constant)]
    if missing:
        raise TypeError(
            f"Front method {name!r} is missing module constant(s): "
            f"{', '.join(missing)}."
        )

    update_mode = backend.UPDATE_MODE
    compatible_transports = backend.COMPATIBLE_TRANSPORTS

    if update_mode not in UPDATE_MODES:
        raise ValueError(
            f"Front method {name!r} has invalid UPDATE_MODE {update_mode!r}."
        )
    if not bool(backend.AVAILABLE):
        reason = backend.UNAVAILABLE_REASON or "the backend is not implemented"
        raise ValueError(
            f"cfg.processes.thk.front.method {name!r} is unavailable: {reason}."
        )
    if (
        compatible_transports is not None
        and transport_name not in compatible_transports
    ):
        compatible = ", ".join(compatible_transports)
        raise ValueError(
            f"Front method {name!r} uses {update_mode!r} and is "
            f"compatible only with thickness scheme(s): {compatible}; got "
            f"{transport_name!r}."
        )
    return name, backend
