"""Calving-front implementations, their dispatch table, and selection."""

from typing import NamedTuple

from . import level_set, sub_grid


class FrontMethod(NamedTuple):
    """Front backend and its transport-composition contract."""

    backend: object
    update_mode: str
    compatible_transports: tuple | None
    available: bool
    unavailable_reason: str


#: How a front composes with the configured transport scheme.
#: ``replace_transport`` means the front owns mass transport itself;
#: ``after_transport`` means it runs immediately after the transport step.
UPDATE_MODES = ("after_transport", "replace_transport")


FrontMethods = {
    "level_set": FrontMethod(
        backend=level_set,
        update_mode="replace_transport",
        compatible_transports=("explicit",),
        available=False,
        unavailable_reason="the level-set implementation is not production-ready",
    ),
    "sub_grid": FrontMethod(
        backend=sub_grid,
        update_mode="replace_transport",
        compatible_transports=("explicit",),
        available=True,
        unavailable_reason="",
    ),
}


def available_front_methods():
    """Return configured front-method names in deterministic order."""
    return tuple(sorted(FrontMethods))


def get_front(cfg, transport_name):
    """Resolve the optional front into ``(name, FrontMethod)``.

    Returns ``(None, None)`` when no calving front is configured. Otherwise
    the method is checked against the selected transport scheme, so a chosen
    transport backend is never silently ignored.
    """
    p = cfg.processes.thk
    if not bool(getattr(p, "calving_front", False)):
        return None, None

    name = str(getattr(p, "method", "sub_grid")).strip().lower()
    try:
        method = FrontMethods[name]
    except KeyError:
        available = ", ".join(available_front_methods())
        raise ValueError(
            "cfg.processes.thk.method must name an available front method; "
            f"available methods: {available}. Got {name!r}."
        ) from None

    if method.update_mode not in UPDATE_MODES:
        raise ValueError(
            f"Front method {name!r} has invalid update_mode "
            f"{method.update_mode!r}."
        )
    if not method.available:
        reason = method.unavailable_reason or "the backend is not implemented"
        raise ValueError(
            f"cfg.processes.thk.method {name!r} is unavailable: {reason}."
        )
    if (
        method.compatible_transports is not None
        and transport_name not in method.compatible_transports
    ):
        compatible = ", ".join(method.compatible_transports)
        raise ValueError(
            f"Front method {name!r} uses {method.update_mode!r} and is "
            f"compatible only with thickness scheme(s): {compatible}; got "
            f"{transport_name!r}."
        )
    return name, method
