"""Calving-front implementations, their dispatch table, and selection."""

from . import level_set, sub_grid


#: How a front composes with the configured transport scheme.
#: ``replace_transport`` means the front owns mass transport itself;
#: ``after_transport`` means it runs immediately after the transport step.
UPDATE_MODES = ("after_transport", "replace_transport")


FrontMethods = {
    "level_set": level_set,
    "sub_grid": sub_grid,
}


def available_front_methods():
    """Return production-available front-method names in deterministic order."""
    return tuple(
        sorted(
            name
            for name, backend in FrontMethods.items()
            if bool(getattr(backend, "AVAILABLE", False))
        )
    )


def get_front(cfg, transport_name):
    """Resolve the optional front into ``(name, backend module)``.

    Returns ``(None, None)`` when no calving front is configured. Otherwise
    the method is checked against the selected transport scheme, so a chosen
    transport backend is never silently ignored.
    """
    p = cfg.processes.thk
    if not bool(getattr(p, "calving_front", False)):
        return None, None

    name = str(getattr(p, "method", "sub_grid")).strip().lower()
    try:
        backend = FrontMethods[name]
    except KeyError:
        available = ", ".join(available_front_methods())
        raise ValueError(
            "cfg.processes.thk.method must name an available front method; "
            f"available methods: {available}. Got {name!r}."
        ) from None

    required = (
        "UPDATE_MODE",
        "COMPATIBLE_TRANSPORTS",
        "AVAILABLE",
        "UNAVAILABLE_REASON",
    )
    missing = [
        constant for constant in required if not hasattr(backend, constant)
    ]
    if missing:
        raise TypeError(
            f"Front method {name!r} is missing module constant(s): "
            f"{', '.join(missing)}."
        )

    update_mode = backend.UPDATE_MODE
    compatible_transports = backend.COMPATIBLE_TRANSPORTS

    if update_mode not in UPDATE_MODES:
        raise ValueError(
            f"Front method {name!r} has invalid UPDATE_MODE "
            f"{update_mode!r}."
        )
    if not bool(backend.AVAILABLE):
        reason = backend.UNAVAILABLE_REASON or "the backend is not implemented"
        raise ValueError(
            f"cfg.processes.thk.method {name!r} is unavailable: {reason}."
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
