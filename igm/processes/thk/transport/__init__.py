"""Thickness-transport implementations, their dispatch table, and selection."""

from . import explicit, ffsl, implicit, implicit_x


TransportSchemes = {
    "explicit": explicit,
    "ffsl": ffsl,
    "implicit": implicit,
    "implicit_x": implicit_x,
}


def available_transport_schemes():
    """Return configured transport names in deterministic order."""
    return tuple(sorted(TransportSchemes))


def get_transport(cfg):
    """Resolve ``cfg.processes.thk.scheme`` into ``(name, module)``."""
    name = str(getattr(cfg.processes.thk, "scheme", "explicit")).strip().lower()
    try:
        return name, TransportSchemes[name]
    except KeyError:
        available = ", ".join(available_transport_schemes())
        raise ValueError(
            "cfg.processes.thk.scheme must name an available thickness "
            f"transport; available schemes: {available}. Got {name!r}."
        ) from None
