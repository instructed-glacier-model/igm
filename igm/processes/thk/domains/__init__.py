"""Composable active-domain constraints for thickness transport.

A configuration selects any number of constraints; the active domain is their
intersection, published as ``state.thk_active_mask``. The resolved constraint
list is plain data, so it is read once by ``get_domain_constraints`` and then
passed to the two functions that use it.
"""

import tensorflow as tf

from . import grounded, initial_ice, interior, state_mask


DomainConstraints = {
    "grounded": grounded,
    "initial_ice": initial_ice,
    "interior": interior,
    "state_mask": state_mask,
}


def get_domain_constraints(cfg):
    """Resolve the configured constraint list once for a simulation."""
    options = getattr(cfg.processes.thk, "domain", None)
    configured = [] if options is None else list(options.get("constraints", []))
    constraints = []
    for entry in configured:
        if isinstance(entry, str) or not hasattr(entry, "get"):
            raise ValueError(
                "cfg.processes.thk.domain.constraints entries must contain "
                "a 'method' key."
            )
        name = str(entry.get("method", "")).strip().lower()
        try:
            backend = DomainConstraints[name]
        except KeyError:
            available = ", ".join(sorted(DomainConstraints))
            raise ValueError(
                "Unknown thickness active-domain constraint "
                f"{name!r}; available constraints: {available}."
            ) from None
        if not callable(getattr(backend, "get_mask", None)):
            raise TypeError(
                f"Thickness domain constraint {name!r} must define get_mask()."
            )
        constraints.append((name, backend, entry))
    return tuple(constraints)


def initialize_active_domain(cfg, state, constraints):
    """Run each constraint's optional setup, then publish the initial mask."""
    if not constraints:
        # This field is owned by the thickness module.  Removing a value loaded
        # from a restart prevents an old mask from silently changing the
        # default, unconstrained explicit/CFL path.
        if hasattr(state, "thk_active_mask"):
            del state.thk_active_mask
        return

    for _, backend, options in constraints:
        callback = getattr(backend, "initialize", None)
        if callable(callback):
            callback(options, cfg, state)
    update_active_domain(cfg, state, constraints)


def update_active_domain(cfg, state, constraints):
    """Publish ``state.thk_active_mask`` as the intersection of constraints."""
    if not constraints:
        return

    mask = tf.ones_like(state.thk, dtype=tf.bool)
    for name, backend, options in constraints:
        component = tf.cast(backend.get_mask(options, cfg, state), tf.bool)
        tf.debugging.assert_equal(
            tf.shape(component),
            tf.shape(mask),
            message=(
                "Thickness active-domain constraint "
                f"{name!r} returned a mask with the wrong shape."
            ),
        )
        mask = tf.logical_and(mask, component)
    state.thk_active_mask = mask


def face_masks(cell_mask):
    """Return active x/y faces, blocking flux across internal mask edges."""
    cell_mask = tf.cast(cell_mask, tf.bool)
    x_interior = tf.logical_and(cell_mask[:, :-1], cell_mask[:, 1:])
    y_interior = tf.logical_and(cell_mask[:-1, :], cell_mask[1:, :])
    return (
        tf.concat([cell_mask[:, :1], x_interior, cell_mask[:, -1:]], axis=1),
        tf.concat([cell_mask[:1, :], y_interior, cell_mask[-1:, :]], axis=0),
    )
