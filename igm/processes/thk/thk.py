#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Evolve ice thickness and the ice surfaces by mass conservation.

Thickness transport and calving-front tracking are separate extension points,
each resolved by its own package (``transport.get_transport``,
``fronts.get_front``, ``domains.get_domain_constraints``). This module selects
them once, checks that the combination can actually run, and stores the result
on ``state.thk_components`` for the update to use.

A front declares how it composes with transport through its ``update_mode``.
``replace_transport`` means it owns mass transport itself, which accommodates
IGM's existing sub-grid front code while making that non-composability
explicit instead of silently ignoring the selected transport scheme;
``after_transport`` means it runs right after the transport step.
"""

from dataclasses import dataclass

from . import boundary
from .domains import (
    get_domain_constraints,
    initialize_active_domain,
    update_active_domain,
)
from .fronts import get_front
from .surfaces import update_surfaces, validate_density_ratio
from .transport import get_transport


@dataclass
class ThkComponents:
    """Container for the components that evolve the ice thickness.

    The selected components are attached here by :func:`_select_components`.
    A component may also park its own per-run bookkeeping on the container
    (``initial_ice_mask``, ``psi_built``, ...) rather than on ``state``, which
    keeps such private values out of the field namespace that the output
    modules write.
    """

    transport_name: str
    transport: object
    front_name: str | None
    front: object | None
    mass_transport_name: str
    mass_transport: object
    front_after_transport: object | None
    domain_constraints: tuple
    boundaries: boundary.BoundaryConditions
    initialized_components: tuple
    transport_options: object | None = None
    initial_ice_mask: object | None = None
    psi_built: bool = False
    steps_since_reinit: int = 0


def _check_component(name, module, kind):
    """Reject a dispatch-table entry that cannot act as a component."""
    missing = [
        callback
        for callback in ("initialize", "update")
        if not callable(getattr(module, callback, None))
    ]
    if missing:
        raise TypeError(
            f"{kind} {name!r} is missing callable(s): {', '.join(missing)}."
        )


def _select_components(cfg):
    """Resolve the configuration into a validated set of components.

    Selection, composition, and the checks are one step because they depend on
    each other: the front is only compatible with certain transport schemes,
    and which component ends up owning mass transport decides which one has to
    support active domains.
    """

    # Select
    transport_name, transport = get_transport(cfg)
    front_name, front_method = get_front(cfg, transport_name)
    front = None if front_method is None else front_method.backend
    constraints = get_domain_constraints(cfg)

    _check_component(transport_name, transport, "Thickness transport")
    if front is not None:
        _check_component(front_name, front, "Front method")

    # Compose: a "replace_transport" front owns mass transport itself, so the
    # selected transport scheme does not run at all; an "after_transport"
    # front instead runs right after it.
    replaces_transport = (
        front_method is not None
        and front_method.update_mode == "replace_transport"
    )
    mass_transport_name = front_name if replaces_transport else transport_name
    mass_transport = front if replaces_transport else transport

    # Check the combination
    boundaries = boundary.get_boundary_conditions(cfg)
    boundary.validate_backend(boundaries, transport, transport_name)
    if front is not None:
        boundary.validate_backend(boundaries, front, front_name)
    if constraints and not getattr(mass_transport, "SUPPORTS_ACTIVE_DOMAIN", False):
        raise ValueError(
            f"Thickness backend {mass_transport_name!r} does not support "
            "active-domain constraints."
        )
    smooth_sigma = float(getattr(cfg.processes.thk, "divflux_smooth_sigma", 0.0))
    if smooth_sigma != 0.0 and not getattr(
        mass_transport, "SUPPORTS_DIVFLUX_SMOOTHING", False
    ):
        raise ValueError(
            f"Thickness backend {mass_transport_name!r} cannot apply "
            "cfg.processes.thk.divflux_smooth_sigma, which is "
            "available only with scheme: explicit."
        )

    front_after_transport = (
        None if front is None or replaces_transport else front
    )
    initialized_components = (
        (front,)
        if replaces_transport
        else ((transport,) if front is None else (transport, front))
    )
    return ThkComponents(
        transport_name=transport_name,
        transport=transport,
        front_name=front_name,
        front=front,
        mass_transport_name=mass_transport_name,
        mass_transport=mass_transport,
        front_after_transport=front_after_transport,
        domain_constraints=constraints,
        boundaries=boundaries,
        initialized_components=initialized_components,
    )


def initialize(cfg, state):
    if not hasattr(state, "topg"):
        raise ValueError(
            "The 'thk' module requires an initial topography ('state.topg')."
        )

    # Select and check everything before any component touches the state.
    validate_density_ratio(cfg)
    components = _select_components(cfg)
    state.thk_components = components

    for component in components.initialized_components:
        component.initialize(cfg, state)
    initialize_active_domain(cfg, state, components.domain_constraints)

    update_surfaces(cfg, state)


def update(cfg, state):
    if state.it < 0:
        return

    if hasattr(state, "logger"):
        # Do not materialize state.t on the host: the implicit solve and the
        # surrounding update can otherwise remain fully asynchronous.
        state.logger.info("Ice thickness equation")

    components = state.thk_components

    # The active domain is refreshed on both sides of the update so that
    # transport sees the mask implied by the current geometry, and the
    # published mask matches the geometry the step produced.
    if components.domain_constraints:
        update_active_domain(cfg, state, components.domain_constraints)
    components.mass_transport.update(cfg, state)
    if components.front_after_transport is not None:
        components.front_after_transport.update(cfg, state)
    if components.domain_constraints:
        update_active_domain(cfg, state, components.domain_constraints)

    update_surfaces(cfg, state)


def finalize(cfg, state):
    """Run the optional finalize callback of each selected component."""
    components = state.thk_components
    for component in reversed(components.initialized_components):
        callback = getattr(component, "finalize", None)
        if callable(callback):
            callback(cfg, state)
