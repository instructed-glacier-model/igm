#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Boundary policies shared by thickness-transport backends.

``zero`` is IGM's historical open boundary: ice thickness is zero outside
the domain, so outflow is allowed and inflow carries no ice. ``symmetric`` is
a reflecting/symmetry boundary with zero normal face velocity, hence exactly
zero mass flux. ``periodic`` connects the opposite domain faces.

Each side is independent. A flowline can consequently use a symmetric left
ice divide and an open right terminus, while a map-plane domain can combine
those with symmetric top and bottom sides.
"""

from typing import NamedTuple

import tensorflow as tf


class BoundaryConditions(NamedTuple):
    """Static modes at the four sides of the horizontal domain."""

    left: str
    right: str
    top: str
    bottom: str


_ALIASES = {
    "closed": "symmetric",
    "no_flux": "symmetric",
    "open": "zero",
    "reflective": "symmetric",
}
_VALID_MODES = ("periodic", "symmetric", "zero")


def _normalize(mode):
    normalized = str(mode).strip().lower().replace("-", "_")
    return _ALIASES.get(normalized, normalized)


def get_boundary_conditions(cfg):
    """Read the canonical four-side boundary configuration."""
    p = cfg.processes.thk
    legacy = [
        name for name in ("flux_mode_h", "flux_mode_u") if hasattr(p, name)
    ]
    if legacy:
        raise ValueError(
            "Legacy thickness boundary option(s) "
            f"{', '.join(legacy)} are not supported; configure "
            "boundary.left/right/top/bottom."
        )

    options = getattr(p, "boundary", None)
    if options is None:
        modes = BoundaryConditions("zero", "zero", "zero", "zero")
    else:
        if isinstance(options, str) or not hasattr(options, "keys"):
            raise ValueError(
                "cfg.processes.thk.boundary must contain the side keys "
                "left, right, top, and bottom."
            )
        allowed = {"left", "right", "top", "bottom"}
        unknown = sorted(set(options.keys()) - allowed)
        if unknown:
            raise ValueError(
                "Unknown cfg.processes.thk.boundary key(s): "
                f"{', '.join(unknown)}; use left, right, top, and bottom."
            )
        missing = sorted(allowed - set(options.keys()))
        if missing:
            raise ValueError(
                "cfg.processes.thk.boundary must define all four sides; "
                f"missing: {', '.join(missing)}."
            )
        modes = BoundaryConditions(
            _normalize(options["left"]),
            _normalize(options["right"]),
            _normalize(options["top"]),
            _normalize(options["bottom"]),
        )

    invalid = [mode for mode in modes if mode not in _VALID_MODES]
    if invalid:
        available = ", ".join(_VALID_MODES)
        raise ValueError(
            "cfg.processes.thk.boundary modes must be one of "
            f"{available}; got left={modes.left!r}, right={modes.right!r}, "
            f"top={modes.top!r}, bottom={modes.bottom!r}."
        )
    if (modes.left == "periodic") != (modes.right == "periodic"):
        raise ValueError(
            "Periodic x boundaries must be paired: set both boundary.left "
            "and boundary.right to 'periodic'."
        )
    if (modes.top == "periodic") != (modes.bottom == "periodic"):
        raise ValueError(
            "Periodic y boundaries must be paired: set both boundary.top "
            "and boundary.bottom to 'periodic'."
        )
    return modes


def validate_backend(boundaries, backend, backend_name):
    """Reject boundary modes a backend does not explicitly support."""
    supported = (
        tuple(backend)
        if isinstance(backend, (tuple, list))
        else tuple(getattr(backend, "SUPPORTED_BOUNDARY_MODES", ("zero",)))
    )
    unsupported = sorted(set(boundaries) - set(supported))
    if unsupported:
        raise ValueError(
            f"Thickness backend {backend_name!r} does not support boundary "
            f"mode(s) {', '.join(unsupported)}; supported modes: "
            f"{', '.join(supported)}."
        )


def x_face_velocities(velocity, left, right):
    """Interpolate an x velocity and impose its two normal boundaries."""
    if left == "periodic":
        edge = 0.5 * (velocity[:, -1:] + velocity[:, :1])
        left_face, right_face = edge, edge
    else:
        left_face = (
            tf.zeros_like(velocity[:, :1])
            if left == "symmetric"
            else velocity[:, :1]
        )
        right_face = (
            tf.zeros_like(velocity[:, -1:])
            if right == "symmetric"
            else velocity[:, -1:]
        )
    return tf.concat(
        [
            left_face,
            0.5 * (velocity[:, :-1] + velocity[:, 1:]),
            right_face,
        ],
        axis=1,
    )


def y_face_velocities(velocity, top, bottom):
    """Interpolate a y velocity and impose its two normal boundaries."""
    if top == "periodic":
        edge = 0.5 * (velocity[-1:, :] + velocity[:1, :])
        top_face, bottom_face = edge, edge
    else:
        top_face = (
            tf.zeros_like(velocity[:1, :])
            if top == "symmetric"
            else velocity[:1, :]
        )
        bottom_face = (
            tf.zeros_like(velocity[-1:, :])
            if bottom == "symmetric"
            else velocity[-1:, :]
        )
    return tf.concat(
        [
            top_face,
            0.5 * (velocity[:-1, :] + velocity[1:, :]),
            bottom_face,
        ],
        axis=0,
    )


def face_velocities(
    ubar,
    vbar,
    left="zero",
    right="zero",
    top="zero",
    bottom="zero",
):
    """Interpolate cell velocities to faces and impose the normal BC."""
    return (
        x_face_velocities(ubar, left, right),
        y_face_velocities(vbar, top, bottom),
    )


def pad_lines(lines, left, right):
    """Pad a batch of horizontal lines with one policy-consistent cell."""
    if left == "periodic":
        left_ghost, right_ghost = lines[:, -1:], lines[:, :1]
    else:
        left_ghost = (
            lines[:, :1] if left == "symmetric" else tf.zeros_like(lines[:, :1])
        )
        right_ghost = (
            lines[:, -1:]
            if right == "symmetric"
            else tf.zeros_like(lines[:, -1:])
        )
    return tf.concat([left_ghost, lines, right_ghost], axis=1)


def remove_nonperiodic_corner_couplings(
    west, east, north, south, left, right, top, bottom
):
    """Prepare stencil magnitudes for boundary-aware neighbor access.

    Periodic boundary coefficients retain their wrap-around coupling.  Other
    policies discard ghost couplings; open-boundary outflow remains encoded in
    the diagonal, while symmetric boundary velocities have already been set to
    zero by :func:`face_velocities`.
    """
    if left != "periodic":
        zero_column = tf.zeros_like(west[:, :1])
        west = tf.concat([zero_column, west[:, 1:]], axis=1)
    if right != "periodic":
        zero_column = tf.zeros_like(east[:, -1:])
        east = tf.concat([east[:, :-1], zero_column], axis=1)
    if top != "periodic":
        zero_row = tf.zeros_like(north[:1, :])
        north = tf.concat([zero_row, north[1:, :]], axis=0)
    if bottom != "periodic":
        zero_row = tf.zeros_like(south[-1:, :])
        south = tf.concat([south[:-1, :], zero_row], axis=0)
    return west, east, north, south


def neighbor_fields(field, left, right, top, bottom):
    """Return west/east/north/south neighbors with static boundary modes.

    The overwhelmingly common non-periodic path uses one padded tensor and
    four views, matching the memory behavior of the historical implicit
    operator. Periodic axes are extended only where their wrap values are
    needed; XLA can fuse the resulting concatenations into the stencil.
    """
    periodic_x = left == "periodic"
    periodic_y = top == "periodic"
    if not periodic_x and not periodic_y:
        padded = tf.pad(field, [[1, 1], [1, 1]])
        return (
            padded[1:-1, :-2],
            padded[1:-1, 2:],
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
        )

    if periodic_x:
        x_padded = tf.concat([field[:, -1:], field, field[:, :1]], axis=1)
    else:
        x_padded = tf.pad(field, [[0, 0], [1, 1]])
    if periodic_y:
        y_padded = tf.concat([field[-1:, :], field, field[:1, :]], axis=0)
    else:
        y_padded = tf.pad(field, [[1, 1], [0, 0]])
    return (
        x_padded[:, :-2],
        x_padded[:, 2:],
        y_padded[:-2, :],
        y_padded[2:, :],
    )
