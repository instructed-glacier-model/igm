#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Source term of the thickness equation.

    dH/dt + div(H u) = smb + bmb

Both terms are in m ice eq. yr-1 and positive for a mass gain: ``smb`` is
the surface mass balance and ``bmb`` the basal mass balance, published by
the ``bmb`` process (ocean-induced melt under floating ice, thermodynamic
melt under grounded ice). Every transport and front backend receives the
source through :func:`mass_balance`, so the two can never be combined
differently by two schemes. ``bmb`` counts only when the ``bmb`` process
is active, so that a ``bmb`` field read from an input file, e.g. an earlier
IGM output, is never applied by mistake.
"""

import tensorflow as tf

from igm.common import State


def mass_balance(state: State) -> tf.Tensor:
    """Return ``smb``, plus ``bmb`` when the ``bmb`` process provides it.

    ``state.smb`` is created as zeros when no module provides it. Without
    ``bmb`` the returned tensor is ``state.smb`` itself, so a run without
    basal mass balance is bit-identical to one that predates it.
    """
    if not hasattr(state, "smb"):
        state.smb = tf.zeros_like(state.thk)
    components = getattr(state, "thk_components", None)
    if not getattr(components, "basal_mass_balance", False) or not hasattr(
        state, "bmb"
    ):
        return state.smb
    return state.smb + state.bmb
