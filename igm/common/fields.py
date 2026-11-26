#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Accessors for state fields with dual naming.

The ice flow creates `state.tau_ref` (the emulated/solved/diagnostic
methods, which used `state.slidingco`, have been removed). `get_tau_ref`
falls back to `state.slidingco` only for a state built without the ice flow.
"""

from igm.common import State


def get_tau_ref(state: State):
    """Reference basal shear stress field (MPa) at the sliding speed u_ref.

    Returns `state.tau_ref`, or `state.slidingco` if the state has no tau_ref.
    """
    return state.tau_ref if hasattr(state, "tau_ref") else state.slidingco
