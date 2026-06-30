#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Stack-agnostic accessors for state fields with dual naming.

During the slidingco → tau_ref migration, the legacy stack
(emulated/solved/diagnostic + data_assimilation) keeps `state.slidingco`
while the new stack (unified + field_inversion + pretraining)
uses `state.tau_ref`. Modules that need to read this field
stack-agnostically (enthalpy/till, stress diagnostics, NetCDF outputs)
go through `get_tau_ref(state)`.

When the legacy stack is retired, this helper collapses to a direct
read of `state.tau_ref` and is inlined.
"""

from igm.common import State


def get_tau_ref(state: State):
    """Reference basal shear stress field (MPa), regardless of stack.

    Returns `state.tau_ref` on the new stack (iceflow.method == unified)
    and `state.slidingco` on the legacy stack.
    """
    return state.tau_ref if hasattr(state, "tau_ref") else state.slidingco
