#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from igm.common import State


def ice_base(state: State) -> tf.Tensor:
    """Elevation of the ice base: the bed under grounded ice, the draft under
    floating ice (``state.lsurf``, set by the ``thk`` process)."""
    return state.lsurf if hasattr(state, "lsurf") else state.topg
