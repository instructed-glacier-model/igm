#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Weertman sliding law.

Special case of the generalised power law with N ≡ 1, N_ref = 1, so
the entire shared math lives in `_power_law.py`. This wrapper just
unpacks fieldin and supplies a uniform N=1.
"""

import tensorflow as tf
from typing import Dict

from ..sliding import SlidingComponent
from igm.processes.iceflow.horizontal import HorizontalDiscr
from igm.processes.iceflow.vertical import VerticalDiscr

from ._power_law import power_law_cost


class WeertmanParams(tf.experimental.ExtensionType):
    """Parameters for Weertman sliding law."""

    regu: float
    exponent: float
    u_ref: float  # (m/yr)
    rho_ratio: float
    use_mask_gr: bool


class Weertman(SlidingComponent):
    """Sliding component implementing Weertman's sliding law."""

    def __init__(self, params: WeertmanParams) -> None:
        """Initialize Weertman sliding component with parameters."""
        self.name = "weertman"
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        discr_h: HorizontalDiscr,
        discr_v: VerticalDiscr,
    ) -> tf.Tensor:
        """Compute Weertman sliding cost."""
        return cost_weertman(U, V, fieldin, discr_h, discr_v, self.params)


def cost_weertman(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    weertman_params: WeertmanParams,
) -> tf.Tensor:
    """Compute Weertman sliding cost from field inputs (N ≡ 1)."""

    h = fieldin["thk"]
    s = fieldin["usurf"]
    tau_ref = fieldin["slidingco"]
    dx = fieldin["dX"]

    V_b = discr_v.V_b

    dtype = U.dtype
    m = tf.cast(weertman_params.exponent, dtype)
    u_regu = tf.cast(weertman_params.regu, dtype)
    u_ref = tf.cast(weertman_params.u_ref, dtype)
    rho_ratio = tf.cast(weertman_params.rho_ratio, dtype)
    use_mask_gr = tf.cast(weertman_params.use_mask_gr, tf.bool)

    # Weertman ≡ power law with N=1, N_ref=1, q=1 (N terms cancel)
    N = tf.ones_like(h)
    N_ref = tf.cast(1.0, dtype)
    q = tf.cast(1.0, dtype)

    return power_law_cost(
        U, V, h, s, tau_ref, N, dx, m, u_regu, u_ref, N_ref, q,
        rho_ratio, use_mask_gr, discr_h, V_b,
    )
