#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Budd sliding law.

Same generalised power law as Weertman but with a non-trivial effective
pressure N read from `fieldin["effective_pressure"]`. The shared math
lives in `_power_law.py`.
"""

import tensorflow as tf
from typing import Dict

from ..sliding import SlidingComponent
from igm.processes.iceflow.horizontal import HorizontalDiscr
from igm.processes.iceflow.vertical import VerticalDiscr

from ._power_law import power_law_cost


class BuddParams(tf.experimental.ExtensionType):
    """Parameters for Budd sliding law."""

    regu: float
    exponent: float
    u_ref: float  # (m/yr)
    N_ref: float
    q_exponent: float  # effective-pressure exponent (1.0=linear, 0.5=Tsai)
    rho_ratio: float
    use_mask_gr: bool


class Budd(SlidingComponent):
    """Sliding component implementing Budd's sliding law."""

    def __init__(self, params: BuddParams) -> None:
        """Initialize Budd sliding component with parameters."""
        self.name = "budd"
        self.params = params

    def cost(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        fieldin: Dict[str, tf.Tensor],
        discr_h: HorizontalDiscr,
        discr_v: VerticalDiscr,
    ) -> tf.Tensor:
        """Compute Budd sliding cost."""
        return cost_budd(U, V, fieldin, discr_h, discr_v, self.params)


def cost_budd(
    U: tf.Tensor,
    V: tf.Tensor,
    fieldin: Dict[str, tf.Tensor],
    discr_h: HorizontalDiscr,
    discr_v: VerticalDiscr,
    budd_params: BuddParams,
) -> tf.Tensor:
    """Compute Budd sliding cost from field inputs.

    Reads effective pressure N from `fieldin["effective_pressure"]`. The
    presence of this field is enforced by the iceflow energy validator
    (utils.get_energy_components).
    """

    h = fieldin["thk"]
    s = fieldin["usurf"]
    tau_ref = fieldin["slidingco"]
    N = fieldin["effective_pressure"]
    dx = fieldin["dX"]

    V_b = discr_v.V_b

    dtype = U.dtype
    m = tf.cast(budd_params.exponent, dtype)
    u_regu = tf.cast(budd_params.regu, dtype)
    u_ref = tf.cast(budd_params.u_ref, dtype)
    N_ref = tf.cast(budd_params.N_ref, dtype)
    q = tf.cast(budd_params.q_exponent, dtype)
    rho_ratio = tf.cast(budd_params.rho_ratio, dtype)
    use_mask_gr = tf.cast(budd_params.use_mask_gr, tf.bool)

    # Floor N to avoid degenerate values in the cost (well-lubricated
    # bed at N≈0 is physically possible but numerically catastrophic).
    N = tf.where(N < tf.cast(1e-3, dtype), tf.cast(1e-3, dtype), N)

    return power_law_cost(
        U, V, h, s, tau_ref, N, dx, m, u_regu, u_ref, N_ref, q,
        rho_ratio, use_mask_gr, discr_h, V_b,
    )
