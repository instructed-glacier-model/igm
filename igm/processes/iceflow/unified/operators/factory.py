#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Select the Hessian operator for a velocity mapping from ``hvp_mode``."""

from typing import Callable

import tensorflow as tf

from .energy_operator import (
    ADOperator,
    BandedADOperator,
    MOLHOBandedADOperator,
    Operator,
    SSABandedADOperator,
)
from .molho_banded import supports_compact_molho
from .ssa_banded import supports_compact_ssa


def build_energy_operator(
    hvp_mode: str,
    probe_mode: str,
    basis_vertical: str,
    precision: str,
    cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    mapping,
    verify_stencil: bool = False,
    owner: str = "cg_newton",
    probe_batch: int = 0,
) -> Operator:
    """Build the Hessian operator selected by ``hvp_mode``.

    Autodiff is exact and general. Banded mode freezes a graph-coloured
    9-point stencil for cheap CG applications; nonperiodic SSA and Nz=2
    MOLHO use specialized compact storage. ``owner`` only labels errors.
    """
    hvp_mode = str(hvp_mode).lower()

    if hvp_mode == "autodiff":
        if int(probe_batch) != 0:
            raise ValueError("probe_batch requires hvp_mode='banded'.")
        return ADOperator(cost_fn, mapping, precision)

    if hvp_mode == "banded":
        basis_vertical = str(basis_vertical or "").lower()
        if supports_compact_ssa(mapping):
            operator_cls = SSABandedADOperator
        elif supports_compact_molho(mapping, basis_vertical):
            operator_cls = MOLHOBandedADOperator
        else:
            operator_cls = BandedADOperator

        operator_args = {
            "verify_stencil": bool(verify_stencil),
            "probe_mode": str(probe_mode),
        }
        if operator_cls is MOLHOBandedADOperator:
            operator_args["probe_batch"] = int(probe_batch)
        elif int(probe_batch) != 0:
            raise ValueError(
                "probe_batch is only supported by the MOLHO banded operator."
            )

        return operator_cls(
            cost_fn,
            mapping,
            precision,
            **operator_args,
        )

    raise ValueError(
        f"❌ Unknown {owner}.hvp_mode: <{hvp_mode}>. Use 'autodiff' or 'banded'."
    )
