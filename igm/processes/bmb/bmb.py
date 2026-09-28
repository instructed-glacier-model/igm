#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
bmb
===

Basal mass balance, the basal source term of the thickness equation

    dH/dt + div(H u) = smb + bmb.

Under grounded ice it is minus the thermodynamic melt of the ``enthalpy``
process (``state.basal_melt_rate``, used when present and
``include_grounded_melt``); under floating ice exposed to the ocean it is
minus the sub-shelf melt rate of the law ``cfg.processes.bmb.method``:

    zero         no ocean melt (a land-only run: bmb = -basal_melt_rate)
    prescribed   constant, 2-D field, or MISMIP+ melt
    quadratic    local or non-local quadratic thermal forcing (ISMIP6)
    pico         PICO box model (Reese et al., 2018)
    picop        PICO box properties with a plume melt (Pelle et al., 2019)
    plume        buoyant-plume melt (Lazeroms et al., 2019)

Across the grounding line both are weighted according to
``grounding_line.treatment`` (nmp, fmp or pmp; see :mod:`.grounding_line`):

    bmb = -(g m_grounded + w m_ocean)   where there is ice, 0 elsewhere.

Published fields, in m ice eq. yr-1:

    state.bmb                basal mass balance, positive for a gain (like smb)
    state.shelf_melt_rate    sub-shelf melt rate of the law, positive for melt
    state.grounded_fraction  grounded fraction of each node's cell (-)

The melt law is evaluated every ``update_freq`` years; the geometry, the
weights and ``bmb`` are updated every step. Between two evaluations, a node
that has become floating takes the melt of its shelf neighbours at the last
evaluation. Run ``bmb`` after ``enthalpy`` (and after ``iceflow`` for picop
and plume) and right before ``thk``.
"""

from types import ModuleType

import tensorflow as tf
from omegaconf import DictConfig

from igm.common import State

from .geometry import compute_geometry
from .grounding_line import TREATMENTS, basal_mass_balance, extend
from .laws import get_melt_law


def get_active_submodule(cfg: DictConfig) -> ModuleType:
    """The melt law, whose metadata lists the state variables it reads."""
    return get_melt_law(cfg)[1]


def initialize(cfg: DictConfig, state: State) -> None:
    p = cfg.processes.bmb
    if "thk" not in cfg.processes:
        raise ValueError(
            "The bmb process needs the 'thk' process, whose flotation and "
            "thickness evolution it feeds."
        )
    if p.grounding_line.treatment not in TREATMENTS:
        raise ValueError(
            "cfg.processes.bmb.grounding_line.treatment = "
            f"{p.grounding_line.treatment!r} is not one of {', '.join(TREATMENTS)}."
        )
    if int(p.grounding_line.sub_samples) < 1:
        raise ValueError("cfg.processes.bmb.grounding_line.sub_samples must be >= 1.")
    _, law = get_melt_law(cfg)
    if hasattr(law, "initialize"):
        law.initialize(cfg, state)

    for name in ("bmb", "shelf_melt_rate", "shelf_melt_cache", "grounded_fraction"):
        setattr(state, name, tf.zeros_like(state.thk))
    state.tlast_bmb = tf.Variable(float("-inf"), trainable=False)


def update(cfg: DictConfig, state: State) -> None:
    p = cfg.processes.bmb
    geom = compute_geometry(cfg, state)

    # Without update_freq, skip the time test: it would wait for the device.
    if p.update_freq <= 0.0 or state.t - state.tlast_bmb >= p.update_freq:
        _, law = get_melt_law(cfg)
        state.shelf_melt_cache = _ocean_melt(
            law.melt_rate(cfg, state, geom),
            geom.shelf,
            float(p.melt_enhancer),
            bool(p.allow_refreezing),
        )
        state.tlast_bmb.assign(state.t)

    if p.include_grounded_melt and hasattr(state, "basal_melt_rate"):
        grounded_melt = state.basal_melt_rate
    else:
        grounded_melt = tf.zeros_like(geom.thk)
    state.bmb, state.grounded_fraction, state.shelf_melt_rate = basal_mass_balance(
        p.grounding_line.treatment,
        p.grounding_line.sub_samples,
        geom,
        state.shelf_melt_cache,
        grounded_melt,
    )


def finalize(cfg: DictConfig, state: State) -> None:
    pass


@tf.function(autograph=False, jit_compile=True)
def _ocean_melt(
    melt: tf.Tensor, shelf: tf.Tensor, enhancer: float, allow_refreezing: bool
) -> tf.Tensor:
    """Law output on the shelf, extended to its neighbours (see ``update``)."""
    melt = enhancer * melt
    if not allow_refreezing:
        melt = tf.maximum(melt, 0.0)
    return extend(tf.where(shelf, melt, 0.0), shelf)
