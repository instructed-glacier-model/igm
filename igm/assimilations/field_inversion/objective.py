#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import tensorflow as tf

from igm.utils.math.precision import normalize_precision

from .context import DAEvaluationContext
from .terms import (
    CostTerm,
    FieldPenaltySpec,
    FieldPenaltyTerm,
    Misfit,
    MisfitRegistry,
    MisfitSpec,
    Regularization,
)
from .utils import _as_list


class DAObjective:
    def __init__(self, cfg: Any, state: Any, da_map: Any, terms: Sequence[CostTerm]) -> None:
        self.cfg = cfg
        self.state = state
        self.da_map = da_map
        self.terms: List[CostTerm] = list(terms)

        self._misfit_idx = [i for i, t in enumerate(self.terms) if t.group == Misfit]
        self._reg_idx = [i for i, t in enumerate(self.terms) if t.group == Regularization]

    def __call__(
        self,
        U: tf.Tensor,
        V: tf.Tensor,
        inputs,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        dtype = normalize_precision(self.cfg.processes.iceflow.numerics.precision)
        ctx = DAEvaluationContext(
            cfg=self.cfg,
            state=self.state,
            da_map=self.da_map,
            U_in=U,
            V_in=V,
            inputs=inputs,
        )

        ta = tf.TensorArray(dtype=dtype, size=len(self.terms), element_shape=())
        for i, term in enumerate(self.terms):
            ta = ta.write(i, tf.cast(term.cost(ctx), dtype))

        term_costs = ta.stack()
        total = tf.reduce_sum(term_costs)
        misfit = (
            tf.reduce_sum(tf.gather(term_costs, self._misfit_idx))
            if self._misfit_idx
            else tf.zeros((), dtype=dtype)
        )
        reg = (
            tf.reduce_sum(tf.gather(term_costs, self._reg_idx))
            if self._reg_idx
            else tf.zeros((), dtype=dtype)
        )

        return total, misfit, reg, term_costs


def build_objective_from_cfg(cfg: Any, state: Any, da_map: Any) -> DAObjective:
    """Build DAObjective from misfit and regularization terms defined in cfg."""

    obj_cfg = cfg.assimilations.field_inversion.objective
    misfit_list = list(obj_cfg.misfit or [])
    reg_list = list(obj_cfg.regularization or [])

    terms: List[CostTerm] = []

    for item in misfit_list:
        d = dict(item)
        kind = str(d.get("kind", "gaussian"))
        name = str(d["name"])
        components = [str(s) for s in _as_list(d.get("components", [name]))]
        obs = [str(s) for s in _as_list(d["obs"])]

        if len(components) != len(obs):
            raise ValueError(
                f"Misfit '{name}' has {len(components)} components but {len(obs)} observations."
            )
        if kind not in MisfitRegistry:
            raise ValueError(f"Unknown misfit '{kind}'. Available: {list(MisfitRegistry.keys())}")

        terms.append(
            MisfitRegistry[kind](
                MisfitSpec(
                    name=name,
                    components=components,
                    obs=obs,
                    std=float(d["std"]),
                    mask=None if d.get("mask") is None else str(d["mask"]),
                    eps=float(d.get("eps", 1e-12)),
                )
            )
        )

    for item in reg_list:
        d = dict(item)
        ref = d.get("prior", None)
        if ref is None:
            ref = d.get("ref", None)
        terms.append(
            FieldPenaltyTerm(
                FieldPenaltySpec(
                    name=str(d["name"]),
                    penalty=str(d["penalty"]),
                    lam=float(d["lam"]),
                    mask=None if d.get("mask") is None else str(d["mask"]),
                    eps=float(d.get("eps", 1e-12)),
                    ref=None if ref is None else str(ref),
                )
            )
        )

    if not terms:
        raise ValueError("Objective has zero terms. Define at least one misfit or regularization term.")

    return DAObjective(cfg=cfg, state=state, da_map=da_map, terms=terms)
