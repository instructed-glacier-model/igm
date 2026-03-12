#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, List, Sequence, Tuple
import tensorflow as tf
from dataclasses import dataclass

from .context import DAEvaluationContext
from igm.utils.math.precision import normalize_precision
from .utils import _as_list
from .terms import (
    CostTerm,
    MisfitSpec,
    MisfitRegistry,
    FieldPenaltySpec,
    FieldPenaltyTerm,
    Misfit,
    Regularization
)


class DAObjective:

    def __init__(self, cfg: Any, state: Any, da_map: Any, terms: Sequence[CostTerm]) -> None:
        self.cfg = cfg
        self.state = state
        self.da_map = da_map
        self.terms: List[CostTerm] = list(terms)

        self.term_names = [t.name for t in self.terms]
        self._misfit_idx = [i for i, t in enumerate(self.terms) if t.group == Misfit]
        self._reg_idx = [i for i, t in enumerate(self.terms) if t.group == Regularization]

    def __call__(self, U: tf.Tensor, V: tf.Tensor, inputs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        dtype = normalize_precision(self.cfg.processes.iceflow.numerics.precision)

        ctx = DAEvaluationContext(cfg=self.cfg, state=self.state, da_map=self.da_map, U_in=U, V_in=V, inputs=inputs)

        ta = tf.TensorArray(dtype=dtype, size=len(self.terms), element_shape=())
        for i, term in enumerate(self.terms):
            ci = term.cost(ctx)
            ta = ta.write(i, tf.cast(ci, dtype))

        term_costs = ta.stack()  # (n_terms,)
        total = tf.reduce_sum(term_costs)

        misfit = tf.reduce_sum(tf.gather(term_costs, self._misfit_idx)) if self._misfit_idx else tf.zeros((), dtype=dtype)
        reg = tf.reduce_sum(tf.gather(term_costs, self._reg_idx)) if self._reg_idx else tf.zeros((), dtype=dtype)

        return total, misfit, reg, term_costs


def build_objective_from_cfg(cfg: Any, state: Any, da_map: Any) -> DAObjective:
    """
    Builds DAObjective from misfit and regularization terms defined in cfg
    """

    cfg_da = cfg.processes.data_assimilation_SR
    obj_cfg = cfg_da.objective

    misfit_list = list(getattr(obj_cfg, "misfit", []) or [])
    reg_list = list(getattr(obj_cfg, "regularization", []) or [])

    terms: List[CostTerm] = []

    # ---- MISFIT ----
    for item in misfit_list:
        d = dict(item)

        kind = str(d.get("kind", "gaussian"))
        name = str(d["name"])

        components = [str(s) for s in _as_list(d.get("components", [name]))]
        obs = [str(s) for s in _as_list(d["obs"])]

        spec = MisfitSpec(
            kind=kind,
            name=name,
            components=components,
            obs=obs,
            std=float(d["std"]),
            mask=str(d["mask"]) if "mask" in d else None,
            eps=float(d.get("eps", 1e-12)),
        )

        TermCls = MisfitRegistry[kind]
        terms.append(TermCls(spec))

    # ---- REGULARIZATION ----
    for item in reg_list:
        d = dict(item)

        ref = d.get("prior", None)
        if ref is None:
            ref = d.get("ref", None)

        spec = FieldPenaltySpec(
            name=str(d["name"]),
            penalty=str(d["penalty"]),
            lam=float(d["lam"]),
            mask=str(d["mask"]) if "mask" in d else None,
            eps=float(d.get("eps", 1e-12)),
            ref=str(ref) if ref is not None else None,
        )
        terms.append(FieldPenaltyTerm(spec))

    if not terms:
        raise ValueError("Objective has zero terms. Define at least one misfit or regularization term.")

    return DAObjective(cfg=cfg, state=state, da_map=da_map, terms=terms)
