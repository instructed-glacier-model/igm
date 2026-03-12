#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import tensorflow as tf

from .context import DAEvaluationContext
from .penalties import PenaltyRegistry
from .utils import masked_integral


Misfit = "misfit"
Regularization = "regularization"


@dataclass(frozen=True)
class MisfitSpec:
    kind: str
    name: str
    components: Sequence[str]
    obs: Sequence[str]
    std: float
    mask: Optional[str] = None
    eps: float = 1e-12


@dataclass(frozen=True)
class FieldPenaltySpec:
    name: str
    penalty: str
    lam: float
    mask: Optional[str] = None
    eps: float = 1e-12
    ref: Optional[str] = None


class CostTerm(ABC):
    name: str
    group: str  # MISFIT / REGULARIZATION

    @abstractmethod
    def cost(self, ctx: DAEvaluationContext) -> tf.Tensor:
        raise NotImplementedError


class GaussianMisfitTerm(CostTerm):
    group = Misfit

    def __init__(self, spec: MisfitSpec) -> None:
        self.spec = spec
        self.name = f"misfit:{spec.name}"

    def cost(self, ctx: DAEvaluationContext) -> tf.Tensor:
        dtype = ctx.dtype
        std = tf.cast(self.spec.std, dtype)

        # Base mask (default: icemask unless user overrides)
        mask = ctx.get_mask(self.spec.mask)

        # Observation availability: NaN means "no data"
        for obs_name in self.spec.obs:
            y = ctx.state_field(obs_name)
            mask = mask & tf.math.is_finite(y)

        res2 = None
        for comp_name, obs_name in zip(self.spec.components, self.spec.obs):
            y = ctx.state_field(obs_name)
            m = ctx.model(comp_name)
            r = (y - m) / std
            r = tf.where(mask, r, tf.zeros_like(r))  # kill NaNs that would pollute gradients
            term = tf.square(tf.cast(r, dtype))
            res2 = term if res2 is None else (res2 + term)

        integral = masked_integral(tf.cast(res2, dtype), mask, ctx.dx)
        denom = tf.cast(ctx.A_domain, dtype) + tf.cast(self.spec.eps, dtype)
        return tf.cast(0.5, dtype) * integral / denom


class HuberMisfitTerm(CostTerm):
    group = Misfit

    def __init__(self, spec: MisfitSpec) -> None:
        self.spec = spec
        self.name = f"misfit:{spec.name}"

    def cost(self, ctx: DAEvaluationContext) -> tf.Tensor:
        dtype = ctx.dtype
        std = tf.cast(self.spec.std, dtype)
        delta = tf.cast(1.0, dtype)  # fixed for now
        mask = ctx.get_mask(self.spec.mask)

        for obs_name in self.spec.obs:
            y = ctx.state_field(obs_name)  # <- fixed (no ctx.obs method)
            mask = mask & tf.math.is_finite(y)

        res = None
        for comp_name, obs_name in zip(self.spec.components, self.spec.obs):
            y = ctx.state_field(obs_name)
            m = ctx.model(comp_name)
            r = (y - m) / std
            r = tf.where(mask, r, tf.zeros_like(r))
            a = tf.abs(tf.cast(r, dtype))
            term = tf.where(a <= delta, 0.5 * tf.square(a), delta * (a - 0.5 * delta))
            res = term if res is None else (res + term)

        integral = masked_integral(tf.cast(res, dtype), mask, ctx.dx)
        denom = tf.cast(ctx.A_domain, dtype) + tf.cast(self.spec.eps, dtype)
        return integral / denom


class FieldPenaltyTerm(CostTerm):
    group = Regularization

    def __init__(self, spec: FieldPenaltySpec) -> None:
        if spec.penalty not in PenaltyRegistry:
            raise ValueError(f"Unknown penalty '{spec.penalty}'. Available: {list(PenaltyRegistry.keys())}")
        self.spec = spec
        self.name = f"reg:{spec.name}:{spec.penalty}"

    def cost(self, ctx: DAEvaluationContext) -> tf.Tensor:
        dtype = ctx.dtype

        field = ctx.physical(self.spec.name)  # ensures tape tracks θ
        lam = tf.cast(self.spec.lam, dtype)
        mask = ctx.get_mask(self.spec.mask)  # None => icemask

        ref_tensor = None
        if self.spec.ref is not None:
            ref_tensor = ctx.state_field(self.spec.ref)

        fn = PenaltyRegistry[self.spec.penalty]
        return fn(
            field=field,
            dx=ctx.dx,
            lam=lam,
            mask=mask,
            A_domain=tf.cast(ctx.A_domain, dtype),
            eps=float(self.spec.eps),
            ref=ref_tensor,
        )


MisfitRegistry = {
    "gaussian": GaussianMisfitTerm,
    "huber": HuberMisfitTerm,
}
