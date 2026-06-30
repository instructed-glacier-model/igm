#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import tensorflow as tf

from .context import DAEvaluationContext
from .penalties import PenaltyRegistry
from .utils import masked_area, masked_integral


Misfit = "misfit"
Regularization = "regularization"


@dataclass(frozen=True)
class MisfitSpec:
    name: str
    components: Sequence[str]
    obs: Sequence[str]
    std: float
    mask: Optional[str] = None
    eps: float = 1e-12
    delta: float = 1.0  # Huber transition (σ-units); ignored by gaussian misfit


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
    group: str

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
        obs_fields = [ctx.state_field(obs_name) for obs_name in self.spec.obs]

        # Misfit defaults to icemask. An explicit mask name must exist on state.
        mask = ctx.get_mask(self.spec.mask) if self.spec.mask is not None else ctx.get_mask(None)

        # Observation NaNs mean "no data" and are excluded before residual arithmetic.
        for y in obs_fields:
            mask = mask & tf.math.is_finite(y)

        res2 = None
        for comp_name, y in zip(self.spec.components, obs_fields):
            m = ctx.model(comp_name)
            y_eff = tf.where(mask, y, tf.zeros_like(y))
            m_eff = tf.where(mask, m, tf.zeros_like(m))
            term = tf.square(tf.cast((y_eff - m_eff) / std, dtype))
            res2 = term if res2 is None else res2 + term

        integral = masked_integral(tf.cast(res2, dtype), mask, ctx.dx)
        denom = masked_area(mask, ctx.dx, res2) + tf.cast(self.spec.eps, dtype)
        return tf.cast(0.5, dtype) * integral / denom


class FieldPenaltyTerm(CostTerm):
    group = Regularization

    def __init__(self, spec: FieldPenaltySpec) -> None:
        if spec.penalty not in PenaltyRegistry:
            raise ValueError(
                f"Unknown penalty '{spec.penalty}'. Available: {list(PenaltyRegistry.keys())}"
            )
        self.spec = spec
        self.name = f"reg:{spec.name}:{spec.penalty}"

    def cost(self, ctx: DAEvaluationContext) -> tf.Tensor:
        dtype = ctx.dtype
        field = ctx.physical(self.spec.name)  # ensures tape tracks θ
        lam = tf.cast(self.spec.lam, dtype)

        # Regularization defaults to the full tensor domain. An explicit mask
        # name must exist on state.
        mask = (
            ctx.get_mask(self.spec.mask)
            if self.spec.mask is not None
            else tf.ones_like(field, dtype=tf.bool)
        )

        ref_tensor = None
        if self.spec.ref is not None:
            ref_tensor = ctx.state_field(self.spec.ref)
            mask = mask & tf.math.is_finite(ref_tensor)

        fn = PenaltyRegistry[self.spec.penalty]
        return fn(
            field=field,
            dx=ctx.dx,
            lam=lam,
            mask=mask,
            area=masked_area(mask, ctx.dx, field),
            eps=float(self.spec.eps),
            ref=ref_tensor,
        )


class HuberMisfitTerm(CostTerm):
    """Robust vector misfit: quadratic for residual norms below `delta`
    (in σ-units), linear beyond. Caps each pixel's gradient contribution at
    `delta` so observational outliers can't dominate the inversion. Reduces to
    the Gaussian misfit as delta -> inf. Huber is applied to the residual
    *norm* (isotropic in component space), not per-component."""

    group = Misfit

    def __init__(self, spec: MisfitSpec) -> None:
        self.spec = spec
        self.name = f"misfit:{spec.name}"

    def cost(self, ctx: DAEvaluationContext) -> tf.Tensor:
        dtype = ctx.dtype
        std = tf.cast(self.spec.std, dtype)
        delta = tf.cast(self.spec.delta, dtype)
        eps = tf.cast(self.spec.eps, dtype)
        obs_fields = [ctx.state_field(obs_name) for obs_name in self.spec.obs]

        # Misfit defaults to icemask. An explicit mask name must exist on state.
        mask = ctx.get_mask(self.spec.mask) if self.spec.mask is not None else ctx.get_mask(None)

        # Observation NaNs mean "no data" and are excluded before residual arithmetic.
        for y in obs_fields:
            mask = mask & tf.math.is_finite(y)

        # Squared residual norm summed across components, in σ-units.
        res2 = None
        for comp_name, y in zip(self.spec.components, obs_fields):
            m = ctx.model(comp_name)
            y_eff = tf.where(mask, y, tf.zeros_like(y))
            m_eff = tf.where(mask, m, tf.zeros_like(m))
            term = tf.square(tf.cast((y_eff - m_eff) / std, dtype))
            res2 = term if res2 is None else res2 + term

        # Isotropic Huber on the residual norm r = ||residual||. eps inside the
        # sqrt keeps the gradient finite at r = 0.
        r = tf.sqrt(res2 + eps)
        quadratic = tf.cast(0.5, dtype) * tf.square(r)
        linear = delta * (r - tf.cast(0.5, dtype) * delta)
        rho = tf.where(r <= delta, quadratic, linear)

        integral = masked_integral(tf.cast(rho, dtype), mask, ctx.dx)
        denom = masked_area(mask, ctx.dx, rho) + eps
        return integral / denom


MisfitRegistry = {
    "gaussian": GaussianMisfitTerm,
    "huber": HuberMisfitTerm,
}
