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
    force_zero_sum: bool = False  # divfluxfcz only: also drive the mean divflux to zero


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


class DivfluxFczMisfitTerm(CostTerm):
    """Port of the legacy data_assimilation `divfluxfcz` cost.

    Penalizes the residual of the modelled flux divergence around its own
    best linear-in-elevation fit (an SMB-shaped apparent-mass-balance proxy)
    — no observation field required. Unlike the legacy module (scipy
    regression outside the tape, i.e. a stop-gradient target refreshed every
    10 Adam iterations), the closed-form least-squares fit here is fully
    DIFFERENTIABLE: the cost is the exact projection residual, so objective
    and gradient stay consistent — required by the L-BFGS line search, which
    chronically fails with an inconsistent (stop-gradient) target. This term
    couples every thickness pixel to the flux field, which both regularizes
    divflux (key for shock-free transient starts) and propagates thickness
    information beyond the observation footprints.

    Spec usage: `std` plays the role of the legacy `divfluxobs_std`;
    `components`/`obs` are unused; `force_zero_sum` adds the legacy optional
    penalty driving the mask-mean divflux to zero.
    """

    group = Misfit

    def __init__(self, spec: MisfitSpec) -> None:
        self.spec = spec
        self.name = f"misfit:{spec.name}"

    def cost(self, ctx: DAEvaluationContext) -> tf.Tensor:
        dtype = ctx.dtype
        std = tf.cast(self.spec.std, dtype)
        eps = tf.cast(self.spec.eps, dtype)

        divflux = tf.cast(ctx.divflux(), dtype)
        # Elevation regressor: θ-dependent field if usurf is inverted for,
        # otherwise the static state field.
        try:
            s = ctx.physical("usurf")
        except ValueError:
            s = ctx.state_field("usurf")
        s = tf.cast(s, dtype)

        mask = ctx.get_mask(self.spec.mask) if self.spec.mask is not None else ctx.get_mask(None)
        m = tf.cast(mask, dtype)
        n = tf.reduce_sum(m) + eps

        # Closed-form linear regression of divflux on elevation over the mask.
        # The regressor (elevation) enters as a constant; the regressand
        # (divflux) stays differentiable so the cost is the exact projection
        # residual with consistent gradients.
        s_c = tf.stop_gradient(s)
        s_mean = tf.reduce_sum(m * s_c) / n
        d_mean = tf.reduce_sum(m * divflux) / n
        cov = tf.reduce_sum(m * (s_c - s_mean) * (divflux - d_mean)) / n
        var = tf.reduce_sum(m * tf.square(s_c - s_mean)) / n + eps
        slope = cov / var
        intercept = d_mean - slope * s_mean
        target = intercept + slope * s_c

        res2 = tf.square((target - divflux) / std)
        integral = masked_integral(res2, mask, ctx.dx)
        denom = masked_area(mask, ctx.dx, res2) + eps
        cost = tf.cast(0.5, dtype) * integral / denom

        if self.spec.force_zero_sum:
            div_mean = tf.reduce_sum(m * divflux) / n
            cost += tf.cast(0.5 * 1000.0, dtype) * tf.square(div_mean / std)

        return cost


MisfitRegistry = {
    "gaussian": GaussianMisfitTerm,
    "huber": HuberMisfitTerm,
    "divfluxfcz": DivfluxFczMisfitTerm,
}

# Misfit kinds that synthesize their own target and take no `obs` fields.
TARGET_FREE_MISFITS = {"divfluxfcz"}
