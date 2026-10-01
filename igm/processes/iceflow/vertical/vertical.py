#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Callable, Optional, Tuple

from igm.utils.math.precision import normalize_precision

from .enthalpy import VerticalDiscrEnthalpy, compute_discr_enthalpy


class VerticalDiscr(ABC):
    """
    Abstract vertical discretization for ice flow modeling.

    This class defines the vertical basis representation for 3D velocity fields.
    A field f(x,y,ζ) is represented as:

        f(x,y,ζ) = Σₙ fₙ(x,y) φₙ(ζ)

    where φₙ are the basis functions and fₙ are the DOF coefficients.

    Attributes
    ----------
    w : tf.Tensor
        Quadrature weights for vertical integration, shape (Nq,).
        Used to compute integrals: ∫₀¹ f dζ ≈ Σⱼ wⱼ f(ζⱼ)

    zeta : tf.Tensor
        Quadrature points in reference element [0, 1], shape (Nq,).
        ζ=0 corresponds to the bed, ζ=1 to the surface.

    V_q : tf.Tensor
        Evaluation matrix at quadrature points, shape (Nq, Ndof).
        Maps DOF coefficients to values: f(ζⱼ) = Σₙ V_q[j,n] fₙ

    V_q_grad : tf.Tensor
        Gradient matrix at quadrature points, shape (Nq, Ndof).
        Maps DOF coefficients to vertical gradients: ∂f/∂ζ(ζⱼ) = Σₙ V_q_grad[j,n] fₙ

    V_b : tf.Tensor
        Evaluation at bed (ζ=0), shape (Ndof,).
        Basal value: f(0) = Σₙ V_b[n] fₙ

    V_s : tf.Tensor
        Evaluation at surface (ζ=1), shape (Ndof,).
        Surface value: f(1) = Σₙ V_s[n] fₙ

    V_bar : tf.Tensor
        Vertical average operator, shape (Ndof,).
        Depth-averaged value: f̄ = ∫₀¹ f dζ = Σₙ V_bar[n] fₙ

    V_int : tf.Tensor
        Integration matrix, shape (Ndof, Ndof).
        Maps DOFs of f to DOFs of its vertical integral Φ(ζ) = ∫₀^ζ f(ζ') dζ'.
        Used for computing vertical velocity from horizontal divergence.

    V_corr_b : tf.Tensor
        Bed terrain correction matrix, shape (Ndof, Ndof).
        Maps DOFs of f to DOFs of ψᵇ(ζ) = ∫₀^ζ f'(ζ')(1-ζ') dζ'.
        Derived via integration by parts: ψᵇₙ(ζ) = (1-ζ)φₙ(ζ) - φₙ(0) + Φₙ(ζ)
        Used for terrain-following coordinate correction in vertical velocity.

    V_corr_s : tf.Tensor
        Surface terrain correction matrix, shape (Ndof, Ndof).
        Maps DOFs of f to DOFs of ψˢ(ζ) = ∫₀^ζ f'(ζ')ζ' dζ'.
        Derived via integration by parts: ψˢₙ(ζ) = ζ φₙ(ζ) - Φₙ(ζ)
        Used for terrain-following coordinate correction in vertical velocity.

    V_const : tf.Tensor
        Coefficients representing the constant function f(ζ) = 1, shape (Ndof,).

    enthalpy : Optional[VerticalDiscrEnthalpy]
        Enthalpy vertical discretization and coupling matrices (if enthalpy is enabled).
    """

    w: tf.Tensor
    zeta: tf.Tensor
    V_q: tf.Tensor
    V_q_grad: tf.Tensor
    V_b: tf.Tensor
    V_s: tf.Tensor
    V_bar: tf.Tensor
    V_int: tf.Tensor
    V_corr_b: tf.Tensor
    V_corr_s: tf.Tensor
    V_const: tf.Tensor
    enthalpy: Optional[VerticalDiscrEnthalpy]

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize vertical discretization."""
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype = normalize_precision(precision)
        basis_fct = self._compute_discr(cfg)
        self.basis_fct = basis_fct  # kept to evaluate fields at arbitrary ζ (e.g. flux-form vertical velocity)

        if "enthalpy" in cfg.processes:
            self.enthalpy = compute_discr_enthalpy(cfg, basis_fct, self.dtype)

    @abstractmethod
    def _compute_discr(
        self, cfg: DictConfig
    ) -> Tuple[Callable[[tf.Tensor], tf.Tensor], ...]:
        """Compute discretization matrices. Returns basis functions."""
        raise NotImplementedError(
            "❌ The discretization is not implemented in this class."
        )
