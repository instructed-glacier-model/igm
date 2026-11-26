#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf
from omegaconf import DictConfig

from igm.utils.math.precision import normalize_precision


class HorizontalDiscr(ABC):
    """Abstract base class for horizontal discretization schemes.

    This class defines the interface for computing horizontal gradients and
    field interpolations on a structured 2D grid.

    All discretization schemes transform field values defined at grid nodes
    (shape: batch, ..., Ny, Nx) into values at quadrature points within cells
    (shape: batch, Nq, ..., Ny-1, Nx-1), where Nq is the number of evaluation
    points per cell. The quadrature weights w_h are used for numerical
    integration over cells.

    Attributes:
        w_h: Quadrature weights of shape (Nq,).
    """

    w_h: tf.Tensor

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize horizontal discretization."""
        precision = cfg.processes.iceflow.numerics.precision
        self.dtype = normalize_precision(precision)
        self._compute_discr(cfg)

    @abstractmethod
    def _compute_discr(self, cfg: DictConfig) -> None:
        """Compute discretization attributes."""
        raise NotImplementedError(
            "❌ The discretization is not implemented in this class."
        )

    @abstractmethod
    def grad_h(self, X: tf.Tensor, dX: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Gradients at quadrature points.

        Input X: (batch, ...,  Ny, Nx)
        Input dX: (batch, Ny, Nx)
        Output: (batch, Nq, ..., Ny-1, Nx-1)
        """
        raise NotImplementedError(
            "❌ The horizontal gradient is not implemented in this class."
        )

    @abstractmethod
    def interp_h(self, X: tf.Tensor) -> tf.Tensor:
        """
        Interpolate to quadrature points.

        Input X: (batch, ...,  Ny, Nx)
        Input dX: (batch, Ny, Nx)
        Output: (batch, Nq, ..., Ny-1, Nx-1)
        """
        raise NotImplementedError(
            "❌ The interpolation is not implemented in this class."
        )
