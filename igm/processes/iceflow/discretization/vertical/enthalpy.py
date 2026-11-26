#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Callable, Tuple

import tensorflow as tf
from omegaconf import DictConfig

from .utils import compute_basis_matrix, compute_depth, compute_trap_quad, compute_zetas
from .utils_lagrange import compute_basis


class VerticalDiscrEnthalpy:
    """
    Enthalpy vertical discretization and coupling with velocity grid.

    Attributes
    ----------
    depth : tf.Tensor
        Normalized depth below ice surface at each level, shape (Ndof_E, 1, 1).
    weights : tf.Tensor
        Trapezoidal quadrature weights, shape (Ndof_E, 1, 1).
    zeta : tf.Tensor
        Normalized elevation of each node/level, shape (Ndof_E, 1, 1).
    dzeta : tf.Tensor
        Spacings between consecutive zeta values, shape (Ndof_E-1, 1, 1).
    V_U_to_E : tf.Tensor
        Map velocity DOFs → values at enthalpy nodes, shape (Ndof_E, Ndof_U).
    """

    depth: tf.Tensor
    weights: tf.Tensor
    zeta: tf.Tensor
    dzeta: tf.Tensor
    V_U_to_E: tf.Tensor


def compute_discr_enthalpy(
    cfg: DictConfig,
    basis_U: Tuple[Callable[[tf.Tensor], tf.Tensor], ...],
    dtype: tf.dtypes.DType = tf.float32,
) -> VerticalDiscrEnthalpy:

    cfg_numerics = cfg.processes.enthalpy.numerics
    Nz_E = cfg_numerics.Nz
    slope_init_E = 1.0 / cfg_numerics.vert_spacing

    zeta_E, _, dzeta_E = compute_zetas(Nz_E, slope_init_E, dtype)
    depth_E = compute_depth(dzeta_E)
    _, weights_E = compute_trap_quad(zeta_E, dzeta_E)

    dzeta_E = dzeta_E[..., None, None]
    depth_E = depth_E[..., None, None]
    weights_E = weights_E[..., None, None]

    V_U_to_E = compute_basis_matrix(basis_U, zeta_E)

    vertical_discr = VerticalDiscrEnthalpy()
    vertical_discr.depth = depth_E
    vertical_discr.weights = weights_E
    vertical_discr.zeta = zeta_E
    vertical_discr.dzeta = dzeta_E
    vertical_discr.V_U_to_E = V_U_to_E

    return vertical_discr
