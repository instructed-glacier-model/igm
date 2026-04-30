#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple


@tf.function()
def compute_bc(
    E: tf.Tensor,
    E_pmp: tf.Tensor,
    E_s: tf.Tensor,
    h_water_till: tf.Tensor,
    dEdz_dry: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    TensorFlow function to compute basal and surface boundary conditions.

    Selects Neumann or Dirichlet at the base following Aschwanden et al. (2012) Fig. 5,
    based on thermal state (cold/temperate) and till water availability (dry/wet):

        BASE_COLD (E[0] < E_pmp or dry):
          dry -> Neumann   dE/dz = dEdz_dry  (geothermal flux, Eq. 49)
          wet -> Dirichlet E = E_pmp         (cold-wet clamp)
        BASE_TEMPERATE (E[0] >= E_pmp and wet):
          cold layer above      -> Dirichlet E = E_pmp
          temperate layer above -> Neumann   dE/dz = 0

    A dry bed is always treated as cold; the Neumann BC may let E[0] exceed E_pmp
    by one step — the caller is responsible for clamping (Sec. 4.7).

    Args:
        E:            3-D enthalpy field (nz, ny, nx) (J kg^-1); E[0] basal, E[1] first interior.
        E_pmp:        3-D pressure-melting-point enthalpy, same shape as E (J kg^-1).
        E_s:          Surface enthalpy, 2-D (J kg^-1).
        h_water_till: Till water thickness, 2-D (m).
        dEdz_dry:     Enthalpy gradient for the cold-dry Neumann BC, 2-D (J kg^-1 m^-1).

    Returns:
        BCB: Basal BC type (1 = Neumann, 0 = Dirichlet), 2-D.
        VB:  Basal BC value (gradient or enthalpy), 2-D.
        VS:  Surface Dirichlet value = E_s, 2-D (J kg^-1).
    """
    shape_2d = E_s.shape

    IS_DRY = h_water_till <= 0.0
    IS_COLD = E[0] < E_pmp[0]
    ICE_COLD = E[1] < E_pmp[1]

    BASE_COLD = IS_COLD | IS_DRY

    BCB = tf.where(
        BASE_COLD,
        tf.where(IS_DRY, tf.ones(shape_2d), tf.zeros(shape_2d)),
        tf.where(ICE_COLD, tf.zeros(shape_2d), tf.ones(shape_2d)),
    )

    VB = tf.where(
        BASE_COLD,
        tf.where(IS_DRY, dEdz_dry, E_pmp[0]),
        tf.where(ICE_COLD, E_pmp[0], 0.0),
    )

    VS = E_s

    return BCB, VB, VS
