#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


@tf.function()
def compute_basal_melt_rate(
    E: tf.Tensor,
    E_pmp: tf.Tensor,
    E_s: tf.Tensor,
    h_water_till: tf.Tensor,
    q_basal: tf.Tensor,
    k_ice: tf.Tensor,
    c_ice: tf.Tensor,
    K_ratio: tf.Tensor,
    rho_ice: tf.Tensor,
    L_ice: tf.Tensor,
    dz: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute the basal melt rate from the heat-flux imbalance.

        Mb = (q_basal - q_ice) / (rho_ice * (1 - omega_basal) * L_ice)

    q_ice uses cold-ice diffusivity K_c = k_ice/c_ice, or K_ratio*K_c when the
    first interior layer is temperate (Aschwanden et al. 2012, Eq. 3 & 11).
    Cold-and-dry beds yield Mb = 0; dry beds with Mb < 0 are clamped to 0.

    Args:
        E:            3-D enthalpy field (nz, ny, nx) (J kg^-1).
        E_pmp:        3-D pressure-melting-point enthalpy, same shape as E (J kg^-1).
        E_s:          Surface enthalpy, 2-D (J kg^-1).
        h_water_till: Till water thickness, 2-D (m).
        q_basal:      Total basal heat flux (geothermal + friction), 2-D (W m^-2).
        k_ice:        Ice thermal conductivity (W m^-1 K^-1).
        c_ice:        Ice specific heat capacity (J kg^-1 K^-1).
        K_ratio:      Temperate-to-cold diffusivity ratio (-).
        rho_ice:      Ice density (kg m^-3).
        L_ice:        Latent heat of fusion (J kg^-1).
        dz:           Basal layer thickness, 2-D (m).

    Returns:
        Basal melt rate, 2-D (m ice yr^-1).
    """
    shape_2d = E_s.shape

    COLD_AND_DRY = (E[0] < E_pmp[0]) & (h_water_till <= 0.0)
    ICE_COLD = E[1] < E_pmp[1]
    IS_DRY = h_water_till <= 0.0

    # Water fraction at the base (zero for cold ice)
    omega_basal = tf.maximum(0.0, (E[0] - E_pmp[0]) / L_ice)

    # Conductive flux into ice (positive = upward, away from bed)
    q_ice = tf.where(
        ICE_COLD,
        -(k_ice / c_ice) * (E[1] - E[0]) / dz,
        -K_ratio * (k_ice / c_ice) * (E[1] - E[0]) / dz,
    )

    # Melt rate from heat-balance residual
    basal_melt_rate = tf.where(
        COLD_AND_DRY,
        tf.zeros(shape_2d),
        (q_basal - q_ice) / (rho_ice * (1.0 - omega_basal) * L_ice),
    )

    # No refreezing on a dry bed (no liquid available)
    basal_melt_rate = tf.where(
        IS_DRY & (basal_melt_rate < 0.0),
        tf.zeros(shape_2d),
        basal_melt_rate,
    )

    spy = 31556926.0
    basal_melt_rate = basal_melt_rate * spy

    return basal_melt_rate
