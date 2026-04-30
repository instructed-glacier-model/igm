#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple


@tf.function
def compute_drainage(
    omega: tf.Tensor,
    omega_threshold_1: tf.Tensor,
    omega_threshold_2: tf.Tensor,
    omega_threshold_3: tf.Tensor,
) -> tf.Tensor:
    """
    TensorFlow function to compute drainage rate from water content.

    Uses a piecewise linear drainage function with three threshold levels.

    Args:
        omega: Water content fraction (-).
        omega_threshold_1: First water content threshold (-).
        omega_threshold_2: Second water content threshold (-).
        omega_threshold_3: Third water content threshold (-).

    Returns:
        Drainage rate (yr^-1).
    """
    result = tf.fill(tf.shape(omega), 0.05)
    result = tf.where(omega <= omega_threshold_3, 4.5 * omega - 0.085, result)
    result = tf.where(omega <= omega_threshold_2, 0.5 * omega - 0.005, result)
    result = tf.where(omega <= omega_threshold_1, 0.0, result)
    return result


@tf.function
def compute_fraction_drained(
    E: tf.Tensor,
    E_pmp: tf.Tensor,
    L_ice: tf.Tensor,
    omega_target: tf.Tensor,
    omega_threshold_1: tf.Tensor,
    omega_threshold_2: tf.Tensor,
    omega_threshold_3: tf.Tensor,
    rho_ice: tf.Tensor,
    rho_water: tf.Tensor,
    dz: tf.Tensor,
    dt: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    TensorFlow function to compute drained water fraction and thickness.

    Calculates the fraction of water content to drain from each cell and
    the total thickness of drained water integrated over the ice column.

    Args:
        E: Enthalpy field (J kg^-1).
        E_pmp: Pressure melting point enthalpy (J kg^-1).
        L_ice: Latent heat of fusion for ice (J kg^-1).
        omega_target: Target water content after drainage (-).
        omega_threshold_1: First water content threshold (-).
        omega_threshold_2: Second water content threshold (-).
        omega_threshold_3: Third water content threshold (-).
        rho_ice: Ice density (kg m^-3).
        rho_water: Water density (kg m^-3).
        dz: Vertical grid spacing field (m).
        dt: Time step (yr).

    Returns:
        Tuple of drained water fraction (-) and drained water thickness (m water).
    """
    # Water content
    omega = tf.maximum((E - E_pmp) / L_ice, 0.0)

    # Identify cells to drain
    DRAINED = omega > omega_target

    # Drainage fraction
    fraction_drained = (
        compute_drainage(
            omega,
            omega_threshold_1,
            omega_threshold_2,
            omega_threshold_3,
        )
        * dt
    )
    fraction_drained = tf.clip_by_value(
        fraction_drained, 0.0, (omega - omega_target) * rho_ice / rho_water
    )
    fraction_drained = tf.where(DRAINED, fraction_drained, 0.0)

    # Drained water thickness
    dz_centered = tf.concat([dz[0:1], dz[:-1] + dz[1:], dz[-1:]], axis=0) / 2.0
    h_drained = tf.reduce_sum(fraction_drained * dz_centered, axis=0)

    return fraction_drained, h_drained
