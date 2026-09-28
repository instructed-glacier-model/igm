#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Linearised freezing point of sea water."""

from typing import Tuple

import tensorflow as tf
from omegaconf import DictConfig


def freezing_coefficients(cfg: DictConfig) -> Tuple[float, float, float]:
    """``(lambda1, lambda2, lambda3)`` of ``cfg.processes.ocean.freezing_point``.

    By default the ISOMIP+ values of Asay-Davis et al. (2016); they are shared
    by the ocean thermal forcing and by the sub-shelf melt laws of the ``bmb``
    process.
    """
    p = cfg.processes.ocean.freezing_point
    return float(p.lambda1), float(p.lambda2), float(p.lambda3)


def freezing_point(
    salinity: tf.Tensor, z: tf.Tensor, coefficients: Tuple[float, float, float]
) -> tf.Tensor:
    """Freezing temperature T_f = lambda1 * S + lambda2 + lambda3 * z (°C).

    ``salinity`` is in g kg-1 and ``z`` is the depth relative to the water
    level (m, negative below it).
    """
    lambda1, lambda2, lambda3 = coefficients
    return lambda1 * salinity + lambda2 + lambda3 * z


def freezing_temperature(
    cfg: DictConfig, salinity: tf.Tensor, z: tf.Tensor
) -> tf.Tensor:
    """Freezing temperature with the coefficients of the configuration."""
    return freezing_point(salinity, z, freezing_coefficients(cfg))
