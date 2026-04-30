#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf


def correct_vertical_velocity(
    W: tf.Tensor,
    basal_melt_rate: tf.Tensor,
    correct_w_for_melt: bool = True,
) -> tf.Tensor:
    """
    Correct vertical velocity for basal melt contribution.

    Adjusts the vertical velocity field to account for the downward motion
    of the ice relative to the melting bed.

    Args:
        W: Vertical velocity field (m yr^-1).
        basal_melt_rate: Basal melt rate field (m ice yr^-1).
        correct_w_for_melt: Whether to apply the correction.

    Returns:
        Corrected vertical velocity field (m yr^-1).
    """
    if correct_w_for_melt:
        W -= tf.expand_dims(basal_melt_rate, axis=0)

    return W
