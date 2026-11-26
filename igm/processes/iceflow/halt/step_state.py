#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from dataclasses import dataclass
from typing import Tuple


@dataclass
class StepState:
    """State at each iteration of the optimization process."""

    iter: tf.Tensor
    u: Tuple[tf.Tensor, tf.Tensor]
    theta: tf.Tensor
    cost: tf.Tensor
    grad_u_norm: tf.Tensor
    grad_theta_norm: tf.Tensor
