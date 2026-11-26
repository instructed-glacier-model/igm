#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DebugState:
    iter: tf.Tensor
    costs: tf.Tensor
    grad_u: tf.Tensor
    grad_theta: list[tf.Tensor]
