#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from typing import List, Tuple

from .mapping import Mapping
from ..bcs import BoundaryCondition
from .normalizer import IdentityNormalizer


class MappingIdentity(Mapping):
    def __init__(
        self,
        bcs: List[BoundaryCondition],
        U_guess: tf.Tensor,
        V_guess: tf.Tensor,
        precision: str = "float32",
    ):

        if U_guess.shape != V_guess.shape:
            raise ValueError("❌ U_guess and V_guess must have the same shape.")

        super().__init__(bcs, precision)
        self.shape = U_guess.shape
        self.type = U_guess.dtype
        self.U = tf.Variable(U_guess, trainable=True)
        self.V = tf.Variable(V_guess, trainable=True)
        self.input_normalizer = IdentityNormalizer()
        self.name = "identity"

    def get_UV_impl(self) -> Tuple[tf.Variable, tf.Variable]:
        return self.U, self.V

    @tf.function(reduce_retracing=True)
    def get_UV(self, inputs: tf.Tensor) -> Tuple[tf.Variable, tf.Variable]:
        U, V = self.get_UV_impl()
        for apply_bc in self.apply_bcs:
            U, V = apply_bc(U, V)
        return U, V

    def copy_theta(self, theta: list[tf.Variable]) -> list[tf.Tensor]:
        return [theta[0].read_value(), theta[1].read_value()]

    def copy_theta_flat(self, theta_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(theta_flat)

    def get_theta(self) -> list[tf.Variable]:
        return [self.U, self.V]

    def set_theta(self, theta: list[tf.Tensor]) -> None:
        self.U.assign(theta[0])
        self.V.assign(theta[1])

    def flatten_theta(self, theta: list[tf.Variable | tf.Tensor]) -> tf.Tensor:
        u_flat = tf.reshape(theta[0], [-1])
        v_flat = tf.reshape(theta[1], [-1])
        return tf.concat([u_flat, v_flat], axis=0)

    def unflatten_theta(self, theta_flat: tf.Tensor) -> list[tf.Tensor]:
        n = int(np.prod(self.shape))
        u_flat = theta_flat[:n]
        v_flat = theta_flat[n : 2 * n]
        U = tf.reshape(u_flat, self.shape)
        V = tf.reshape(v_flat, self.shape)
        return [U, V]
