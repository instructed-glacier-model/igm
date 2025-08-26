#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple

from .mapping import Mapping


class MappingNodal(Mapping):
    def __init__(self, U_guess: tf.Tensor, V_guess: tf.Tensor):

        if U_guess.shape != V_guess.shape:
            raise ValueError("❌ U_guess and V_guess must have the same shape.")

        self.shape = U_guess.shape
        self.type = U_guess.dtype
        self.U = tf.Variable(U_guess, trainable=True)
        self.V = tf.Variable(V_guess, trainable=True)

    def get_UV_impl(self) -> Tuple[tf.Variable, tf.Variable]:
        return self.U, self.V

    def copy_w(self, w: list[tf.Variable]) -> list[tf.Tensor]:
        return [w[0].read_value(), w[1].read_value()]

    def copy_w_flat(self, w_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(w_flat)

    def get_w(self) -> list[tf.Variable]:
        return [self.U, self.V]

    def set_w(self, w: list[tf.Tensor]) -> None:
        self.U.assign(w[0])
        self.V.assign(w[1])

    def flatten_w(self, w: list[tf.Variable | tf.Tensor]) -> tf.Tensor:
        u_flat = tf.reshape(w[0], [-1])
        v_flat = tf.reshape(w[1], [-1])
        return tf.concat([u_flat, v_flat], axis=0)

    def unflatten_w(self, w_flat: tf.Tensor) -> list[tf.Tensor]:
        n = tf.size(w_flat) // 2
        u_flat = w_flat[:n]
        v_flat = w_flat[n:]
        U = tf.reshape(u_flat, self.shape)
        V = tf.reshape(v_flat, self.shape)
        return [U, V]
