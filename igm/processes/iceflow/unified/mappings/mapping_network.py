#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Tuple

from .mapping import Mapping
from igm.processes.iceflow.utils.data_preprocessing import Y_to_UV


class MappingNetwork(Mapping):
    def __init__(
        self,
        network: tf.keras.Model,
        Nz: tf.Tensor,
        output_scale: tf.Tensor = 1.0,
    ):
        self.network = network
        self.Nz = Nz
        self.output_scale = output_scale
        self.shapes = [w.shape for w in network.trainable_variables]
        self.sizes = [tf.reduce_prod(s) for s in self.shapes]

    def get_UV_impl(self) -> Tuple[tf.Tensor, tf.Tensor]:
        Y = self.network(self.inputs) * self.output_scale
        U, V = Y_to_UV(self.Nz, Y)
        return U, V

    def copy_w(self, w: list[tf.Variable]) -> list[tf.Tensor]:
        return [wi.read_value() for wi in w]

    def copy_w_flat(self, w_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(w_flat)

    def get_w(self) -> list[tf.Variable]:
        return self.network.trainable_variables

    def set_w(self, w: list[tf.Tensor]) -> None:
        for var, val in zip(self.network.trainable_variables, w):
            var.assign(val)

    def flatten_w(self, w: list[tf.Variable | tf.Tensor]) -> tf.Tensor:
        w_flat = [tf.reshape(wi, [-1]) for wi in w]
        return tf.concat(w_flat, axis=0)

    def unflatten_w(self, w_flat: tf.Tensor) -> list[tf.Tensor]:
        splits = tf.split(w_flat, self.sizes)
        return [tf.reshape(t, s) for t, s in zip(splits, self.shapes)]
