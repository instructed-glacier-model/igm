#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import List, Tuple

from .mapping import Mapping
from ..bcs import BoundaryCondition
from igm.processes.iceflow.utils.data_preprocessing import Y_to_UV


class MappingNetwork(Mapping):
    def __init__(
        self,
        bcs: List[BoundaryCondition],
        network: tf.keras.Model,
        Nz: tf.Tensor,
        output_scale: tf.Tensor = 1.0,
        precision: str = "float32",
    ):
        super().__init__(bcs, precision)
        self.network = network
        if not network.built:
            network.build((None, None, None, network.nb_inputs))
        self.output_scale = output_scale
        self.shapes = [w.shape for w in network.trainable_variables]
        self.sizes = [tf.reduce_prod(s) for s in self.shapes]
        self.Nz = Nz
        self.name = "network"
        
    def get_UV_impl(self) -> Tuple[tf.Tensor, tf.Tensor]:
        Y = self.network(self.inputs) * self.output_scale
        U, V = Y_to_UV(self.Nz, Y)
        return U, V


    def copy_theta(self, theta: list[tf.Variable]) -> list[tf.Tensor]:
        return [theta_i.read_value() for theta_i in theta]

    def copy_theta_flat(self, theta_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(theta_flat)

    def get_theta(self) -> list[tf.Variable]:
        return [v._value for v in self.network.trainable_variables]

    def set_theta(self, theta: list[tf.Tensor]) -> None:
        for var, val in zip(self.network.trainable_variables, theta):
            var.assign(val)

    def flatten_theta(self, theta: list[tf.Variable | tf.Tensor]) -> tf.Tensor:
        theta_flat = [tf.reshape(theta_i, [-1]) for theta_i in theta]
        return tf.concat(theta_flat, axis=0)

    def unflatten_theta(self, theta_flat: tf.Tensor) -> list[tf.Tensor]:
        splits = tf.split(theta_flat, self.sizes)
        return [tf.reshape(t, s) for t, s in zip(splits, self.shapes)]
