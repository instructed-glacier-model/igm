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
        trainable_tail: int = 0,
    ):
        super().__init__(bcs, precision)
        self.network = network
        if not network.built:
            network.build((None, None, None, network.nb_inputs))
        self.output_scale = output_scale
        # trainable_tail > 0 restricts training to the LAST n variables of the
        # network (construction order, so the head of the net stays fixed);
        # 0 trains everything. Backends create their variables directly, so
        # this is the variable-level equivalent of freezing early layers.
        self.trainable_tail = int(trainable_tail)
        self.shapes = [w.shape for w in self._theta_vars()]
        self.sizes = [tf.reduce_prod(s) for s in self.shapes]
        self.Nz = Nz
        self.name = "network"

    def _theta_vars(self) -> list[tf.Variable]:
        tv = self.network.trainable_variables
        if self.trainable_tail > 0:
            return tv[-self.trainable_tail:]
        return tv
        
    def get_UV_impl(self) -> Tuple[tf.Tensor, tf.Tensor]:
        Y = self.network(self.inputs) * self.output_scale
        U, V = Y_to_UV(self.Nz, Y)
        return U, V


    def copy_theta(self, theta: list[tf.Variable]) -> list[tf.Tensor]:
        return [theta_i.read_value() for theta_i in theta]

    def copy_theta_flat(self, theta_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(theta_flat)

    def get_theta(self) -> list[tf.Variable]:
        return [getattr(v, "_value", v) for v in self._theta_vars()]

    def set_theta(self, theta: list[tf.Tensor]) -> None:
        for var, val in zip(self._theta_vars(), theta):
            var.assign(val)

    def flatten_theta(self, theta: list[tf.Variable | tf.Tensor]) -> tf.Tensor:
        theta_flat = [tf.reshape(theta_i, [-1]) for theta_i in theta]
        return tf.concat(theta_flat, axis=0)

    def unflatten_theta(self, theta_flat: tf.Tensor) -> list[tf.Tensor]:
        splits = tf.split(theta_flat, self.sizes)
        return [tf.reshape(t, s) for t, s in zip(splits, self.shapes)]
