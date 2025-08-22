#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

TensorOrVariable = Union[tf.Tensor, tf.Variable]


class Mapping(ABC):

    def set_inputs(self, inputs: tf.Tensor) -> None:
        self.inputs = inputs

    @abstractmethod
    def get_UV_impl(self) -> Tuple[TensorOrVariable, TensorOrVariable]:
        pass

    @tf.function(jit_compile=True)
    def get_UV(self, inputs: tf.Tensor) -> Tuple[TensorOrVariable, TensorOrVariable]:
        self.set_inputs(inputs)
        U, V = self.get_UV_impl()
        return U, V

    @abstractmethod
    def get_w(self) -> Any:
        pass

    @abstractmethod
    def set_w(self, w: Any) -> None:
        pass

    @abstractmethod
    def copy_w(self, w: Any) -> Any:
        pass

    @abstractmethod
    def copy_w_flat(self, w_flat: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def flatten_w(self, w: Any) -> tf.Tensor:
        pass

    @abstractmethod
    def unflatten_w(self, w_flat: tf.Tensor) -> Any:
        pass
