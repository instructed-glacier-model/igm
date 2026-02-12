#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Any, Dict, List, Tuple, Union

from .mapping import Mapping
from .masks import masks_gr
from ..bcs import BoundaryCondition

TV = Union[tf.Tensor, tf.Variable]


class MappingComposite(Mapping):

    def __init__(
        self,
        bcs: List[BoundaryCondition],
        mapping_gr: Mapping,
        mapping_fl: Mapping,
        mask_gr_name: str = "friction",
        mask_gr_kwargs: Dict[str, Any] = None,
        active: str = "all",
        precision: str = "float32",
    ):
        if mask_gr_name not in masks_gr:
            raise ValueError(
                f"Unknown indicator: {mask_gr_name!r}. "
                f"Available: {list(masks_gr.keys())}"
            )
        super().__init__(bcs, precision)

        self.mapping_gr = mapping_gr
        self.mapping_fl = mapping_fl
        self.mask_gr_fn = masks_gr[mask_gr_name]
        self.mask_gr_kwargs = mask_gr_kwargs or {}

        shapes_gr = [w.shape for w in self.mapping_gr.get_theta()]
        shapes_fl = [w.shape for w in self.mapping_fl.get_theta()]
        self.shapes = {
            "gr": shapes_gr,
            "fl": shapes_fl,
            "all": shapes_gr + shapes_fl,
        }
        self.sizes = {k: [tf.reduce_prod(s) for s in v] for k, v in self.shapes.items()}

        self.active = active

    @property
    def active(self) -> str:
        return self._active

    @active.setter
    def active(self, mode: str) -> None:
        if mode not in ("all", "gr", "fl"):
            raise ValueError(f"Unknown active mode: {mode!r}")
        self._active = mode

    def _compute_weights(self) -> Tuple[tf.Tensor, tf.Tensor]:
        w_g = self.mask_gr_fn(self.inputs, **self.mask_gr_kwargs)
        return w_g, 1.0 - w_g

    def get_UV_impl(self) -> Tuple[TV, TV]:
        self.mapping_gr.set_inputs(self.inputs)
        self.mapping_fl.set_inputs(self.inputs)

        U_g, V_g = self.mapping_gr.get_UV_impl()
        U_f, V_f = self.mapping_fl.get_UV_impl()

        w_g, w_f = self._compute_weights()
        w_g = tf.transpose(w_g, perm=[0, 3, 1, 2])
        w_f = tf.transpose(w_f, perm=[0, 3, 1, 2])

        U = w_g * U_g + w_f * U_f
        V = w_g * V_g + w_f * V_f
        return U, V

    def get_theta(self) -> list[tf.Variable]:
        if self.active == "gr":
            return self.mapping_gr.get_theta()
        if self.active == "fl":
            return self.mapping_fl.get_theta()
        return self.mapping_gr.get_theta() + self.mapping_fl.get_theta()

    def set_theta(self, theta: list[tf.Tensor]) -> None:
        if self.active == "gr":
            self.mapping_gr.set_theta(theta)
        elif self.active == "fl":
            self.mapping_fl.set_theta(theta)
        else:
            n = len(self.shapes["gr"])
            self.mapping_gr.set_theta(theta[:n])
            self.mapping_fl.set_theta(theta[n:])

    def copy_theta(self, theta: list[tf.Variable]) -> list[tf.Tensor]:
        return [
            t.read_value() if isinstance(t, tf.Variable) else tf.identity(t)
            for t in theta
        ]

    def copy_theta_flat(self, theta_flat: tf.Tensor) -> tf.Tensor:
        return tf.identity(theta_flat)

    def flatten_theta(self, theta: list) -> tf.Tensor:
        return tf.concat([tf.reshape(t, [-1]) for t in theta], axis=0)

    def unflatten_theta(self, theta_flat: tf.Tensor) -> list[tf.Tensor]:
        shapes = self.shapes[self.active]
        sizes = self.sizes[self.active]
        splits = tf.split(theta_flat, sizes)
        return [tf.reshape(t, s) for t, s in zip(splits, shapes)]

    def update_normalizer(self, inputs: tf.Tensor) -> None:
        self.mapping_gr.update_normalizer(inputs)
        self.mapping_fl.update_normalizer(inputs)

    def on_minimize_start(self, iter_max: int) -> None:
        self.mapping_gr.on_minimize_start(iter_max)
        self.mapping_fl.on_minimize_start(iter_max)

    @tf.function
    def on_step_end(self, iteration: tf.Tensor) -> None:
        self.mapping_gr.on_step_end(iteration)
        self.mapping_fl.on_step_end(iteration)
