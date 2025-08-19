#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from ..utils import compute_levels, compute_zeta_dzeta, define_vertical_weight


class VerticalDiscr(ABC):

    levels: tf.Tensor
    zeta: tf.Tensor
    dzeta: tf.Tensor
    Vnodal: tf.Tensor
    Vnodal_b: tf.Tensor
    Vnodal_s: tf.Tensor
    Vnodal_bar: tf.Tensor
    Vnodal_grad: tf.Tensor

    def __init__(self, cfg: DictConfig):
        cfg_numerics = cfg.processes.iceflow.numerics

        self.dtype = tf.float32
        self.Nz = cfg_numerics.Nz
        self.vertical_spacing = cfg_numerics.vert_spacing
        self.staggered_grid = cfg_numerics.staggered_grid
        self.vertical_basis = cfg_numerics.vert_basis

    @abstractmethod
    def build_matrices(self) -> None:
        raise NotImplementedError(
            "❌ The matrices construction is not implemented in this class."
        )


class LagrangreDiscr(VerticalDiscr):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.build_matrices()

    def build_matrices(self) -> None:
        self.levels = compute_levels(self.Nz, self.vertical_spacing)
        self.zeta, self.dzeta = compute_zeta_dzeta(self.levels)

        vert_weight = define_vertical_weight(self.Nz, self.vertical_spacing)[:, 0, 0]

        Nz = self.Nz

        self.Vnodal = tf.eye(Nz, dtype=self.dtype)
        self.Vnodal_b = tf.zeros(Nz, dtype=self.dtype)
        self.Vnodal_s = tf.zeros(Nz, dtype=self.dtype)
        self.Vnodal_b = tf.tensor_scatter_nd_update(self.Vnodal_b, [[0]], [1.0])
        self.Vnodal_s = tf.tensor_scatter_nd_update(self.Vnodal_s, [[Nz - 1]], [1.0])
        self.Vnodal_bar = vert_weight

        if Nz == 1:
            self.Vnodal_grad = tf.zeros(Nz, dtype=self.dtype)
        else:
            fd_upper = tf.linalg.diag(+1.0 / self.dzeta[None, :, None, None], k=1)
            fd_lower = tf.linalg.diag(-1.0 / self.dzeta[None, :, None, None], k=0)
            self.Vnodal_grad = fd_upper + fd_lower
