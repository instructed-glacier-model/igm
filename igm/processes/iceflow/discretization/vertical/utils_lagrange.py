#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf


class BasisP1:
    """P1 (linear) basis functions on the reference element [0,1]."""

    @staticmethod
    def phi0(xi: tf.Tensor) -> tf.Tensor:
        """Basis function: phi0(0)=1, phi0(1)=0."""
        return 1.0 - xi

    @staticmethod
    def phi1(xi: tf.Tensor) -> tf.Tensor:
        """Basis function: phi1(0)=0, phi1(1)=1."""
        return xi

    @staticmethod
    def grad_phi0(xi: tf.Tensor) -> tf.Tensor:
        """Gradient of basis function phi0."""
        return -tf.ones_like(xi)

    @staticmethod
    def grad_phi1(xi: tf.Tensor) -> tf.Tensor:
        """Gradient of basis function phi1."""
        return tf.ones_like(xi)

    @staticmethod
    def int_phi0(xi: tf.Tensor) -> tf.Tensor:
        """Integral of basis function phi0."""
        return xi - 0.5 * xi * xi

    @staticmethod
    def int_phi1(xi: tf.Tensor) -> tf.Tensor:
        """Integral of basis function phi1."""
        return 0.5 * xi * xi


@dataclass
class Element:
    """1D finite element with endpoints x0 and x1."""

    x0: tf.Tensor
    x1: tf.Tensor

    def xi(self, x: tf.Tensor) -> tf.Tensor:
        """Map physical coordinate x to reference coordinate xi in [0,1]."""
        return (x - self.x0) / (self.x1 - self.x0)

    def jac(self) -> tf.Tensor:
        """Jacobian of the transformation (element length)."""
        return self.x1 - self.x0

    def mask(
        self, x: tf.Tensor, include_l: bool = True, include_r: bool = False
    ) -> tf.Tensor:
        """Boolean mask for points inside the element."""
        cond_l = (x >= self.x0) if include_l else (x > self.x0)
        cond_r = (x <= self.x1) if include_r else (x < self.x1)
        return cond_l & cond_r


def compute_basis(
    nodes: tf.Tensor, idx: int, elem_ref: type[BasisP1] = BasisP1
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Compute hat function basis centered at node idx."""

    n_nodes = int(nodes.shape[0])
    is_last = idx == n_nodes - 1

    elem_l = Element(nodes[idx - 1], nodes[idx]) if idx > 0 else None
    elem_r = Element(nodes[idx], nodes[idx + 1]) if idx < n_nodes - 1 else None

    def basis_fn(x: tf.Tensor) -> tf.Tensor:
        result = tf.zeros_like(x)

        if elem_l is not None:
            xi = elem_l.xi(x)
            mask = elem_l.mask(x, include_l=True, include_r=is_last)
            result = tf.where(mask, elem_ref.phi1(xi), result)

        if elem_r is not None:
            xi = elem_r.xi(x)
            mask = elem_r.mask(x, include_l=True, include_r=False)
            result = tf.where(mask, elem_ref.phi0(xi), result)

        return result

    return basis_fn


def compute_basis_grad(
    nodes: tf.Tensor, idx: int, elem_ref: type[BasisP1] = BasisP1
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Compute gradient of hat function basis centered at node idx."""
    n_nodes = int(nodes.shape[0])
    is_last = idx == n_nodes - 1

    elem_l = Element(nodes[idx - 1], nodes[idx]) if idx > 0 else None
    elem_r = Element(nodes[idx], nodes[idx + 1]) if idx < n_nodes - 1 else None

    def basis_grad_fn(x: tf.Tensor) -> tf.Tensor:
        result = tf.zeros_like(x)

        if elem_l is not None:
            xi = elem_l.xi(x)
            mask = elem_l.mask(x, include_l=True, include_r=is_last)
            result = tf.where(mask, elem_ref.grad_phi1(xi) / elem_l.jac(), result)

        if elem_r is not None:
            xi = elem_r.xi(x)
            mask = elem_r.mask(x, include_l=True, include_r=False)
            result = tf.where(mask, elem_ref.grad_phi0(xi) / elem_r.jac(), result)

        return result

    return basis_grad_fn


def compute_basis_int(
    nodes: tf.Tensor, idx: int, elem_ref: type[BasisP1] = BasisP1
) -> Callable[[tf.Tensor], tf.Tensor]:
    """Compute integral of hat function basis centered at node idx."""
    n_nodes = int(nodes.shape[0])
    is_last = idx == n_nodes - 1

    elem_l = Element(nodes[idx - 1], nodes[idx]) if idx > 0 else None
    elem_r = Element(nodes[idx], nodes[idx + 1]) if idx < n_nodes - 1 else None

    int_l = 0.5 * elem_l.jac() if elem_l is not None else 0.0
    int_r = 0.5 * elem_r.jac() if elem_r is not None else 0.0
    int_total = int_l + int_r

    def basis_int_fn(x: tf.Tensor) -> tf.Tensor:
        result = tf.zeros_like(x)

        if elem_l is not None:
            xi = elem_l.xi(x)
            mask = elem_l.mask(x, include_l=True, include_r=is_last)
            result = tf.where(mask, elem_l.jac() * elem_ref.int_phi1(xi), result)

        if elem_r is not None:
            xi = elem_r.xi(x)
            mask = elem_r.mask(x, include_l=True, include_r=False)
            result = tf.where(
                mask,
                int_l + elem_r.jac() * elem_ref.int_phi0(xi),
                result,
            )

        x_plateau = elem_r.x1 if elem_r is not None else elem_l.x1
        result = tf.where(x >= x_plateau, int_total, result)

        return result

    return basis_int_fn
