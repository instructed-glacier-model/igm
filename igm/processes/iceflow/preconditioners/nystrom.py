#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Randomized Nyström preconditioning for flat network parameters."""

import time
import warnings
from typing import Optional

import tensorflow as tf

from ..operators.banded import as_dtype
from .preconditioner import Preconditioner


class NystromPreconditioner(Preconditioner):
    """Low-rank spectral preconditioner for a PSD matrix-free operator."""

    name = "nystrom"
    needs_operator = True
    layout = "theta_flat"

    def __init__(
        self,
        mapping,
        rank: int = 32,
        precision: str = "float32",
        seed: int = 0,
        sketch_memory_gb: float = 2.0,
        minimum_rank: int = 8,
        print_timing: bool = False,
    ):
        if int(rank) < 1:
            raise ValueError("nystrom.rank must be positive.")
        if float(sketch_memory_gb) <= 0.0:
            raise ValueError("nystrom.sketch_memory_gb must be positive.")
        if int(minimum_rank) < 1:
            raise ValueError("nystrom.minimum_rank must be positive.")

        self.dtype = as_dtype(precision)
        self.requested_rank = int(rank)
        self.seed = int(seed)
        self.sketch_memory_gb = float(sketch_memory_gb)
        self.minimum_rank = int(minimum_rank)
        self.print_timing = bool(print_timing)
        self._operator = None
        self._zero = tf.constant(0.0, self.dtype)
        self._tiny = tf.constant(
            1.0e-30 if self.dtype == tf.float32 else 1.0e-300,
            self.dtype,
        )
        self._eps = tf.constant(
            1.0e-6 if self.dtype == tf.float32 else 1.0e-13,
            self.dtype,
        )

        theta_flat = mapping.flatten_theta(mapping.get_theta())
        n_static = theta_flat.shape.num_elements()
        if n_static is None:
            raise ValueError("nystrom requires a statically sized parameter vector.")
        self.n_parameters = int(n_static)
        bytes_per_value = self.dtype.size
        budget_bytes = int(self.sketch_memory_gb * (1024**3))
        rank_from_budget = budget_bytes // max(
            3 * self.n_parameters * bytes_per_value, 1
        )
        self.effective_rank = min(
            self.requested_rank,
            self.n_parameters,
            int(rank_from_budget),
        )
        self.active = self.effective_rank >= self.minimum_rank
        if not self.active:
            self.effective_rank = 0

        self.estimated_sketch_bytes = (
            3 * self.n_parameters * self.effective_rank * bytes_per_value
        )
        if self.active and self.effective_rank < min(
            self.requested_rank, self.n_parameters
        ):
            gib = self.estimated_sketch_bytes / float(1024**3)
            warnings.warn(
                "Nyström rank reduced from "
                f"{self.requested_rank} to {self.effective_rank} by the "
                f"sketch memory limit (estimated storage: {gib:.3f} GiB).",
                RuntimeWarning,
                stacklevel=2,
            )
        self._eigenvectors: Optional[tf.Variable] = None
        self._eigenvalues: Optional[tf.Variable] = None
        self._damping = tf.Variable(
            self._zero, trainable=False, name="nystrom_damping"
        )
        self._refresh_index = 0

    def set_operator(self, operator) -> None:
        self._operator = operator

    def set_damping(self, damping: tf.Tensor) -> None:
        self._damping.assign(tf.cast(damping, self.dtype))

    def update(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        if self._operator is None:
            raise RuntimeError("nystrom operator was not registered.")
        self.set_damping(damping)
        if not self.active:
            if self._refresh_index == 0:
                warnings.warn(
                    "Nyström disabled: the sketch memory budget allows fewer "
                    f"than {self.minimum_rank} vectors; using identity.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._refresh_index += 1
            return

        n = self.n_parameters
        rank = self.effective_rank
        update_start = time.perf_counter()
        omega = tf.random.stateless_normal(
            [n, rank],
            seed=[self.seed, self._refresh_index],
            dtype=self.dtype,
        )
        if self.print_timing:
            _ = omega[0, 0].numpy()
            random_seconds = time.perf_counter() - update_start
            stage_start = time.perf_counter()
        # Use the small Gram matrix instead of a memory-heavy tall QR.
        omega_gram = tf.matmul(omega, omega, transpose_a=True)
        omega_gram = 0.5 * (omega_gram + tf.transpose(omega_gram))
        omega_values, omega_vectors = tf.linalg.eigh(omega_gram)
        omega_floor = self._eps * tf.maximum(
            tf.reduce_max(omega_values), self._tiny
        )
        omega_transform = omega_vectors * tf.math.rsqrt(
            tf.maximum(omega_values, omega_floor)
        )[tf.newaxis, :]
        omega = tf.matmul(omega, omega_transform)
        if self.print_timing:
            _ = omega[0, 0].numpy()
            orthogonalize_seconds = time.perf_counter() - stage_start
            stage_start = time.perf_counter()

        # Reuse one HVP graph without rank-scaled network activations.
        columns = [
            self._operator.hvp(inputs, omega[:, column], self._zero)
            for column in range(rank)
        ]
        Y = tf.stack(columns, axis=1)
        del columns
        if self.print_timing:
            _ = Y[0, 0].numpy()
            hvp_seconds = time.perf_counter() - stage_start
            stage_start = time.perf_counter()

        shift = self._eps * tf.sqrt(tf.cast(n, self.dtype)) * tf.norm(Y)
        shifted_Y = Y + shift * omega
        core = tf.matmul(omega, shifted_Y, transpose_a=True)
        core = 0.5 * (core + tf.transpose(core))
        del Y, omega
        if self.print_timing:
            _ = core[0, 0].numpy()
            core_assembly_seconds = time.perf_counter() - stage_start
            stage_start = time.perf_counter()

        # Clipping stabilizes the nearly positive-semidefinite fp32 core.
        core_eigenvalues, core_vectors = tf.linalg.eigh(core)
        if self.print_timing:
            _ = core_eigenvalues[0].numpy()
            core_eigh_seconds = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
        core_scale = tf.maximum(
            tf.reduce_max(tf.abs(core_eigenvalues)), self._tiny
        )
        core_floor = self._eps * core_scale
        inverse_sqrt = tf.math.rsqrt(
            tf.maximum(core_eigenvalues, core_floor)
        )
        factor = tf.matmul(shifted_Y, core_vectors)
        factor = factor * inverse_sqrt[tf.newaxis, :]
        del shifted_Y
        if self.print_timing:
            _ = factor[0, 0].numpy()
            factor_seconds = time.perf_counter() - stage_start
            stage_start = time.perf_counter()

        # Diagonalize the small Gram matrix instead of the tall factor.
        factor_gram = tf.matmul(factor, factor, transpose_a=True)
        factor_gram = 0.5 * (factor_gram + tf.transpose(factor_gram))
        if self.print_timing:
            _ = factor_gram[0, 0].numpy()
            factor_gram_seconds = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
        factor_eigenvalues, factor_vectors = tf.linalg.eigh(factor_gram)
        if self.print_timing:
            _ = factor_eigenvalues[0].numpy()
            factor_eigh_seconds = time.perf_counter() - stage_start
            stage_start = time.perf_counter()
        factor_eigenvalues = tf.reverse(factor_eigenvalues, axis=[0])
        factor_vectors = tf.reverse(factor_vectors, axis=[1])
        factor_scale = tf.maximum(factor_eigenvalues[0], self._tiny)
        factor_floor = self._eps * factor_scale
        inverse_singular_values = tf.math.rsqrt(
            tf.maximum(factor_eigenvalues, factor_floor)
        )
        eigenvectors = tf.matmul(factor, factor_vectors)
        eigenvectors *= inverse_singular_values[tf.newaxis, :]
        del factor
        eigenvalues = tf.maximum(
            factor_eigenvalues - shift,
            tf.cast(0.0, self.dtype),
        )
        if self.print_timing:
            _ = eigenvectors[0, 0].numpy()
            reconstruction_seconds = time.perf_counter() - stage_start

        if self._eigenvectors is None:
            self._eigenvectors = tf.Variable(
                eigenvectors, trainable=False, name="nystrom_eigenvectors"
            )
            self._eigenvalues = tf.Variable(
                eigenvalues, trainable=False, name="nystrom_eigenvalues"
            )
        else:
            self._eigenvectors.assign(eigenvectors)
            self._eigenvalues.assign(eigenvalues)

        if self.print_timing:
            _ = self._eigenvectors[0, 0].numpy()
            total_seconds = time.perf_counter() - update_start
            print(
                "[gauss_newton_nystrom_timing] "
                f"rank={rank} "
                f"requested_rank={self.requested_rank} "
                f"sketch_gib={self.estimated_sketch_bytes / float(1024**3):.3f} "
                f"random={random_seconds:.6f} "
                f"orthogonalize={orthogonalize_seconds:.6f} "
                f"hvp={hvp_seconds:.6f} "
                f"core_assembly={core_assembly_seconds:.6f} "
                f"core_eigh={core_eigh_seconds:.6f} "
                f"factor={factor_seconds:.6f} "
                f"factor_gram={factor_gram_seconds:.6f} "
                f"factor_eigh={factor_eigh_seconds:.6f} "
                f"reconstruct={reconstruction_seconds:.6f} "
                f"total={total_seconds:.6f}",
                flush=True,
            )

        self._refresh_index += 1

    @property
    def lambda_max(self) -> Optional[tf.Tensor]:
        if self._eigenvalues is None:
            return None
        return self._eigenvalues[0]

    @tf.function(reduce_retracing=True)
    def apply(self, residual_flat: tf.Tensor) -> tf.Tensor:
        if self._eigenvectors is None or self._eigenvalues is None:
            return residual_flat
        value = tf.cast(residual_flat, self.dtype)
        projected = tf.linalg.matvec(
            self._eigenvectors, value, transpose_a=True
        )
        damping = tf.maximum(self._damping, self._tiny)
        reference = tf.maximum(self._eigenvalues[-1], tf.cast(0.0, self.dtype))
        scaled = (reference + damping) * projected / (
            self._eigenvalues + damping
        )
        result = (
            value
            - tf.linalg.matvec(self._eigenvectors, projected)
            + tf.linalg.matvec(self._eigenvectors, scaled)
        )
        return tf.cast(result, residual_flat.dtype)

    def synchronization_token(self) -> Optional[tf.Tensor]:
        return self._eigenvectors
