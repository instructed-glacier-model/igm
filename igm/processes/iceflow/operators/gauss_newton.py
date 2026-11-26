#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Matrix-free generalized Gauss--Newton operator for velocity networks."""

from typing import Optional, Tuple

import tensorflow as tf

from ..mappings import MappingIdentity
from .banded import as_dtype
from .energy_operator import (
    ADOperator,
    BandedADOperator,
    MOLHOBandedADOperator,
    Operator,
    SSABandedADOperator,
)
from .molho_banded import supports_compact_molho
from .ssa_banded import supports_compact_ssa


class GaussNewtonOperator(Operator):
    """Apply ``J.T @ H_u @ J`` without constructing the network Jacobian.

    The network Jacobian is evaluated with a forward JVP and a reverse VJP.
    An identity mapping holds the current velocity while the existing compact
    SSA/MOLHO operators apply the velocity-space Hessian.
    """

    name = "gauss_newton"
    preconditioner_layout = "theta_flat"

    def __init__(
        self,
        cost_fn,
        mapping,
        precision: str = "float32",
        hu_mode: str = "banded",
        basis_vertical: str = "ssa",
        probe_mode: str = "autodiff",
        verify_stencil: bool = False,
    ):
        self.cost_fn = cost_fn
        self.map = mapping
        self.precision = as_dtype(precision)
        self.hu_mode = str(hu_mode).lower()
        self.basis_vertical = str(basis_vertical).lower()
        self.probe_mode = str(probe_mode).lower()
        self.verify_stencil = bool(verify_stencil)
        self._zero = tf.constant(0.0, self.precision)
        self.velocity_mapping: Optional[MappingIdentity] = None
        self.velocity_operator: Optional[Operator] = None
        self._network_nz = (
            int(mapping.Nz) if getattr(mapping, "name", "") == "network" else None
        )

        if self.hu_mode not in ("autodiff", "banded"):
            raise ValueError(
                f"Unknown gauss_newton.hu_mode: <{hu_mode}>. "
                "Use 'autodiff' or 'banded'."
            )
        if self.probe_mode not in ("autodiff", "forward", "fd"):
            raise ValueError(
                f"Unknown gauss_newton.probe_mode: <{probe_mode}>. "
                "Use 'autodiff', 'forward', or 'fd'."
            )

    def _build_velocity_operator(self) -> Operator:
        assert self.velocity_mapping is not None
        if self.hu_mode == "autodiff":
            return ADOperator(
                self.cost_fn, self.velocity_mapping, self.precision.name
            )
        if supports_compact_ssa(self.velocity_mapping):
            operator_cls = SSABandedADOperator
        elif supports_compact_molho(
            self.velocity_mapping, self.basis_vertical
        ):
            operator_cls = MOLHOBandedADOperator
        else:
            operator_cls = BandedADOperator
        return operator_cls(
            self.cost_fn,
            self.velocity_mapping,
            self.precision.name,
            verify_stencil=self.verify_stencil,
            probe_mode=self.probe_mode,
        )

    def _ensure_velocity_operator(self, inputs: tf.Tensor) -> None:
        if self.velocity_mapping is not None:
            return
        U, V = self.map.get_UV(inputs)
        shape = U.shape
        if shape.rank != 4 or not shape.is_fully_defined():
            raise ValueError(
                "gauss_newton requires statically shaped velocity batches; "
                f"received {shape}."
            )
        self.velocity_mapping = MappingIdentity(
            list(self.map.apply_bcs),
            tf.zeros_like(U),
            tf.zeros_like(V),
            self.precision.name,
        )
        self.velocity_operator = self._build_velocity_operator()

    def _network_velocities_at(
        self,
        inputs: tf.Tensor,
        theta: list[tf.Tensor | tf.Variable],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Evaluate the network without assigning weights or updating state."""
        network = self.map.network
        if not hasattr(network, "stateless_call"):
            raise RuntimeError(
                "gauss_newton requires a Keras model with stateless_call support."
            )

        non_trainable = [
            getattr(variable, "_value", variable)
            for variable in network.non_trainable_variables
        ]
        # The outer scope keeps legacy underscore-prefixed layer names valid.
        with tf.name_scope("stateless_network"):
            outputs, _ = network.stateless_call(
                theta,
                non_trainable,
                inputs,
                training=False,
            )

        outputs *= tf.cast(self.map.output_scale, outputs.dtype)
        U = tf.transpose(outputs[..., : self._network_nz], [0, 3, 1, 2])
        V = tf.transpose(outputs[..., self._network_nz :], [0, 3, 1, 2])
        for apply_bc in self.map.apply_bcs:
            U, V = apply_bc(U, V)
        return U, V

    def _velocities_at(
        self,
        inputs: tf.Tensor,
        theta: list[tf.Tensor | tf.Variable],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        mapping_name = getattr(self.map, "name", "")
        if mapping_name == "network":
            return self._network_velocities_at(inputs, theta)
        if mapping_name != "identity":
            raise TypeError(
                "gauss_newton supports network and identity mappings only."
            )

        U, V = theta
        for apply_bc in self.map.apply_bcs:
            U, V = apply_bc(U, V)
        return U, V

    def velocities_at(
        self, inputs: tf.Tensor, theta_flat: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Evaluate velocities at flat parameters without mutating the mapping."""
        return self._velocities_at(inputs, self.map.unflatten_theta(theta_flat))

    @tf.function(autograph=False, reduce_retracing=True)
    def cost_and_grad(self, inputs: tf.Tensor):
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(theta_flat)
            U, V = self.velocities_at(inputs, theta_flat)
            tape.watch((U, V))
            cost = self.cost_fn(U, V, inputs)
        grad_u = tuple(tape.gradient(cost, (U, V)))
        grad_flat = tape.gradient(cost, theta_flat)
        del tape
        if grad_flat is None:
            grad_flat = tf.zeros_like(theta_flat)
        grad_theta = self.map.unflatten_theta(grad_flat)
        return cost, grad_u, grad_theta

    @tf.function(reduce_retracing=True)
    def cost_grad_at(self, inputs: tf.Tensor, theta_flat: tf.Tensor):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(theta_flat)
            U, V = self.velocities_at(inputs, theta_flat)
            cost = self.cost_fn(U, V, inputs)
        grad = tape.gradient(cost, theta_flat)
        if grad is None:
            grad = tf.zeros_like(theta_flat)
        return cost, grad

    @tf.function(reduce_retracing=True)
    def cost_at(self, inputs: tf.Tensor, theta_flat: tf.Tensor) -> tf.Tensor:
        U, V = self.velocities_at(inputs, theta_flat)
        return self.cost_fn(U, V, inputs)

    def prepare(self, inputs: tf.Tensor, damping: tf.Tensor) -> None:
        del damping
        self._ensure_velocity_operator(inputs)
        assert self.velocity_mapping is not None
        assert self.velocity_operator is not None
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        U, V = self.velocities_at(inputs, theta_flat)
        self.velocity_mapping.U.assign(U)
        self.velocity_mapping.V.assign(V)
        self.velocity_operator.prepare(inputs, self._zero)

    @tf.function(reduce_retracing=True)
    def hvp(
        self,
        inputs: tf.Tensor,
        v_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        if self.velocity_mapping is None or self.velocity_operator is None:
            raise RuntimeError(
                "GaussNewtonOperator.prepare() must be called before hvp()."
            )

        theta = self.map.get_theta()
        tangents = self.map.unflatten_theta(v_flat)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for variable in theta:
                tape.watch(variable)
            with tf.autodiff.ForwardAccumulator(theta, tangents) as accumulator:
                U, V = self._velocities_at(inputs, theta)
            dU = accumulator.jvp(U)
            dV = accumulator.jvp(V)

        if dU is None:
            dU = tf.zeros_like(U)
        if dV is None:
            dV = tf.zeros_like(V)

        velocity_tangent = self.velocity_mapping.flatten_theta([dU, dV])
        weighted_tangent = self.velocity_operator.hvp(
            inputs, velocity_tangent, self._zero
        )
        HU, HV = self.velocity_mapping.unflatten_theta(weighted_tangent)
        pullback = tape.gradient(
            [U, V], theta, output_gradients=[HU, HV]
        )
        pullback = [
            tf.zeros_like(variable) if value is None else value
            for value, variable in zip(pullback, theta)
        ]
        result = self.map.flatten_theta(pullback)
        return result + tf.cast(damping, result.dtype) * v_flat

    def assemble_bands(self, inputs: tf.Tensor, damping: tf.Tensor):
        del inputs, damping
        return None

    def synchronization_token(self) -> Optional[tf.Tensor]:
        if self.velocity_operator is None:
            return None
        return self.velocity_operator.synchronization_token()
