#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Analytic Q1/SSA Hessian assembly for y-invariant flowlines.

For ``Ny=2`` plus ``PeriodicNS`` the two Q1 rows represent the same unknown.
Each cell therefore has only two vector-valued nodal velocities, and both the
viscous and basal-sliding energies contribute a 2x2-block, two-node element
matrix.  This module evaluates those local second derivatives directly and
scatters them into the west/center/east bands used by ``tridiag_newton``.

Gravity and floating-front work are linear in velocity (with the standard
unclipped gravity configuration), so their Hessians are exactly zero.
"""

import math
from typing import Dict, Iterable, Tuple

import tensorflow as tf

from igm.processes.iceflow.utils.velocities import compute_cell_ice_mask

from .tridiag1d import Tridiag1DADOperator


def _outer_blocks(
    h_uu: tf.Tensor,
    h_uv: tf.Tensor,
    h_vv: tf.Tensor,
) -> tf.Tensor:
    """Pack symmetric 2x2 fields as ``(B,2,2,N)`` blocks."""
    row_u = tf.stack([h_uu, h_uv], axis=1)
    row_v = tf.stack([h_uv, h_vv], axis=1)
    return tf.stack([row_u, row_v], axis=1)


class Tridiag1DAnalyticOperator(Tridiag1DADOperator):
    """Analytic Q1/SSA energy derivatives for flowline problems.

    The two-component path assembles exact Hessian bands and uses the standard
    IGM energy for cost and gradient evaluation. The scalar path directly
    assembles the complete cost, gradient, and Hessian for its supported
    energy components.
    """

    name = "tridiag1d_analytic"

    _LINEAR_COMPONENTS = frozenset(("gravity", "floating"))
    _POWER_LAW_SLIDING = frozenset(("weertman", "budd", "mohr_coulomb"))

    def __init__(
        self,
        cost_fn,
        mapping,
        cfg,
        energy_components: Iterable,
        precision: str = "float32",
        verify_stencil: bool = False,
    ):
        # Reuse the storage, application, full-gradient, and verification
        # machinery.  No AD/FD probing method is called by this subclass.
        super().__init__(
            cost_fn,
            mapping,
            precision=precision,
            verify_stencil=verify_stencil,
            probe_mode="autodiff",
        )

        numerics = cfg.processes.iceflow.numerics
        if str(numerics.basis_horizontal).lower() != "q1":
            raise ValueError(
                "Analytic tridiag_newton assembly requires "
                "numerics.basis_horizontal: q1."
            )
        if str(numerics.basis_vertical).lower() != "ssa":
            raise ValueError(
                "Analytic tridiag_newton assembly requires "
                "numerics.basis_vertical: ssa."
            )

        self._input_indices: Dict[str, int] = {
            name: i for i, name in enumerate(cfg.processes.iceflow.unified.inputs)
        }
        self._viscosity = None
        self._sliding = None
        self._gravity = None
        self._floating = None
        for component in energy_components:
            name = str(component.name).lower()
            if name == "viscosity":
                self._viscosity = component
            elif name == "gravity":
                self._gravity = component
            elif name == "floating":
                self._floating = component
            elif name in self._POWER_LAW_SLIDING or name == "regu_coulomb":
                self._sliding = component
            elif name not in self._LINEAR_COMPONENTS:
                raise ValueError(
                    "Analytic tridiag_newton assembly does not support the "
                    f"nonlinear energy component {name!r}."
                )

        if self._gravity is not None and bool(self._gravity.params.fnge):
            raise ValueError(
                "Analytic tridiag_newton assembly requires "
                "force_negative_gravitational_energy=false because clipping "
                "makes gravity nonlinear in velocity."
            )

        required = {"thk"}
        if self._viscosity is not None:
            required.update(("arrhenius", "dX"))
        if self._gravity is not None:
            required.update(("usurf", "dX"))
        if self._floating is not None:
            required.update(("usurf", "water_level", "dX"))
        if self._sliding is not None:
            required.update(("usurf", "dX"))
            if self._sliding.name != "mohr_coulomb":
                if (
                    "tau_ref" not in self._input_indices
                    and "slidingco" not in self._input_indices
                ):
                    required.add("tau_ref")
            if self._sliding.name in ("budd", "regu_coulomb", "mohr_coulomb"):
                required.add("effective_pressure")
        missing = sorted(required.difference(self._input_indices))
        if missing:
            raise ValueError(
                "Analytic tridiag_newton assembly is missing input channel(s): "
                + ", ".join(missing)
            )

        # Jacobian of the affine boundary map, restricted to the live row.
        # Computing it as BC(1)-BC(0) handles non-zero Dirichlet values and
        # respects the configured BC ordering without type-specific logic.
        ones = tf.ones(mapping.shape, self.precision)
        zeros = tf.zeros(mapping.shape, self.precision)
        u_one, v_one = ones, ones
        u_zero, v_zero = zeros, zeros
        for apply_bc in mapping.apply_bcs:
            u_one, v_one = apply_bc(u_one, v_one)
            u_zero, v_zero = apply_bc(u_zero, v_zero)
        self._active = tf.stack(
            [
                u_one[:, 0, 0, :] - u_zero[:, 0, 0, :],
                v_one[:, 0, 0, :] - v_zero[:, 0, 0, :],
            ],
            axis=1,
        )

        gp_a = 0.5 - 0.5 / math.sqrt(3.0)
        gp_b = 0.5 + 0.5 / math.sqrt(3.0)
        self._xi = tf.constant([gp_a, gp_a, gp_b, gp_b], self.precision)[
            tf.newaxis, :, tf.newaxis
        ]
        self._eta = tf.constant([gp_a, gp_b, gp_a, gp_b], self.precision)[
            tf.newaxis, :, tf.newaxis
        ]
        self._quad_weight = tf.cast(0.25, self.precision)
        self._energy_normalization = tf.cast(
            1.0 / (self.B * (self.Ny - 1) * (self.Nx - 1)), self.precision
        )
        self.supports_scalar_flowline = (
            self._sliding is None or self._sliding.name in self._POWER_LAW_SLIDING
        )

    def _uv_from_u_row(self, u_row: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        U = self._embed_row0(u_row)
        V = tf.zeros_like(U)
        for apply_bc in self.map.apply_bcs:
            U, V = apply_bc(U, V)
        return U, V

    @tf.function(reduce_retracing=True)
    def cost_grad_u_at(
        self, inputs: tf.Tensor, u_row: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Full IGM energy and gradient restricted to flow-parallel velocity."""
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(u_row)
            U, V = self._uv_from_u_row(u_row)
            cost = self._ad.cost_fn(U, V, inputs)
        return cost, tape.gradient(cost, u_row)

    def _scalar_viscosity_terms(
        self, inputs: tf.Tensor, u: tf.Tensor
    ) -> Tuple[tf.Tensor, ...]:
        dtype = self.precision
        n = tf.cast(self._viscosity.params.n, dtype)
        p = 1.0 + 1.0 / n
        exponent = (p - 2.0) / 2.0
        regularization2 = tf.cast(
            self._viscosity.params.eps_dot_regu, dtype
        ) ** 2
        maximum2 = tf.cast(self._viscosity.params.eps_dot_max, dtype) ** 2

        h = self._field(inputs, "thk")
        arrhenius = self._field(inputs, "arrhenius")
        dx = self._field(inputs, "dX")[:, 1:, 1:][:, 0, :]
        stiffness = 2.0 * tf.pow(arrhenius, -1.0 / n)
        coefficient = self._quad_weight * tf.reduce_sum(
            self._interp_q1(h) * self._interp_q1(stiffness), axis=1
        )

        du = (u[:, 1:] - u[:, :-1]) / dx
        strain2 = du * du
        capped = tf.minimum(strain2, maximum2)
        base = capped + regularization2
        energy = coefficient * tf.pow(base, exponent) * capped / p
        active_cap = tf.cast(strain2 <= maximum2, dtype)
        f_prime = active_cap * (
            tf.pow(base, exponent)
            + exponent * capped * tf.pow(base, exponent - 1.0)
        ) / p
        f_second = active_cap * (
            2.0 * exponent * tf.pow(base, exponent - 1.0)
            + exponent
            * (exponent - 1.0)
            * capped
            * tf.pow(base, exponent - 2.0)
        ) / p
        derivative = coefficient * 2.0 * f_prime * du / dx
        curvature = coefficient * (
            2.0 * f_prime + 4.0 * f_second * du * du
        ) / (dx * dx)
        return energy, -derivative, derivative, curvature, -curvature, curvature

    def _scalar_sliding_terms(
        self, inputs: tf.Tensor, u: tf.Tensor
    ) -> Tuple[tf.Tensor, ...]:
        params = self._sliding.params
        dtype = self.precision
        p = 1.0 + 1.0 / tf.cast(params.exponent, dtype)
        left = 1.0 - self._xi
        right = self._xi
        u_q = left * u[:, tf.newaxis, :-1] + right * u[:, tf.newaxis, 1:]
        bed = self._field(inputs, "usurf") - self._field(inputs, "thk")
        bed_x, _ = self._grad_q1(bed, self._field(inputs, "dX"))
        correction = u_q * bed_x
        metric = u_q + bed_x * correction
        metric_second = 1.0 + bed_x * bed_x
        speed2 = (
            u_q * u_q
            + tf.cast(params.regularization, dtype) ** 2
            + correction * correction
        )
        coefficient = self._sliding_coefficient(inputs)
        radial = coefficient * tf.pow(speed2, p / 2.0 - 1.0)
        gradient_q = radial * metric
        hessian_q = radial * metric_second + coefficient * (p - 2.0) * tf.pow(
            speed2, p / 2.0 - 2.0
        ) * metric * metric
        energy_q = coefficient * tf.pow(speed2, p / 2.0) / p

        def integrate(values: tf.Tensor, shape: tf.Tensor) -> tf.Tensor:
            return self._quad_weight * tf.reduce_sum(values * shape, axis=1)

        energy = integrate(energy_q, tf.ones_like(left))
        grad_left = integrate(gradient_q, left)
        grad_right = integrate(gradient_q, right)
        h_ll = integrate(hessian_q, left * left)
        h_lr = integrate(hessian_q, left * right)
        h_rr = integrate(hessian_q, right * right)
        return energy, grad_left, grad_right, h_ll, h_lr, h_rr

    def _scalar_gravity_terms(
        self, inputs: tf.Tensor, u: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        dtype = self.precision
        params = self._gravity.params
        left = 1.0 - self._xi
        right = self._xi
        u_q = left * u[:, tf.newaxis, :-1] + right * u[:, tf.newaxis, 1:]
        h_q = self._interp_q1(self._field(inputs, "thk"))
        surface_x, _ = self._grad_q1(
            self._field(inputs, "usurf"), self._field(inputs, "dX")
        )
        coefficient = h_q * surface_x
        scale = (
            tf.cast(1e-6, dtype)
            * tf.cast(params.rho, dtype)
            * tf.cast(params.g, dtype)
        )
        energy = scale * self._quad_weight * tf.reduce_sum(
            coefficient * u_q, axis=1
        )
        grad_left = scale * self._quad_weight * tf.reduce_sum(
            coefficient * left, axis=1
        )
        grad_right = scale * self._quad_weight * tf.reduce_sum(
            coefficient * right, axis=1
        )
        return energy, grad_left, grad_right

    def _scalar_floating_terms(
        self, inputs: tf.Tensor, u: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        dtype = self.precision
        params = self._floating.params
        h = self._field(inputs, "thk")
        surface = self._field(inputs, "usurf")
        water = self._field(inputs, "water_level")
        dx = self._field(inputs, "dX")
        bed = surface - h

        h0 = h[:, 0, :]
        bed0 = bed[:, 0, :]
        water0 = water[:, 0, :]
        ice = h0 > 0.0
        land = tf.logical_and(h0 <= 0.0, bed0 > water0)
        cell_ice = tf.logical_and(ice[:, :-1], ice[:, 1:])
        cell_land = tf.logical_or(land[:, :-1], land[:, 1:])
        wet = tf.cast(tf.logical_not(tf.logical_or(cell_ice, cell_land)), dtype)
        edge_w = float("W" in params.cf_eswn)
        edge_e = float("E" in params.cf_eswn)
        wet_padded = tf.pad(wet, [[0, 0], [1, 0]], constant_values=edge_w)
        wet_padded = tf.pad(
            wet_padded, [[0, 0], [0, 1]], constant_values=edge_e
        )
        wet_west = wet_padded[:, :-2]
        wet_east = wet_padded[:, 2:]

        rho = tf.cast(params.rho, dtype)
        rho_water = tf.cast(params.rho_water, dtype)
        gravity = tf.cast(params.g, dtype)
        water_depth = tf.maximum(water0 - bed0, 0.0)
        pressure = 0.5 * gravity * (
            rho * h0 * h0 - rho_water * water_depth * water_depth
        )
        dx_cell = 0.5 * (dx[:, 0, :-1] + dx[:, 1, 1:])
        scale = tf.cast(cell_ice, dtype) / dx_cell
        grad_left = (
            tf.cast(1e-6, dtype) * scale * pressure[:, :-1] * wet_west
        )
        grad_right = (
            -tf.cast(1e-6, dtype) * scale * pressure[:, 1:] * wet_east
        )
        energy = grad_left * u[:, :-1] + grad_right * u[:, 1:]
        return energy, grad_left, grad_right

    def _scalar_system_impl(
        self, inputs: tf.Tensor, u: tf.Tensor, damping: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Exact scalar Q1/SSA energy, gradient, and Hessian bands."""
        U, _ = self._uv_from_u_row(u)
        physical_u = U[:, 0, 0, :]
        shape = (self.B, self.Nx - 1)
        energy = tf.zeros(shape, self.precision)
        grad_left = tf.zeros(shape, self.precision)
        grad_right = tf.zeros(shape, self.precision)
        h_ll = tf.zeros(shape, self.precision)
        h_lr = tf.zeros(shape, self.precision)
        h_rr = tf.zeros(shape, self.precision)

        if self._viscosity is not None:
            terms = self._scalar_viscosity_terms(inputs, physical_u)
            energy += terms[0]
            grad_left += terms[1]
            grad_right += terms[2]
            h_ll += terms[3]
            h_lr += terms[4]
            h_rr += terms[5]
        if self._sliding is not None:
            terms = self._scalar_sliding_terms(inputs, physical_u)
            energy += terms[0]
            grad_left += terms[1]
            grad_right += terms[2]
            h_ll += terms[3]
            h_lr += terms[4]
            h_rr += terms[5]
        if self._gravity is not None:
            terms = self._scalar_gravity_terms(inputs, physical_u)
            energy += terms[0]
            grad_left += terms[1]
            grad_right += terms[2]
        if self._floating is not None:
            terms = self._scalar_floating_terms(inputs, physical_u)
            energy += terms[0]
            grad_left += terms[1]
            grad_right += terms[2]

        cell_mask = compute_cell_ice_mask(self._field(inputs, "thk"))[:, 0, :]
        cell_weight = tf.cast(cell_mask, self.precision) * self._energy_normalization
        cost = tf.reduce_sum(cell_weight * energy)
        gradient = tf.pad(cell_weight * grad_left, [[0, 0], [0, 1]])
        gradient += tf.pad(cell_weight * grad_right, [[0, 0], [1, 0]])

        h_ll *= cell_weight
        h_lr *= cell_weight
        h_rr *= cell_weight
        active = self._active[:, 0, :]
        active_west = tf.pad(active[:, :-1], [[0, 0], [1, 0]])
        active_east = tf.pad(active[:, 1:], [[0, 0], [0, 1]])
        lower = tf.pad(h_lr, [[0, 0], [1, 0]]) * active * active_west
        diagonal = (
            tf.pad(h_ll, [[0, 0], [0, 1]])
            + tf.pad(h_rr, [[0, 0], [1, 0]])
        ) * active * active
        upper = tf.pad(h_lr, [[0, 0], [0, 1]]) * active * active_east
        diagonal += tf.cast(damping, self.precision)
        return cost, gradient * active, lower, diagonal, upper

    @tf.function(reduce_retracing=True)
    def scalar_system_at(
        self, inputs: tf.Tensor, u: tf.Tensor, damping: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return self._scalar_system_impl(inputs, u, damping)

    @tf.function(reduce_retracing=True)
    def scalar_cost_at(self, inputs: tf.Tensor, u: tf.Tensor) -> tf.Tensor:
        cost, _, _, _, _ = self._scalar_system_impl(
            inputs, u, tf.cast(0.0, self.precision)
        )
        return cost

    def _field(self, inputs: tf.Tensor, name: str) -> tf.Tensor:
        return inputs[..., self._input_indices[name]]

    def _friction_field(self, inputs: tf.Tensor) -> tf.Tensor:
        name = "tau_ref" if "tau_ref" in self._input_indices else "slidingco"
        return self._field(inputs, name)

    def _interp_q1(self, field: tf.Tensor) -> tf.Tensor:
        """Q1 interpolation, squeezed to ``(B,4,Nx-1)`` for ``Ny=2``."""
        sw = field[:, :-1, :-1]
        se = field[:, :-1, 1:]
        nw = field[:, 1:, :-1]
        ne = field[:, 1:, 1:]
        xi = self._xi[..., tf.newaxis]
        eta = self._eta[..., tf.newaxis]
        values = (
            (1.0 - xi) * (1.0 - eta) * sw[:, tf.newaxis]
            + xi * (1.0 - eta) * se[:, tf.newaxis]
            + (1.0 - xi) * eta * nw[:, tf.newaxis]
            + xi * eta * ne[:, tf.newaxis]
        )
        return values[:, :, 0, :]

    def _grad_q1(self, field: tf.Tensor, dx: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Q1 gradients, squeezed to ``(B,4,Nx-1)`` for ``Ny=2``."""
        sw = field[:, :-1, :-1]
        se = field[:, :-1, 1:]
        nw = field[:, 1:, :-1]
        ne = field[:, 1:, 1:]
        inverse_dx = tf.math.reciprocal(dx[:, 1:, 1:])
        grad_x_s = (se - sw) * inverse_dx
        grad_x_n = (ne - nw) * inverse_dx
        grad_y_w = (nw - sw) * inverse_dx
        grad_y_e = (ne - se) * inverse_dx
        eta = self._eta[..., tf.newaxis]
        xi = self._xi[..., tf.newaxis]
        grad_x = (
            (1.0 - eta) * grad_x_s[:, tf.newaxis]
            + eta * grad_x_n[:, tf.newaxis]
        )
        grad_y = (
            (1.0 - xi) * grad_y_w[:, tf.newaxis]
            + xi * grad_y_e[:, tf.newaxis]
        )
        return grad_x[:, :, 0, :], grad_y[:, :, 0, :]

    def _viscosity_element_blocks(
        self,
        inputs: tf.Tensor,
        u: tf.Tensor,
        v: tf.Tensor,
        cell_weight: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        params = self._viscosity.params
        dtype = self.precision
        n = tf.cast(params.n, dtype)
        regularization2 = tf.cast(params.eps_dot_regu, dtype) ** 2
        maximum2 = tf.cast(params.eps_dot_max, dtype) ** 2
        p = 1.0 + 1.0 / n
        exponent = (p - 2.0) / 2.0

        h = self._field(inputs, "thk")
        arrhenius = self._field(inputs, "arrhenius")
        dx = self._field(inputs, "dX")[:, 1:, 1:][:, 0, :]
        stiffness = 2.0 * tf.pow(arrhenius, -1.0 / n)
        coefficient = tf.reduce_sum(
            self._interp_q1(h) * self._interp_q1(stiffness), axis=1
        ) * self._quad_weight

        du = (u[:, 1:] - u[:, :-1]) / dx
        dv = (v[:, 1:] - v[:, :-1]) / dx
        strain2 = du * du + tf.cast(0.25, dtype) * dv * dv
        capped = tf.minimum(strain2, maximum2)
        base = capped + regularization2

        f_prime = (
            tf.pow(base, exponent)
            + exponent * capped * tf.pow(base, exponent - 1.0)
        ) / p
        f_second = (
            2.0 * exponent * tf.pow(base, exponent - 1.0)
            + exponent
            * (exponent - 1.0)
            * capped
            * tf.pow(base, exponent - 2.0)
        ) / p
        cap_jacobian = tf.cast(strain2 <= maximum2, dtype)
        f_prime *= cap_jacobian
        f_second *= cap_jacobian

        q_u = du
        q_v = tf.cast(0.25, dtype) * dv
        h_uu = 2.0 * f_prime + 4.0 * f_second * q_u * q_u
        h_uv = 4.0 * f_second * q_u * q_v
        h_vv = tf.cast(0.5, dtype) * f_prime + 4.0 * f_second * q_v * q_v
        local = _outer_blocks(h_uu, h_uv, h_vv)
        local *= (coefficient * cell_weight / (dx * dx))[:, tf.newaxis, tf.newaxis]
        return local, -local, -local, local

    def _sliding_coefficient(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return the power-law coefficient at the four Q1 points."""
        component = self._sliding
        params = component.params
        dtype = self.precision
        h = self._field(inputs, "thk")
        surface = self._field(inputs, "usurf")
        bed = surface - h

        if component.name == "mohr_coulomb":
            effective_pressure = self._field(inputs, "effective_pressure")
            bed_min = tf.cast(params.bed_min, dtype)
            bed_max = tf.cast(params.bed_max, dtype)
            phi_uniform = tf.cast(params.phi, dtype)
            phi_lo = tf.cast(params.phi_min, dtype)
            phi_hi = tf.cast(params.phi_max, dtype)
            phi_interp = phi_lo + (phi_hi - phi_lo) * (bed - bed_min) / (
                bed_max - bed_min
            )
            phi_interp = tf.where(
                bed <= bed_min,
                phi_lo,
                tf.where(bed >= bed_max, phi_hi, phi_interp),
            )
            use_interp = tf.math.is_finite(bed_min) & tf.math.is_finite(bed_max)
            phi = tf.where(use_interp, phi_interp, phi_uniform * tf.ones_like(bed))
            stress = effective_pressure * tf.math.tan(
                phi * tf.cast(math.pi / 180.0, dtype)
            )
            stress = tf.where(h > 0.0, stress, tf.cast(params.tauc_ice_free, dtype))
            stress = tf.clip_by_value(
                stress,
                tf.cast(params.tauc_min, dtype),
                tf.cast(params.tauc_max, dtype),
            )
        else:
            stress = self._friction_field(inputs)

        if bool(params.use_mask_gr):
            grounded = h + tf.cast(params.rho_ratio, dtype) * bed > 0.0
            stress *= tf.cast(grounded, dtype)

        coefficient = self._interp_q1(stress) / tf.pow(
            tf.cast(params.u_ref, dtype), 1.0 / tf.cast(params.exponent, dtype)
        )
        if component.name == "budd":
            effective_pressure = tf.maximum(
                self._field(inputs, "effective_pressure"), tf.cast(1e-3, dtype)
            )
            pressure_q = self._interp_q1(effective_pressure)
            q = tf.cast(params.q_exponent, dtype)
            coefficient *= tf.pow(
                pressure_q / tf.cast(params.N_ref, dtype), q
            )
        return coefficient

    def _sliding_element_blocks(
        self,
        inputs: tf.Tensor,
        u: tf.Tensor,
        v: tf.Tensor,
        cell_weight: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        component = self._sliding
        params = component.params
        dtype = self.precision
        p = 1.0 + 1.0 / tf.cast(params.exponent, dtype)

        left_shape = 1.0 - self._xi
        right_shape = self._xi
        u_q = left_shape * u[:, tf.newaxis, :-1] + right_shape * u[:, tf.newaxis, 1:]
        v_q = left_shape * v[:, tf.newaxis, :-1] + right_shape * v[:, tf.newaxis, 1:]
        bed = self._field(inputs, "usurf") - self._field(inputs, "thk")
        bed_x, bed_y = self._grad_q1(bed, self._field(inputs, "dX"))

        correction = u_q * bed_x + v_q * bed_y
        metric_u = u_q + bed_x * correction
        metric_v = v_q + bed_y * correction
        speed2 = (
            u_q * u_q
            + v_q * v_q
            + tf.cast(params.regularization, dtype) ** 2
            + correction * correction
        )
        metric_uu = 1.0 + bed_x * bed_x
        metric_uv = bed_x * bed_y
        metric_vv = 1.0 + bed_y * bed_y

        if component.name in self._POWER_LAW_SLIDING:
            coefficient = self._sliding_coefficient(inputs)
            radial_first_twice = coefficient * tf.pow(speed2, p / 2.0 - 1.0)
            radial_second_four = (
                coefficient
                * (p - 2.0)
                * tf.pow(speed2, p / 2.0 - 2.0)
            )
        else:
            # F(t)=tau_c*((t^(p/2)+u_c^p)^(1/p)-u_c), t=speed^2.
            # Hessian_x F = 2 F'(t) M + 4 F''(t) (Mx)(Mx)^T.
            effective_pressure = self._interp_q1(
                self._field(inputs, "effective_pressure")
            )
            tau_c = tf.cast(params.mu, dtype) * effective_pressure
            friction = self._interp_q1(self._friction_field(inputs))
            if bool(params.use_mask_gr):
                h = self._field(inputs, "thk")
                bed_nodes = self._field(inputs, "usurf") - h
                grounded = h + tf.cast(params.rho_ratio, dtype) * bed_nodes > 0.0
                friction = self._interp_q1(
                    self._friction_field(inputs) * tf.cast(grounded, dtype)
                )
            base_coefficient = friction / tf.pow(
                tf.cast(params.u_ref, dtype), 1.0 / tf.cast(params.exponent, dtype)
            )
            critical_speed = tf.pow(
                tau_c / base_coefficient, tf.cast(params.exponent, dtype)
            )
            transition = tf.pow(speed2, p / 2.0) + tf.pow(critical_speed, p)
            f_prime = (
                tau_c
                * tf.cast(0.5, dtype)
                * tf.pow(transition, 1.0 / p - 1.0)
                * tf.pow(speed2, p / 2.0 - 1.0)
            )
            f_second = tau_c * tf.cast(0.5, dtype) * (
                (1.0 / p - 1.0)
                * tf.pow(transition, 1.0 / p - 2.0)
                * (p / 2.0)
                * tf.pow(speed2, p - 2.0)
                + tf.pow(transition, 1.0 / p - 1.0)
                * (p / 2.0 - 1.0)
                * tf.pow(speed2, p / 2.0 - 2.0)
            )
            radial_first_twice = 2.0 * f_prime
            radial_second_four = 4.0 * f_second

        h_uu = radial_first_twice * metric_uu + radial_second_four * metric_u * metric_u
        h_uv = radial_first_twice * metric_uv + radial_second_four * metric_u * metric_v
        h_vv = radial_first_twice * metric_vv + radial_second_four * metric_v * metric_v
        hessian_q = _outer_blocks(h_uu, h_uv, h_vv)
        weights = self._quad_weight * cell_weight[:, tf.newaxis, :]

        def integrate(shape_a: tf.Tensor, shape_b: tf.Tensor) -> tf.Tensor:
            weighted = hessian_q * (
                weights * shape_a * shape_b
            )[:, tf.newaxis, tf.newaxis, :, :]
            return tf.reduce_sum(weighted, axis=3)

        return (
            integrate(left_shape, left_shape),
            integrate(left_shape, right_shape),
            integrate(right_shape, left_shape),
            integrate(right_shape, right_shape),
        )

    @tf.function(reduce_retracing=True)
    def _extract_bands(self, inputs: tf.Tensor) -> tf.Tensor:
        theta_flat = self.map.flatten_theta(self.map.get_theta())
        return self._extract_bands_at(inputs, theta_flat)

    @tf.function(reduce_retracing=True)
    def _extract_bands_at(
        self, inputs: tf.Tensor, theta_flat: tf.Tensor
    ) -> tf.Tensor:
        """Assemble bands as a pure function of the current velocity.

        The stateful ``prepare`` API still uses ``_extract_bands``.  This
        variant lets a complete Newton solve be XLA-compiled without reading
        or updating the mapping and band-storage resources at every nonlinear
        iteration.
        """
        U, V = self.map.unflatten_theta(theta_flat)
        for apply_bc in self.map.apply_bcs:
            U, V = apply_bc(U, V)
        u = U[:, 0, 0, :]
        v = V[:, 0, 0, :]

        cell_mask = compute_cell_ice_mask(self._field(inputs, "thk"))[:, 0, :]
        cell_weight = tf.cast(cell_mask, self.precision) * self._energy_normalization
        zeros = tf.zeros((self.B, 2, 2, self.Nx - 1), self.precision)
        ll = lr = rl = rr = zeros
        if self._viscosity is not None:
            blocks = self._viscosity_element_blocks(inputs, u, v, cell_weight)
            ll, lr, rl, rr = tuple(a + b for a, b in zip((ll, lr, rl, rr), blocks))
        if self._sliding is not None:
            blocks = self._sliding_element_blocks(inputs, u, v, cell_weight)
            ll, lr, rl, rr = tuple(a + b for a, b in zip((ll, lr, rl, rr), blocks))

        spatial_padding_left = [[0, 0], [0, 0], [0, 0], [1, 0]]
        spatial_padding_right = [[0, 0], [0, 0], [0, 0], [0, 1]]
        west = tf.pad(rl, spatial_padding_left)
        center = tf.pad(ll, spatial_padding_right) + tf.pad(
            rr, spatial_padding_left
        )
        east = tf.pad(lr, spatial_padding_right)

        def apply_active(block: tf.Tensor, neighbour: tf.Tensor) -> tf.Tensor:
            return (
                block
                * self._active[:, :, tf.newaxis, :]
                * neighbour[:, tf.newaxis, :, :]
            )

        active_west = tf.pad(self._active[:, :, :-1], [[0, 0], [0, 0], [1, 0]])
        active_east = tf.pad(self._active[:, :, 1:], [[0, 0], [0, 0], [0, 1]])
        west = apply_active(west, active_west)
        center = apply_active(center, self._active)
        east = apply_active(east, active_east)
        return tf.stack([west, center, east], axis=0)

    def assemble_bands_at(
        self,
        inputs: tf.Tensor,
        theta_flat: tf.Tensor,
        damping: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Return analytic bands without touching ``self._bands``."""
        bands = self._extract_bands_at(inputs, theta_flat)
        identity = tf.eye(2, dtype=self.precision)[tf.newaxis, :, :, tf.newaxis]
        return {
            "west": bands[0],
            "center": bands[1] + tf.cast(damping, self.precision) * identity,
            "east": bands[2],
        }
