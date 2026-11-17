#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Physical regression checks for nonuniform vertical heat diffusion."""

import numpy as np
import pytest
import tensorflow as tf

from igm.processes.enthalpy.solver.vertical.utils.assembly import assemble_system
from igm.processes.enthalpy.solver.vertical.utils.solver import solve_tridiagonal_system

SECONDS_PER_YEAR = 31556926.0
HEAT_CAPACITY = 2009.0
ICE_DENSITY = 910.0
CONDUCTIVITY = 2.1


def _grid(stretch, nz=16):
    xi = np.linspace(0.0, 1.0, nz)
    return 1000.0 * (xi / stretch + (1.0 - 1.0 / stretch) * xi**2)


def _column(z, enthalpy, diffusivity=None, basal_gradient=0.0, basal_value=None):
    """One cold column with fixed surface enthalpy and zero advection/heating."""
    if diffusivity is None:
        diffusivity = np.full(len(z) - 1, CONDUCTIVITY / (ICE_DENSITY * HEAT_CAPACITY))

    def vector(values):
        return tf.constant(np.asarray(values)[:, None, None], tf.float32)

    def scalar(value):
        return tf.constant([[value]], tf.float32)

    energy = vector(enthalpy)
    return dict(
        E=energy,
        dt=tf.constant(100.0 * SECONDS_PER_YEAR, tf.float32),
        dz=vector(np.diff(z)),
        w=tf.zeros_like(energy),
        K=vector(diffusivity),
        f=tf.zeros_like(energy),
        BCB=scalar(1.0 if basal_value is None else 0.0),
        VB=scalar(basal_gradient if basal_value is None else basal_value),
        VS=scalar(enthalpy[-1]),
    )


def _step(column):
    return solve_tridiagonal_system(*assemble_system(**column)).numpy()[:, 0, 0]


def _steady_solution(column):
    """Remove the interior transient mass term to solve the static equations."""
    lower, diagonal, upper, rhs = assemble_system(**column)
    matrix = (
        np.diag(diagonal.numpy()[:, 0, 0].astype(float))
        + np.diag(lower.numpy()[:, 0, 0], -1)
        + np.diag(upper.numpy()[:, 0, 0], 1)
    )
    interior = np.arange(1, len(matrix) - 1)
    matrix[interior, interior] -= 1.0
    rhs = rhs.numpy()[:, 0, 0].astype(float)
    rhs[1:-1] = 0.0
    return np.linalg.solve(matrix, rhs)


@pytest.mark.parametrize("stretch", [1.0, 4.0])
@pytest.mark.parametrize("basal_bc", ["gradient", "dirichlet"])
def test_exact_linear_conduction_profile_is_stationary(stretch, basal_bc):
    z = _grid(stretch)
    heat_flux = 0.04
    enthalpy = HEAT_CAPACITY * (-30.0 + heat_flux * (1000.0 - z) / CONDUCTIVITY + 50.0)
    column = _column(
        z,
        enthalpy,
        basal_gradient=-HEAT_CAPACITY * heat_flux / CONDUCTIVITY,
        basal_value=enthalpy[0] if basal_bc == "dirichlet" else None,
    )
    np.testing.assert_allclose(
        _step(column) / HEAT_CAPACITY, enthalpy / HEAT_CAPACITY, rtol=0, atol=3e-4
    )


@pytest.mark.parametrize("stretch", [1.0, 4.0])
def test_constant_flux_with_discontinuous_conductivity(stretch):
    z = _grid(stretch)
    diffusivity = np.where(np.arange(15) < 7, 1e-6, 2e-6)
    flux = -2e-5
    resistance = np.r_[np.cumsum((np.diff(z) / diffusivity)[::-1])[::-1], 0.0]
    enthalpy = HEAT_CAPACITY * 20.0 - flux * resistance
    column = _column(z, enthalpy, diffusivity, basal_gradient=flux / diffusivity[0])
    np.testing.assert_allclose(
        _step(column) / HEAT_CAPACITY, enthalpy / HEAT_CAPACITY, rtol=0, atol=4e-4
    )


@pytest.mark.parametrize("stretch", [1.0, 4.0])
def test_cold_steady_column_basal_temperature(stretch):
    z = _grid(stretch)
    heat_flux = 0.04
    column = _column(
        z,
        np.full(16, HEAT_CAPACITY * 20.0),
        basal_gradient=-HEAT_CAPACITY * heat_flux / CONDUCTIVITY,
    )
    temperature = _steady_solution(column) / HEAT_CAPACITY - 50.0
    expected = -30.0 + heat_flux * (1000.0 - z) / CONDUCTIVITY
    np.testing.assert_allclose(temperature, expected, atol=0.003, rtol=0)


@pytest.mark.parametrize("stretch", [1.0, 4.0])
def test_wet_bed_can_refreeze_in_a_negative_heat_budget(stretch):
    z = _grid(stretch)
    melting_point = -7.9e-8 * ICE_DENSITY * 9.81 * 1000.0
    enthalpy = HEAT_CAPACITY * (-30.0 + (melting_point + 30.0) * (1000.0 - z) / 1000.0 + 50.0)
    result = _steady_solution(_column(z, enthalpy, basal_value=enthalpy[0]))
    upward_flux = -CONDUCTIVITY / HEAT_CAPACITY * (result[1] - result[0]) / (z[1] - z[0])
    correct_flux = CONDUCTIVITY * (melting_point + 30.0) / 1000.0
    np.testing.assert_allclose(upward_flux, correct_flux, rtol=3e-4)
    melt = (0.04 - upward_flux) / (ICE_DENSITY * 334000.0) * SECONDS_PER_YEAR
    assert melt < 0.0, "This column must freeze, not melt"


@pytest.mark.parametrize("stretch", [1.0, 4.0])
def test_interior_energy_budget_matches_boundary_fluxes(stretch):
    z = _grid(stretch)
    dz = np.diff(z)
    cell_width = 0.5 * (dz[:-1] + dz[1:])
    enthalpy = HEAT_CAPACITY * (30.0 + 5.0 * np.sin(np.pi * z / 1000.0))
    column = _column(z, enthalpy, basal_value=enthalpy[0])
    new = _step(column).astype(float)
    old = column["E"].numpy()[:, 0, 0].astype(float)
    diffusivity = column["K"].numpy()[:, 0, 0].astype(float)
    flux = diffusivity * np.diff(new) / dz
    change = np.sum(cell_width * (new[1:-1] - old[1:-1])) / float(column["dt"])
    net_flux = flux[-1] - flux[0]
    assert abs(change - net_flux) < 2e-8, (change, net_flux)


@pytest.mark.parametrize("nz", [2, 3, 16])
def test_small_grids_and_batched_columns(nz):
    z = _grid(4.0, nz)
    enthalpy = HEAT_CAPACITY * (20.0 + 10.0 * (1.0 - z / 1000.0))
    column = _column(z, enthalpy, basal_value=enthalpy[0])
    for key in ("E", "dz", "w", "K", "f"):
        column[key] = tf.tile(column[key], [1, 2, 3])
    for key in ("BCB", "VB", "VS"):
        column[key] = tf.tile(column[key], [2, 3])
    result = solve_tridiagonal_system(*assemble_system(**column)).numpy()
    assert result.shape == (nz, 2, 3)
    np.testing.assert_allclose(
        result / HEAT_CAPACITY, column["E"].numpy() / HEAT_CAPACITY, atol=4e-4, rtol=0
    )
