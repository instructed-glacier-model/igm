#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Unit tests for vertical enthalpy solver components."""

import pytest
import numpy as np
import tensorflow as tf

from igm.processes.enthalpy.solver.vertical.utils.assembly import assemble_system
from igm.processes.enthalpy.solver.vertical.utils.solver import solve_tridiagonal_system

# Data type
dtype = tf.float32


@pytest.fixture
def simple_grid():
    """Simple 5-level grid for testing."""
    nz, ny, nx = 5, 2, 2
    return {
        "E": tf.ones((nz, ny, nx), dtype=dtype) * 1000.0,
        "dt": tf.constant(1.0, dtype=dtype),
        "dz": tf.ones((nz - 1, ny, nx), dtype=dtype) * 100.0,
        "w": tf.zeros((nz, ny, nx), dtype=dtype),
        "K": tf.ones((nz - 1, ny, nx), dtype=dtype) * 1e-6,
        "f": tf.zeros((nz, ny, nx), dtype=dtype),
        "BCB": tf.ones((ny, nx), dtype=dtype),
        "VB": tf.zeros((ny, nx), dtype=dtype),
        "VS": tf.zeros((ny, nx), dtype=dtype),
    }


def test_assembly_shapes(simple_grid) -> None:
    """Test output shapes of assembled system."""
    L, M, U, R = assemble_system(**simple_grid)

    nz, ny, nx = 5, 2, 2
    assert L.shape == (nz - 1, ny, nx)
    assert M.shape == (nz, ny, nx)
    assert U.shape == (nz - 1, ny, nx)
    assert R.shape == (nz, ny, nx)


@pytest.mark.parametrize("bc_type", [0, 1])
def test_assembly_boundary_conditions(simple_grid, bc_type: int) -> None:
    """Test Dirichlet (0) and Neumann (1) BC assembly."""
    simple_grid["BCB"] = tf.ones((2, 2), dtype=dtype) * bc_type
    L, M, U, R = assemble_system(**simple_grid)

    if bc_type == 0:  # Dirichlet
        np.testing.assert_array_equal(M[0].numpy(), 1.0)
        np.testing.assert_array_equal(U[0].numpy(), 0.0)
    else:  # Neumann
        np.testing.assert_array_equal(M[0].numpy(), -1.0)
        np.testing.assert_array_equal(U[0].numpy(), 1.0)


def test_tridiagonal_solver_identity() -> None:
    """Test solver with identity system (M=1, L=U=0)."""
    nz, ny, nx = 5, 2, 2
    L = tf.zeros((nz - 1, ny, nx), dtype=dtype)
    M = tf.ones((nz, ny, nx), dtype=dtype)
    U = tf.zeros((nz - 1, ny, nx), dtype=dtype)
    R = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype)
    R = tf.reshape(R, (nz, 1, 1)) * tf.ones((1, ny, nx), dtype=dtype)

    solution = solve_tridiagonal_system(L, M, U, R)

    solution_expected = R

    np.testing.assert_allclose(solution.numpy(), solution_expected.numpy(), rtol=1e-5)


def test_tridiagonal_solver_simple() -> None:
    """Test solver with simple tridiagonal system."""
    nz, ny, nx = 3, 1, 1
    # System: [2, -1, 0; -1, 2, -1; 0, -1, 2] * x = [0, 1, 0]
    # Solution: x = [0.5, 1, 0.5]
    L = tf.constant([[-1.0], [-1.0]], dtype=dtype)[:, :, None]
    M = tf.constant([[2.0], [2.0], [2.0]], dtype=dtype)[:, :, None]
    U = tf.constant([[-1.0], [-1.0]], dtype=dtype)[:, :, None]
    R = tf.constant([[0.0], [1.0], [0.0]], dtype=dtype)[:, :, None]

    solution = solve_tridiagonal_system(L, M, U, R)

    solution_expected = np.array([[[0.5]], [[1.0]], [[0.5]]])

    np.testing.assert_allclose(solution.numpy(), solution_expected, rtol=1e-5)
