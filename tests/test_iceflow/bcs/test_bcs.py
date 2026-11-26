#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf

from igm.processes.iceflow.bcs import (
    BoundaryCondition,
    BoundaryConditions,
    FrozenBed,
    InterfaceBoundaryCondition,
    InterfaceBoundaryConditions,
    PeriodicNS,
    PeriodicNSGlobal,
    PeriodicWE,
    PeriodicWEGlobal,
)


def create_UV(shape: Tuple[int, ...] = (1, 3, 3, 1)) -> Tuple[tf.Tensor, tf.Tensor]:
    size = np.prod(shape)
    arr = np.arange(size).reshape(shape).astype(np.float32)
    return tf.constant(arr), tf.constant(arr + 100.0)


def test_base_class_abstract() -> None:
    with pytest.raises(TypeError):
        BoundaryCondition()


def test_base_class_apply() -> None:

    class IncompleteBC(BoundaryCondition):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteBC()


def test_registry_names() -> None:
    assert "frozen_bed" in BoundaryConditions
    assert "periodic_ns" in BoundaryConditions
    assert "periodic_we" in BoundaryConditions
    assert "periodic_ns_global" in BoundaryConditions
    assert "periodic_we_global" in BoundaryConditions
    assert BoundaryConditions["frozen_bed"] == FrozenBed
    assert BoundaryConditions["periodic_ns"] == PeriodicNS
    assert BoundaryConditions["periodic_we"] == PeriodicWE
    assert BoundaryConditions["periodic_ns_global"] == PeriodicNSGlobal
    assert BoundaryConditions["periodic_we_global"] == PeriodicWEGlobal


def test_frozen_bed_validation() -> None:
    """Test that FrozenBed raises error when V_b[0] == 0."""
    Nz = 3
    V_b = tf.one_hot(0, Nz)
    bc = FrozenBed(V_b)

    V_b_invalid = tf.constant([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="frozen bed BC requires V_b ≠ 0"):
        FrozenBed(V_b_invalid)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 3, 3, 1),
        (2, 5, 1, 1),
        (1, 2, 7, 2),
        (1, 5, 2, 3),
    ],
)
def test_frozen_bed(shape: Tuple[int, ...]) -> None:
    """Test that FrozenBed correctly computes basal velocity using weights."""
    Nz = shape[1]
    V_b = tf.one_hot(0, Nz)
    bc = BoundaryConditions["frozen_bed"](V_b)
    U_in, V_in = create_UV(shape)
    U_out, V_out = bc.apply(U_in, V_in)

    np.testing.assert_array_equal(U_out[:, 1:, :, :], U_in[:, 1:, :, :])
    np.testing.assert_array_equal(V_out[:, 1:, :, :], V_in[:, 1:, :, :])

    expected_U0 = tf.einsum("i,bijk->bjk", bc.weights, U_in[:, 1:, :, :])
    expected_V0 = tf.einsum("i,bijk->bjk", bc.weights, V_in[:, 1:, :, :])

    np.testing.assert_allclose(U_out[:, 0, :, :], expected_U0)
    np.testing.assert_allclose(V_out[:, 0, :, :], expected_V0)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 3, 3, 1),
        (2, 5, 1, 1),
        (1, 1, 7, 2),
        (1, 5, 2, 3),
    ],
)
def test_periodic_NS(shape: Tuple[int, ...]) -> None:
    """Test PeriodicNS replaces last row with first row."""
    bc = BoundaryConditions["periodic_ns"]()
    U_in, V_in = create_UV(shape)
    U_out, V_out = bc.apply(U_in, V_in)

    np.testing.assert_array_equal(U_out[:, :, -1, :], U_in[:, :, 0, :])
    np.testing.assert_array_equal(V_out[:, :, -1, :], V_in[:, :, 0, :])

    np.testing.assert_array_equal(U_out[:, :, :-1, :], U_in[:, :, :-1, :])
    np.testing.assert_array_equal(V_out[:, :, :-1, :], V_in[:, :, :-1, :])


@pytest.mark.parametrize(
    "shape",
    [
        (4, 3, 3, 1),
        (2, 5, 1, 1),
        (1, 1, 7, 2),
        (1, 5, 2, 3),
    ],
)
def test_periodic_WE(shape: Tuple[int, ...]) -> None:
    """Test PeriodicWE replaces last column with first column."""
    bc = BoundaryConditions["periodic_we"]()
    U_in, V_in = create_UV(shape)
    U_out, V_out = bc.apply(U_in, V_in)

    np.testing.assert_array_equal(U_out[:, :, :, -1], U_in[:, :, :, 0])
    np.testing.assert_array_equal(V_out[:, :, :, -1], V_in[:, :, :, 0])

    np.testing.assert_array_equal(U_out[:, :, :, :-1], U_in[:, :, :, :-1])
    np.testing.assert_array_equal(V_out[:, :, :, :-1], V_in[:, :, :, :-1])


@pytest.mark.parametrize(
    "Nx,Ny,Nz",
    [
        (3, 3, 3),
        (5, 1, 5),
        (1, 7, 1),
        (2, 5, 2),
    ],
)
def test_periodic_NS_global(Nx: int, Ny: int, Nz: int) -> None:
    """Test PeriodicNSGlobal with reshaping (batch_size=1 case)."""
    bc = BoundaryConditions["periodic_ns_global"](Nx, Ny, Nz)

    shape = (1, Nz, Ny, Nx)
    U_in, V_in = create_UV(shape)
    U_out, V_out = bc.apply(U_in, V_in)

    assert U_out.shape == U_in.shape
    assert V_out.shape == V_in.shape

    np.testing.assert_array_equal(U_out[:, :, -1, :], U_in[:, :, 0, :])
    np.testing.assert_array_equal(V_out[:, :, -1, :], V_in[:, :, 0, :])

    np.testing.assert_array_equal(U_out[:, :, :-1, :], U_in[:, :, :-1, :])
    np.testing.assert_array_equal(V_out[:, :, :-1, :], V_in[:, :, :-1, :])


@pytest.mark.parametrize(
    "Nx,Ny,Nz",
    [
        (3, 3, 3),
        (5, 1, 5),
        (1, 7, 1),
        (2, 5, 2),
    ],
)
def test_periodic_WE_global(Nx: int, Ny: int, Nz: int) -> None:
    """Test PeriodicWEGlobal with reshaping (batch_size=1 case)."""
    bc = BoundaryConditions["periodic_we_global"](Nx, Ny, Nz)

    shape = (1, Nz, Ny, Nx)
    U_in, V_in = create_UV(shape)
    U_out, V_out = bc.apply(U_in, V_in)

    assert U_out.shape == U_in.shape
    assert V_out.shape == V_in.shape

    np.testing.assert_array_equal(U_out[:, :, :, -1], U_in[:, :, :, 0])
    np.testing.assert_array_equal(V_out[:, :, :, -1], V_in[:, :, :, 0])

    np.testing.assert_array_equal(U_out[:, :, :, :-1], U_in[:, :, :, :-1])
    np.testing.assert_array_equal(V_out[:, :, :, :-1], V_in[:, :, :, :-1])


def test_interface_registry_names() -> None:
    """Test that all interface boundary conditions are registered."""
    assert "frozen_bed" in InterfaceBoundaryConditions
    assert "periodic_ns" in InterfaceBoundaryConditions
    assert "periodic_we" in InterfaceBoundaryConditions
    assert "periodic_ns_global" in InterfaceBoundaryConditions
    assert "periodic_we_global" in InterfaceBoundaryConditions
