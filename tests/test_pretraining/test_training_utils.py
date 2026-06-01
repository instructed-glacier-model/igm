"""
Unit tests for igm.processes.pretraining.training_utils

Covers:
  - validate_dataset_matches_inputs: channel-count and name-order validation
  - build_velocity_data_loss: error paths and numerical behaviour (MSE vs Huber)
"""

from __future__ import annotations

import pytest
import tensorflow as tf

from igm.processes.pretraining.training_utils import (
    build_velocity_data_loss,
    validate_dataset_matches_inputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uv_ybatch(error_u: float, error_v: float, shape=(1, 1, 1, 1)):
    """
    Build (U, V, y_batch) such that:
      U_pred - U_true = error_u   (uniformly)
      V_pred - V_true = error_v   (uniformly)
    """
    U = tf.ones(shape, dtype=tf.float32)
    V = tf.ones(shape, dtype=tf.float32)
    U_target = U - error_u
    V_target = V - error_v
    y_batch = tf.stack([U_target, V_target], axis=-1)   # [..., 2]
    return U, V, y_batch


# ---------------------------------------------------------------------------
# validate_dataset_matches_inputs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_validate_cx_mismatch():
    """`Cx` not matching the number of input names raises ValueError."""
    with pytest.raises(ValueError, match=r"C=3"):
        validate_dataset_matches_inputs(inputs=["thk", "usurf"], Cx=3)


@pytest.mark.unit
def test_validate_name_mismatch():
    """Matching `Cx` but mismatched `metadata_inputs` raises ValueError."""
    with pytest.raises(ValueError, match=r"declares input_names"):
        validate_dataset_matches_inputs(
            inputs=["thk", "usurf"],
            Cx=2,
            metadata_inputs=["thk", "arrhenius"],
        )


@pytest.mark.unit
def test_validate_name_order_matters():
    """Same names in a different order must raise ValueError."""
    with pytest.raises(ValueError):
        validate_dataset_matches_inputs(
            inputs=["usurf", "thk"],
            Cx=2,
            metadata_inputs=["thk", "usurf"],
        )


@pytest.mark.unit
def test_validate_no_metadata_happy():
    """Correct `Cx` with `metadata_inputs=None` passes without error."""
    validate_dataset_matches_inputs(inputs=["thk", "usurf", "slidingco"], Cx=3)


@pytest.mark.unit
def test_validate_full_match_happy():
    """Correct `Cx` and matching `metadata_inputs` (same order) passes."""
    validate_dataset_matches_inputs(
        inputs=["thk", "usurf", "slidingco"],
        Cx=3,
        metadata_inputs=["thk", "usurf", "slidingco"],
    )


# ---------------------------------------------------------------------------
# build_velocity_data_loss — error paths
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_unknown_loss_type():
    """`loss_type` outside {'mse', 'huber'} raises ValueError."""
    with pytest.raises(ValueError, match=r"loss_type"):
        build_velocity_data_loss(loss_type="l1")


# ---------------------------------------------------------------------------
# build_velocity_data_loss — MSE numerical behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_mse_perfect_prediction():
    """Loss is (near) zero when prediction equals target."""
    fn = build_velocity_data_loss(loss_type="mse")
    U, V, y_batch = _uv_ybatch(error_u=0.0, error_v=0.0)
    loss = fn(U, V, y_batch).numpy()
    assert loss == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_mse_known_error():
    """MSE loss matches the analytical value for known constant errors.

    U_error=1, V_error=2  →  loss = mean((1^2 + 2^2)) = 5.0
    """
    fn = build_velocity_data_loss(loss_type="mse")
    U, V, y_batch = _uv_ybatch(error_u=1.0, error_v=2.0)
    loss = fn(U, V, y_batch).numpy()
    assert loss == pytest.approx(5.0, rel=1e-5)


# ---------------------------------------------------------------------------
# build_velocity_data_loss — Huber numerical behaviour
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_huber_small_error():
    """In the quadratic regime (|error| << delta) Huber behaves like 0.5 * MSE."""
    delta = 10.0
    error = 0.5          # well below delta
    fn_huber = build_velocity_data_loss(loss_type="huber", huber_delta=delta)
    fn_mse   = build_velocity_data_loss(loss_type="mse")
    U, V, y_batch = _uv_ybatch(error_u=error, error_v=0.0)
    huber_loss = fn_huber(U, V, y_batch).numpy()
    mse_loss   = fn_mse(U, V, y_batch).numpy()
    # Huber = 0.5 * error^2 per element; MSE = error^2; so huber ≈ 0.5 * mse
    assert huber_loss == pytest.approx(0.5 * mse_loss, rel=1e-4)


@pytest.mark.unit
def test_huber_large_error():
    """In the linear regime (|error| >> delta) Huber is much smaller than MSE."""
    delta = 1.0
    error = 20.0         # well above delta
    fn_huber = build_velocity_data_loss(loss_type="huber", huber_delta=delta)
    fn_mse   = build_velocity_data_loss(loss_type="mse")
    U, V, y_batch = _uv_ybatch(error_u=error, error_v=error)
    huber_loss = fn_huber(U, V, y_batch).numpy()
    mse_loss   = fn_mse(U, V, y_batch).numpy()
    assert huber_loss < mse_loss / 10.0, (
        f"Huber loss ({huber_loss:.4f}) should be << MSE ({mse_loss:.4f}) for large errors"
    )


@pytest.mark.unit
def test_huber_delta_captured_in_closure():
    """Two functions built with different deltas produce different losses on the same input.

    With error=50:
      fn_tight (delta=1)  → linear regime  → loss ≈  2*(1*50 - 0.5) = 99
      fn_loose (delta=100) → quadratic regime → loss ≈ 2*(0.5*50^2) = 2500
    """
    fn_tight = build_velocity_data_loss(loss_type="huber", huber_delta=1.0)
    fn_loose = build_velocity_data_loss(loss_type="huber", huber_delta=100.0)
    U, V, y_batch = _uv_ybatch(error_u=50.0, error_v=50.0)
    loss_tight = fn_tight(U, V, y_batch).numpy()
    loss_loose = fn_loose(U, V, y_batch).numpy()
    assert loss_tight < loss_loose, (
        f"Tight-delta loss ({loss_tight:.2f}) should be < loose-delta loss ({loss_loose:.2f})"
    )
