"""
Unit tests for private helpers in igm.assimilations.pretraining.pretraining

Covers:
  - _resolve_accum_steps: batch-size / divisibility validation
  - _prepare_run_dirs: directory creation and resume consistency checks
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from igm.assimilations.pretraining.pretraining import (
    _prepare_run_dirs,
    _resolve_accum_steps,
)


# ---------------------------------------------------------------------------
# _resolve_accum_steps
# ---------------------------------------------------------------------------

def _cfg(batch_size: int, micro_batch_size: int):
    return OmegaConf.create({"batch_size": batch_size, "micro_batch_size": micro_batch_size})


@pytest.mark.unit
def test_accum_steps_valid_exact():
    """batch == micro → accum_steps = 1."""
    effective_bs, micro_bs, accum = _resolve_accum_steps(_cfg(8, 8))
    assert (effective_bs, micro_bs, accum) == (8, 8, 1)


@pytest.mark.unit
def test_accum_steps_valid_accumulation():
    """batch=8, micro=2 → 4 accumulation steps."""
    effective_bs, micro_bs, accum = _resolve_accum_steps(_cfg(8, 2))
    assert (effective_bs, micro_bs, accum) == (8, 2, 4)


@pytest.mark.unit
def test_accum_steps_zero_batch():
    """`batch_size=0` raises ValueError."""
    with pytest.raises(ValueError, match=r"batch_size"):
        _resolve_accum_steps(_cfg(0, 1))


@pytest.mark.unit
def test_accum_steps_zero_micro():
    """`micro_batch_size=0` raises ValueError."""
    with pytest.raises(ValueError, match=r"micro_batch_size"):
        _resolve_accum_steps(_cfg(8, 0))


@pytest.mark.unit
def test_accum_steps_micro_exceeds_batch():
    """`micro_batch_size > batch_size` raises ValueError."""
    with pytest.raises(ValueError, match=r"cannot exceed"):
        _resolve_accum_steps(_cfg(2, 4))


@pytest.mark.unit
def test_accum_steps_not_divisible():
    """`batch_size` not divisible by `micro_batch_size` raises ValueError."""
    with pytest.raises(ValueError, match=r"divisible"):
        _resolve_accum_steps(_cfg(7, 2))


# ---------------------------------------------------------------------------
# _prepare_run_dirs
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_prepare_fresh_creates_ckpt_dir(tmp_path: Path):
    """Fresh run with save_model=True creates and returns the checkpoint directory."""
    out_dir = tmp_path / "exp"
    out_dir.mkdir()
    ckpt_dir, _ = _prepare_run_dirs(out_dir, resume=False, save_model=True, make_plots=False)
    assert ckpt_dir.exists()
    assert ckpt_dir == out_dir / "checkpoints"


@pytest.mark.unit
def test_prepare_plots_creates_fig_dir(tmp_path: Path):
    """make_plots=True creates and returns the figures directory."""
    out_dir = tmp_path / "exp"
    out_dir.mkdir()
    _, fig_dir = _prepare_run_dirs(out_dir, resume=False, save_model=True, make_plots=True)
    assert fig_dir.exists()
    assert fig_dir == out_dir / "figures"


@pytest.mark.unit
def test_prepare_resume_missing_out_dir(tmp_path: Path):
    """`resume=True` but `out_dir` does not exist raises FileNotFoundError."""
    out_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match=r"resume=True"):
        _prepare_run_dirs(out_dir, resume=True, save_model=True, make_plots=False)


@pytest.mark.unit
def test_prepare_resume_missing_ckpt_dir(tmp_path: Path):
    """`resume=True`, `out_dir` exists but no checkpoints/ subdirectory raises FileNotFoundError."""
    out_dir = tmp_path / "exp"
    out_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=r"checkpoints"):
        _prepare_run_dirs(out_dir, resume=True, save_model=True, make_plots=False)


@pytest.mark.unit
def test_prepare_fresh_existing_checkpoints(tmp_path: Path):
    """`resume=False` but pre-existing checkpoint files raise FileExistsError."""
    out_dir = tmp_path / "exp"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "ckpt-1.index").touch()   # simulate an existing checkpoint
    with pytest.raises(FileExistsError, match=r"resume=False"):
        _prepare_run_dirs(out_dir, resume=False, save_model=True, make_plots=False)


@pytest.mark.unit
def test_prepare_no_save_no_dirs(tmp_path: Path):
    """`save_model=False, make_plots=False` creates no directories."""
    out_dir = tmp_path / "exp"
    out_dir.mkdir()
    ckpt_dir, fig_dir = _prepare_run_dirs(out_dir, resume=False, save_model=False, make_plots=False)
    assert not ckpt_dir.exists()
    assert not fig_dir.exists()
