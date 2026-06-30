"""
Unit tests for igm.assimilations.pretraining.history

Covers: History.append_epoch, save_history_yaml, load_history_yaml
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from igm.assimilations.pretraining.history import History, load_history_yaml, save_history_yaml


def _full_epoch_kwargs(**overrides) -> dict:
    """Return a complete set of append_epoch keyword arguments."""
    base = dict(
        train_total=1.0, val_total=2.0,
        train_data=0.5,  val_data=1.5,
        train_phys=0.3,  val_phys=0.7,
        lambda_phys=0.1,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# History dataclass
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_append_epoch_coerces_to_float():
    """append_epoch accepts int and numpy scalar inputs and stores them as float."""
    h = History()
    h.append_epoch(**_full_epoch_kwargs(
        train_total=np.float32(3.14),
        val_total=2,               # plain int
        lambda_phys=np.int64(0),
    ))
    assert len(h.train_total) == 1
    assert type(h.train_total[0]) is float
    assert type(h.val_total[0]) is float
    assert type(h.lambda_phys[0]) is float
    # All seven lists grow by one
    for lst in (h.train_total, h.val_total, h.train_data, h.val_data,
                h.train_phys, h.val_phys, h.lambda_phys):
        assert len(lst) == 1


# ---------------------------------------------------------------------------
# save_history_yaml / load_history_yaml
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_save_load_round_trip(tmp_path: Path):
    """Values written by save_history_yaml are recovered exactly by load_history_yaml."""
    h = History()
    for i in range(3):
        h.append_epoch(**_full_epoch_kwargs(
            train_total=float(i),
            val_total=float(i) + 0.5,
            lambda_phys=float(i) * 0.01,
        ))

    save_history_yaml(tmp_path, epoch=3, history=h)
    start_epoch, loaded = load_history_yaml(tmp_path)

    assert start_epoch == 3
    assert loaded.train_total == h.train_total
    assert loaded.val_total == h.val_total
    assert loaded.train_data == h.train_data
    assert loaded.val_data == h.val_data
    assert loaded.train_phys == h.train_phys
    assert loaded.val_phys == h.val_phys
    assert loaded.lambda_phys == h.lambda_phys


@pytest.mark.unit
def test_load_missing_file(tmp_path: Path):
    """load_history_yaml raises FileNotFoundError when history.yaml is absent."""
    with pytest.raises(FileNotFoundError):
        load_history_yaml(tmp_path)


@pytest.mark.unit
def test_load_empty_yaml(tmp_path: Path):
    """An empty YAML file yields epoch=0 and all empty History lists."""
    (tmp_path / "history.yaml").write_text("")
    start_epoch, h = load_history_yaml(tmp_path)
    assert start_epoch == 0
    assert h.train_total == []
    assert h.val_total == []
    assert h.lambda_phys == []


@pytest.mark.unit
def test_load_partial_yaml(tmp_path: Path):
    """A YAML with only some fields fills missing ones with empty lists."""
    (tmp_path / "history.yaml").write_text(
        "epoch: 2\nval_total: [1.0, 2.0]\n"
    )
    start_epoch, h = load_history_yaml(tmp_path)
    assert start_epoch == 2
    assert h.val_total == [1.0, 2.0]
    assert h.train_total == []
    assert h.train_data == []
    assert h.lambda_phys == []


@pytest.mark.unit
def test_epoch_counter(tmp_path: Path):
    """The epoch field is stored and recovered independently of the list lengths."""
    h = History()
    h.append_epoch(**_full_epoch_kwargs())
    save_history_yaml(tmp_path, epoch=5, history=h)
    start_epoch, _ = load_history_yaml(tmp_path)
    assert start_epoch == 5


@pytest.mark.unit
def test_atomic_write_no_tmp_left(tmp_path: Path):
    """save_history_yaml must not leave a .tmp file behind after success."""
    h = History()
    h.append_epoch(**_full_epoch_kwargs())
    save_history_yaml(tmp_path, epoch=1, history=h)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Unexpected .tmp files left behind: {tmp_files}"
    assert (tmp_path / "history.yaml").exists()
