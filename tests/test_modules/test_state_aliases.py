"""Tests for the State variable aliasing mechanism."""

import pytest
import tensorflow as tf

from igm.common import State
from igm.common.aliases import (
    load_builtin_aliases,
    load_aliases_from_yaml,
    load_aliases_from_core_cfg,
)


@pytest.fixture(autouse=True)
def isolated_aliases():
    """Save and restore State._aliases around each test."""
    saved = dict(State._aliases)
    State._aliases.clear()
    yield
    State._aliases.clear()
    State._aliases.update(saved)


# ---------------------------------------------------------------------------
# Core mechanism
# ---------------------------------------------------------------------------

def test_getattr_reads_via_alias():
    State.register_aliases({"bed_elevation": "topg"})
    s = State()
    s.topg = tf.constant(42.0)
    assert s.bed_elevation.numpy() == pytest.approx(42.0)


def test_setattr_writes_via_alias():
    State.register_aliases({"bed_elevation": "topg"})
    s = State()
    s.bed_elevation = tf.constant(99.0)
    assert s.topg.numpy() == pytest.approx(99.0)


def test_alias_and_canonical_are_the_same_object():
    State.register_aliases({"bed_elevation": "topg"})
    s = State()
    s.topg = tf.Variable(7.0)
    assert s.bed_elevation is s.topg


def test_canonical_name_has_zero_overhead_path():
    """Canonical names must never invoke __getattr__."""
    State.register_aliases({"bed_elevation": "topg"})
    s = State()
    s.topg = tf.constant(1.0)

    getattr_calls = []
    original = State.__getattr__

    def tracking(self, name):
        getattr_calls.append(name)
        return original(self, name)

    State.__getattr__ = tracking
    try:
        _ = s.topg
        assert getattr_calls == [], "canonical name triggered __getattr__"
        _ = s.bed_elevation
        assert getattr_calls == ["bed_elevation"]
    finally:
        State.__getattr__ = original


def test_missing_attribute_raises():
    s = State()
    with pytest.raises(AttributeError):
        _ = s.nonexistent_var


def test_hasattr_works_for_alias():
    State.register_aliases({"bed_elevation": "topg"})
    s = State()
    s.topg = tf.constant(1.0)
    assert hasattr(s, "topg")
    assert hasattr(s, "bed_elevation")
    assert not hasattr(s, "nonexistent")


def test_alias_for_unset_canonical_raises():
    State.register_aliases({"bed_elevation": "topg"})
    s = State()
    # topg not set yet — alias should also raise
    with pytest.raises(AttributeError):
        _ = s.bed_elevation


def test_multiple_aliases_same_canonical():
    State.register_aliases({"bed_elevation": "topg", "bedrock_elevation": "topg"})
    s = State()
    s.topg = tf.constant(5.0)
    assert s.bed_elevation.numpy() == pytest.approx(5.0)
    assert s.bedrock_elevation.numpy() == pytest.approx(5.0)


def test_register_aliases_merges():
    State.register_aliases({"bed_elevation": "topg"})
    State.register_aliases({"surface_elevation": "usurf"})
    assert "bed_elevation" in State._aliases
    assert "surface_elevation" in State._aliases


# ---------------------------------------------------------------------------
# Loader utilities
# ---------------------------------------------------------------------------

def test_builtin_aliases_contains_expected_entries():
    aliases = load_builtin_aliases("descriptive", "pism")
    assert aliases["bed_elevation"] == "topg"
    assert aliases["bedrock_elevation"] == "topg"
    assert aliases["surface_elevation"] == "usurf"
    assert aliases["ice_surface_elevation"] == "usurf"
    assert aliases["ice_thickness"] == "thk"


def test_no_conflicts_between_builtin_alias_sets():
    """Alias names shared across built-in YAML files must point to the same canonical."""
    sets = {name: load_builtin_aliases(name) for name in ("descriptive", "pism")}
    names = list(sets)
    conflicts = {
        alias: {a: sets[a][alias], b: sets[b][alias]}
        for i, a in enumerate(names)
        for b in names[i + 1:]
        for alias in set(sets[a]) & set(sets[b])
        if sets[a][alias] != sets[b][alias]
    }
    assert not conflicts, (
        "Alias conflicts found between built-in sets:\n"
        + "\n".join(f"  {alias!r}: {mapping}" for alias, mapping in conflicts.items())
    )


def test_load_builtin_aliases_descriptive():
    aliases = load_builtin_aliases("descriptive")
    assert "bed_elevation" in aliases
    assert aliases["bed_elevation"] == "topg"


def test_load_aliases_from_yaml(tmp_path):
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text("my_bed: topg\nmy_surf: usurf\n")
    aliases = load_aliases_from_yaml(yaml_file)
    assert aliases == {"my_bed": "topg", "my_surf": "usurf"}


def test_load_aliases_from_yaml_empty(tmp_path):
    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("")
    assert load_aliases_from_yaml(yaml_file) == {}


def test_custom_yaml_aliases_work_end_to_end(tmp_path):
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text("my_bed: topg\n")
    State.register_aliases(load_aliases_from_yaml(yaml_file))

    s = State()
    s.topg = tf.constant(3.0)
    assert s.my_bed.numpy() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# load_aliases_from_core_cfg
# ---------------------------------------------------------------------------

def test_load_aliases_from_core_cfg_both_builtins():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"builtin": ["descriptive", "pism"], "extra_files": []})
    aliases = load_aliases_from_core_cfg(cfg)
    assert "bed_elevation" in aliases   # descriptive set
    assert "temp" in aliases            # pism set


def test_load_aliases_from_core_cfg_empty():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"builtin": [], "extra_files": []})
    assert load_aliases_from_core_cfg(cfg) == {}


def test_load_aliases_from_core_cfg_single_builtin():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"builtin": ["descriptive"], "extra_files": []})
    aliases = load_aliases_from_core_cfg(cfg)
    assert "bed_elevation" in aliases
    assert "temp" not in aliases  # pism set not loaded


def test_load_aliases_from_core_cfg_custom_file(tmp_path):
    from omegaconf import OmegaConf
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text("my_alias: topg\n")
    cfg = OmegaConf.create({"builtin": [], "extra_files": [str(yaml_file)]})
    assert load_aliases_from_core_cfg(cfg) == {"my_alias": "topg"}
