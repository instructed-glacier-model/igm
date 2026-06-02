"""Unit tests for igm.common.runner.modules.src.check_module_needs."""

import types
from pathlib import Path

import pytest

from igm.common import State
from igm.common.runner.modules.src import check_module_needs


def _make_module(
    tmp_path: Path, name: str, yaml_content: str | None = None
) -> types.ModuleType:
    """Return a mock module whose __file__ sits in a fresh temp subdirectory.

    If yaml_content is given, a module.yaml is written there (opt-in taken).
    If omitted, no module.yaml is written (opt-in not taken).
    """
    mod_dir = tmp_path / name
    mod_dir.mkdir()
    if yaml_content is not None:
        (mod_dir / f"{name}.yaml").write_text(yaml_content)
    mod = types.ModuleType(f"igm.processes.{name}")
    mod.__name__ = f"igm.processes.{name}"
    mod.__file__ = str(mod_dir / "__init__.py")
    return mod


def _state_with(*var_names: str) -> State:
    state = State()
    for v in var_names:
        setattr(state, v, True)
    return state


@pytest.mark.fast
@pytest.mark.unit
def test_passes_when_all_needed_vars_present(tmp_path):
    mod = _make_module(tmp_path, "enthalpy", "needs: [thk, U, V]\n")
    state = _state_with("thk", "U", "V")
    check_module_needs([mod], state)  # no raise


@pytest.mark.fast
@pytest.mark.unit
def test_raises_when_a_needed_var_is_missing(tmp_path):
    mod = _make_module(tmp_path, "enthalpy", "needs: [thk, U, V, W]\n")
    state = _state_with("thk", "U", "V")  # W missing
    with pytest.raises(RuntimeError, match="enthalpy"):
        check_module_needs([mod], state)


@pytest.mark.fast
@pytest.mark.unit
def test_error_names_every_failing_module(tmp_path):
    enth = _make_module(tmp_path, "enthalpy", "needs: [thk, U]\n")
    thk = _make_module(tmp_path, "thk", "needs: [ubar, vbar]\n")
    state = _state_with("thk")  # U, ubar, vbar all missing
    with pytest.raises(RuntimeError) as exc_info:
        check_module_needs([enth, thk], state)
    msg = str(exc_info.value)
    assert "enthalpy" in msg
    assert "thk" in msg


@pytest.mark.fast
@pytest.mark.unit
def test_skips_module_without_yaml(tmp_path):
    mod = _make_module(tmp_path, "custom")  # no custom.yaml
    check_module_needs([mod], State())  # no raise


@pytest.mark.fast
@pytest.mark.unit
def test_skips_module_without_file_attribute():
    mod = types.ModuleType("igm.processes.builtin")
    mod.__name__ = "igm.processes.builtin"
    # deliberately no __file__
    check_module_needs([mod], State())  # no raise


@pytest.mark.fast
@pytest.mark.unit
def test_skips_bypassed_module(tmp_path):
    """A module whose declared outputs are all absent is assumed to have run
    in a reduced/bypassed mode (e.g. iceflow in pretraining) — its needs are
    not checked."""
    mod = _make_module(
        tmp_path, "iceflow",
        "needs: [thk, usurf]\nupdates: [U, V]\n"
    )
    state = _state_with()  # neither thk/usurf (needs) nor U/V (updates) present
    check_module_needs([mod], state)  # no raise


@pytest.mark.fast
@pytest.mark.unit
def test_does_not_skip_when_updates_are_present(tmp_path):
    """If at least one declared output is on state, the module ran normally —
    its needs ARE checked."""
    mod = _make_module(
        tmp_path, "iceflow",
        "needs: [thk, usurf]\nupdates: [U, V]\n"
    )
    state = _state_with("U")  # U present → module ran → check needs → missing thk, usurf
    with pytest.raises(RuntimeError, match="iceflow"):
        check_module_needs([mod], state)


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("missing", ["thk", "U", "V", "W", "arrhenius"])
def test_raises_for_each_missing_enthalpy_var(tmp_path, missing):
    needs = ["thk", "U", "V", "W", "arrhenius"]
    yaml_content = f"needs: {needs}\n"
    mod = _make_module(tmp_path, f"enthalpy_{missing}", yaml_content)
    present = [v for v in needs if v != missing]
    state = _state_with(*present)
    with pytest.raises(RuntimeError):
        check_module_needs([mod], state)
