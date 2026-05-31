"""Unit tests for igm.common.legacy.check_legacy_keys."""

import pytest
from omegaconf import OmegaConf

from igm.common.legacy import check_legacy_keys


@pytest.mark.fast
@pytest.mark.unit
def test_passes_on_clean_config():
    cfg = OmegaConf.create({
        "processes": {"iceflow": {"physics": {
            "sliding": {"slidingco": 0.0464, "regularization": 1.0e-10},
            "viscosity": {"arrhenius": 78.0, "exponent": 3.0},
        }}}
    })
    check_legacy_keys(cfg)  # no raise


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("legacy_path,value", [
    ("processes.iceflow.physics.init_slidingco", 0.0464),
    ("processes.iceflow.physics.init_tau_ref", 0.0464),
    ("processes.iceflow.physics.init_arrhenius", 78.0),
    ("processes.iceflow.physics.enhancement_factor", 1.0),
    ("processes.iceflow.physics.exp_glen", 3.0),
    ("processes.iceflow.physics.regu_glen", 1.0e-5),
])
def test_raises_on_legacy_flat_key(legacy_path, value):
    cfg = OmegaConf.create({})
    OmegaConf.update(cfg, legacy_path, value, force_add=True)
    with pytest.raises(ValueError, match="parameter names that have been renamed"):
        check_legacy_keys(cfg)


@pytest.mark.fast
@pytest.mark.unit
@pytest.mark.parametrize("law", ["weertman", "coulomb", "budd", "mohr_coulomb"])
def test_raises_on_legacy_sliding_sub_block(law):
    cfg = OmegaConf.create({})
    OmegaConf.update(
        cfg, f"processes.iceflow.physics.sliding.{law}.exponent", 3.0,
        force_add=True,
    )
    with pytest.raises(ValueError, match="parameter names that have been renamed"):
        check_legacy_keys(cfg)


@pytest.mark.fast
@pytest.mark.unit
def test_message_lists_all_offenders():
    cfg = OmegaConf.create({})
    OmegaConf.update(cfg, "processes.iceflow.physics.exp_glen", 3.0, force_add=True)
    OmegaConf.update(cfg, "processes.iceflow.physics.init_slidingco", 0.0464, force_add=True)
    with pytest.raises(ValueError) as excinfo:
        check_legacy_keys(cfg)
    msg = str(excinfo.value)
    assert "init_slidingco" in msg
    assert "exp_glen" in msg
    assert "PARAM_CHANGE.md" in msg
