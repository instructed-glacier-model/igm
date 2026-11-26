import pytest
from omegaconf import OmegaConf

from igm.assimilations.data_assimilation.utils import resolve_friction_name


def _cfg(control_list, infer_params=False):
    return OmegaConf.create(
        {
            "assimilations": {
                "data_assimilation": {
                    "control_list": control_list,
                    "cook": {"infer_params": infer_params},
                }
            }
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize("control_list", [["thk"], ["thk", "tau_ref"]])
def test_friction_control_is_tau_ref(control_list):
    assert resolve_friction_name(_cfg(control_list)) == "tau_ref"


@pytest.mark.unit
@pytest.mark.parametrize("control_list", [["slidingco"], ["thk", "slidingco"]])
def test_slidingco_control_is_refused(control_list):
    with pytest.raises(ValueError, match="tau_ref"):
        resolve_friction_name(_cfg(control_list))


@pytest.mark.unit
def test_cook_infer_params_is_refused():
    with pytest.raises(ValueError, match="cook.infer_params"):
        resolve_friction_name(_cfg(["thk"], infer_params=True))
