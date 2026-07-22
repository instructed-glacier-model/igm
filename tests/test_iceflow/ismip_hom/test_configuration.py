from tests.test_iceflow.ismip_hom.utils.config import (
    get_unified_mapping_optimizers,
    load_test_config,
)


def test_cg_newton_is_configured_for_identity_mapping_only():
    pairs = set(get_unified_mapping_optimizers(load_test_config()))

    assert ("identity", "cg_newton") in pairs
    assert ("network", "cg_newton") not in pairs
