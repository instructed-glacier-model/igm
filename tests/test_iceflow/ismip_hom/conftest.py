import pytest
from tests.test_iceflow.ismip_hom.utils.config import load_test_config


def pytest_collection_modifyitems(config, items):
    """Deselect tests for experiments not listed in the active test config."""
    active_exps = set(load_test_config().get("experiments", []))
    deselected, remaining = [], []
    for item in items:
        parts = item.nodeid.replace("\\", "/").split("/")
        exp_part = next((p for p in parts if p.startswith("exp_")), None)
        if exp_part is not None and "ismip_hom" in item.nodeid and exp_part not in active_exps:
            deselected.append(item)
        else:
            remaining.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining
