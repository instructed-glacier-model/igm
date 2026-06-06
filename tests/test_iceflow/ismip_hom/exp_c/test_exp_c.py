#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import pytest
from tests.test_iceflow.ismip_hom.utils import (
    run_experiment_test,
    get_unified_parameters,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("length,mapping,optimizer", get_unified_parameters("exp_c"))
def test_exp_c_unified(
    monkeypatch: pytest.MonkeyPatch, length: int, mapping: str, optimizer: str
) -> None:
    """Test ISMIP-HOM Experiment C with unified method."""
    run_experiment_test(
        monkeypatch,
        experiment="exp_c",
        length=length,
        method="unified",
        mapping=mapping,
        optimizer=optimizer,
    )
