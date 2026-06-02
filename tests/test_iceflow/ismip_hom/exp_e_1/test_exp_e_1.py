#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import pytest
from tests.test_iceflow.ismip_hom.utils import (
    run_experiment_test,
    get_unified_parameters_no_length,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "mapping,optimizer", get_unified_parameters_no_length("exp_e_1")
)
def test_exp_e_1_unified(
    monkeypatch: pytest.MonkeyPatch, mapping: str, optimizer: str
) -> None:
    """Test ISMIP-HOM Experiment E-1 with unified method."""
    run_experiment_test(
        monkeypatch,
        experiment="exp_e_1",
        length=None,
        method="unified",
        mapping=mapping,
        optimizer=optimizer,
    )
