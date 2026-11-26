#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import pytest

from tests.test_iceflow.ismip_hom.utils import (
    get_parameters_no_length,
    run_experiment_test,
)

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("mapping,optimizer", get_parameters_no_length("exp_e_1"))
def test_exp_e_1(monkeypatch: pytest.MonkeyPatch, mapping: str, optimizer: str) -> None:
    """Test ISMIP-HOM Experiment E-1."""
    run_experiment_test(
        monkeypatch,
        experiment="exp_e_1",
        length=None,
        mapping=mapping,
        optimizer=optimizer,
    )
