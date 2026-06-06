import sys
import pytest
from igm.igm_run import main


@pytest.mark.skip(reason="API deprecated - needs updating for current IGM version")
def test_run_igm_with_params(monkeypatch):
    # Simulate: python igm_run.py +experiment=params
    monkeypatch.setattr(sys, "argv", ["igm_run.py", "+experiment=params"])
    main()  # This will now behave as if run from CLI
