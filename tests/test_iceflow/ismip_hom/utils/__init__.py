from .experiments import (
    ExperimentA,
    ExperimentB,
    ExperimentC,
    ExperimentCInversion,
    ExperimentD,
    ExperimentE1,
    ExperimentE2,
)
from .simulator import run_igm
from .runner import run_experiment_test
from .validation import validate_results
from .config import get_unified_parameters, get_unified_parameters_no_length

Experiments = {
    "A": ExperimentA,
    "B": ExperimentB,
    "C": ExperimentC,
    "CInversion": ExperimentCInversion,
    "D": ExperimentD,
    "E1": ExperimentE1,
    "E2": ExperimentE2,
}
