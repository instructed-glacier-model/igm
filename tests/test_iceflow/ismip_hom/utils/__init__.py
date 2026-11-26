from .config import get_parameters, get_parameters_no_length
from .experiments import (
    ExperimentA,
    ExperimentB,
    ExperimentC,
    ExperimentCInversion,
    ExperimentD,
    ExperimentE1,
    ExperimentE2,
)
from .runner import run_experiment_test
from .simulator import run_igm
from .validation import validate_results

Experiments = {
    "A": ExperimentA,
    "B": ExperimentB,
    "C": ExperimentC,
    "CInversion": ExperimentCInversion,
    "D": ExperimentD,
    "E1": ExperimentE1,
    "E2": ExperimentE2,
}
