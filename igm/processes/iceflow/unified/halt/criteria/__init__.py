from .criterion import Criterion
from .abs_tol import CriterionAbsTol
from .rel_tol import CriterionRelTol
from .patience import CriterionPatience
from .inf import CriterionInf
from .nan import CriterionNaN
from .threshold import CriterionThreshold
from .log_burst_patience import CriterionLogBurstPatience

Criteria = {
    "abs_tol": CriterionAbsTol,
    "rel_tol": CriterionRelTol,
    "patience": CriterionPatience,
    "inf": CriterionInf,
    "nan": CriterionNaN,
    "threshold": CriterionThreshold,
    "log_burst": CriterionLogBurstPatience,
}
