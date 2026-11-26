from .abs_change import CriterionAbsChange
from .abs_tol import CriterionAbsTol
from .criterion import Criterion
from .inf import CriterionInf
from .log_burst_patience import CriterionLogBurstPatience
from .nan import CriterionNaN
from .patience import CriterionPatience
from .rel_initial import CriterionRelInitial
from .rel_tol import CriterionRelTol
from .threshold import CriterionThreshold

Criteria = {
    "abs_tol": CriterionAbsTol,
    "abs_change": CriterionAbsChange,
    "rel_tol": CriterionRelTol,
    "rel_initial": CriterionRelInitial,
    "patience": CriterionPatience,
    "inf": CriterionInf,
    "nan": CriterionNaN,
    "threshold": CriterionThreshold,
    "log_burst": CriterionLogBurstPatience,
}
