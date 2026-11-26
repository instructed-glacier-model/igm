from .cost import MetricCost
from .grad_theta_norm import MetricGradThetaNorm
from .grad_u_norm import MetricGradUNorm
from .metric import Metric
from .theta import MetricTheta
from .u import MetricU

Metrics = {
    "cost": MetricCost,
    "grad_u_norm": MetricGradUNorm,
    "grad_theta_norm": MetricGradThetaNorm,
    "u": MetricU,
    "theta": MetricTheta,
}
