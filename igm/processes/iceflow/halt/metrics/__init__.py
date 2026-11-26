from .metric import Metric
from .cost import MetricCost
from .grad_u_norm import MetricGradUNorm
from .grad_theta_norm import MetricGradThetaNorm
from .u import MetricU
from .theta import MetricTheta

Metrics = {
    "cost": MetricCost,
    "grad_u_norm": MetricGradUNorm,
    "grad_theta_norm": MetricGradThetaNorm,
    "u": MetricU,
    "theta": MetricTheta,
}
