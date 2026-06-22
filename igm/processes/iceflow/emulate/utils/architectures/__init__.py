from .cnns import CNN
from .mlps import MLP
from .nos import FNO
from .dahunet import DahuNet

Architectures = {
    "cnn":     CNN,
    "mlp":     MLP,
    "fno":     FNO,
    "dahunet": DahuNet,
}
