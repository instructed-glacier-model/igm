from .cnns import CNN
from .mlps import MLP
from .nos import FNO
from .dahunet import DahuNet

Architectures = {
    "CNN":     CNN,
    "MLP":     MLP,
    "FNO":     FNO,
    "dahunet": DahuNet,
}
