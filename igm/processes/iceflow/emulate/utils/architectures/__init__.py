from .cnns import CNN, CNNPeriodic, CNNPatch, CNNSkip
from .dahunet import DahuNet
from .mlps import MLP, FourierMLP
from .nos import FNO, FNO2, CNO2d
from .nicenet import NiceNet
from .utils import DTypeActivation

Architectures = {
    "CNN": CNN,
    "CNNPeriodic": CNNPeriodic,
    "CNNPatch": CNNPatch,
    "CNNSkip": CNNSkip,
    "MLP": MLP,
    "FourierMLP": FourierMLP,
    "FNO": FNO,
    "CNO2d": CNO2d,
    "FNO2": FNO2,
    "NiceNet": NiceNet,
    "DahuNet": DahuNet,
}
