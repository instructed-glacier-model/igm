from .cnns import CNN
from .mlps import MLP
from .nos import FNO
from .dahunet import DahuNet
from .chimera import Chimera

Architectures = {
    "CNN":     CNN,
    "MLP":     MLP,
    "FNO":     FNO,
    "dahunet": DahuNet,
    "chimera": Chimera,
}
