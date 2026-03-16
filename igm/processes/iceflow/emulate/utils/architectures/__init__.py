from .cnns import CNN, CNNPeriodic, CNNPatch, CNNSkip
from .mlps import MLP, FourierMLP
from .nos import FNO, FNO2
from .utils import DTypeActivation
from .SR import SIADecompNet

Architectures = {
    'CNN': CNN,
    'CNNPeriodic': CNNPeriodic,
    'CNNPatch': CNNPatch,
    'CNNSkip': CNNSkip,
    'MLP': MLP,
    'FourierMLP': FourierMLP,
    'FNO': FNO,
    'FNO2': FNO2,
    'SIADecompNet': SIADecompNet,
}
