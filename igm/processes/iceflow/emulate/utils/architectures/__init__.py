from .cnns import CNN, CNNPeriodic, CNNPatch, CNNSkip
from .mlps import MLP, FourierMLP
from .nos import FNO, FNO2
from .utils import DTypeActivation
from .SR import SIADecompNet
from .SR_cno_style import CNO_DecompNet

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
    'CNO_DecompNet': CNO_DecompNet,
}
