from .cnns import CNN
from .mlps import MLP, FourierMLP
from .nos import FNO, FNO2, CNO2d
from .nicenet import NiceNet
from .utils import DTypeActivation
from .SR import SIADecompNet
from .SR_v2 import SIADecompNetV2, SIADecompNetV2SharedHead
from .SR_cno_style import CNO_DecompNet
from .dahunet import DahuNet

Architectures = {
    'CNN': CNN,
    'MLP': MLP,
    'FourierMLP': FourierMLP,
    'FNO': FNO,
    'CNO2d': CNO2d,
    'FNO2': FNO2,
    'SIADecompNet': SIADecompNet,
    'SIADecompNetV2': SIADecompNetV2,
    "SIADecompNetV2SharedHead": SIADecompNetV2SharedHead,
    'CNO_DecompNet': CNO_DecompNet,
    'dahunet': DahuNet,
}
