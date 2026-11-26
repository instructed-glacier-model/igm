from .cnns import CNN
from .mlps import MLP
from .nos import FNO
from .cno import CNO
from .dahunet import DahuNet
from .hybrid_dahunet_fno import HybridDahuNetFNO
from .hybrid_dahunet_cno import HybridDahuNetCNO

Architectures = {
    "cnn":     CNN,
    "mlp":     MLP,
    "fno":     FNO,
    "cno":     CNO,
    "dahunet": DahuNet,
    "hybrid_dahunet_fno": HybridDahuNetFNO,
    "hybrid_dahunet_cno": HybridDahuNetCNO,
}
