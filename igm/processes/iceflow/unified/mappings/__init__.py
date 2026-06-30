from .mapping import Mapping
from .network import MappingNetwork
from .identity import MappingIdentity
from .data_assimilation import MappingDataAssimilation

Mappings = {
    "identity": MappingIdentity,
    "network": MappingNetwork,
    "data_assimilation": MappingDataAssimilation,
}

from .interfaces import InterfaceMapping, InterfaceMappings
