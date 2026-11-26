from .data_assimilation import MappingDataAssimilation
from .identity import MappingIdentity
from .mapping import Mapping
from .network import MappingNetwork

Mappings = {
    "identity": MappingIdentity,
    "network": MappingNetwork,
    "data_assimilation": MappingDataAssimilation,
}

from .interfaces import InterfaceMapping, InterfaceMappings
