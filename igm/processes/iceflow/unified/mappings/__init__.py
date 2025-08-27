from .mapping import Mapping
from .mapping_network import MappingNetwork
from .mapping_identity import MappingIdentity
from .interface_network import InterfaceNetwork
from .interface_identity import InterfaceIdentity

Mappings = {
    "network": MappingNetwork,
    "identity": MappingIdentity,
}

InterfaceMappings = {
    "network": InterfaceNetwork,
    "identity": InterfaceIdentity,
}
