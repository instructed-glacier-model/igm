from .mapping import Mapping
from .mapping_network import MappingNetwork
from .mapping_nodal import MappingNodal
from .interface_network import InterfaceNetwork
from .interface_nodal import InterfaceNodal

Mappings = {
    "network": MappingNetwork,
    "nodal": MappingNodal,
}

InterfaceMappings = {
    "network": InterfaceNetwork,
    "nodal": InterfaceNodal,
}
