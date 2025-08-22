from .mapping import Mapping
from .mapping_nodal import MappingNodal
from .mapping_network import MappingNetwork

Mappings = {
    "nodal": MappingNodal,
    "network": MappingNetwork,
}
