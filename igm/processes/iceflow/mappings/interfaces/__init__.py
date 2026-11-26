from .data_assimilation import InterfaceDataAssimilation
from .identity import InterfaceIdentity
from .interface import InterfaceMapping
from .network import InterfaceNetwork

InterfaceMappings = {
    "identity": InterfaceIdentity,
    "network": InterfaceNetwork,
    "data_assimilation": InterfaceDataAssimilation,
}
