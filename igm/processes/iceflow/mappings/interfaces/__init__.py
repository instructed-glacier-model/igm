from .interface import InterfaceMapping
from .network import InterfaceNetwork
from .identity import InterfaceIdentity
from .data_assimilation import InterfaceDataAssimilation

InterfaceMappings = {
    "identity": InterfaceIdentity,
    "network": InterfaceNetwork,
    "data_assimilation": InterfaceDataAssimilation,
}
