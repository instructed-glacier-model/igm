import tensorflow as tf
from typing import Dict, Tuple, Optional
from abc import ABC, abstractmethod

from igm.processes.iceflow.energy.utils import stag4h
from igm.processes.iceflow.utils import get_velbase, Y_to_UV


class SlidingLaw(ABC):
    @abstractmethod
    def compute_shear_stress() -> Tuple[tf.Tensor, tf.Tensor]:
        pass

class WeertmanParams(tf.experimental.ExtensionType):
    exponent: float
    coefficient: float
    regu_weertman: float
    Nz: int
    staggered_grid: int
    vert_basis: str

class Weertman(SlidingLaw):
    name = "weertman"
    
    def __init__(self, params):
        self.params = params

    @tf.function(jit_compile=True)
    def compute_shear_stress(self, Y: tf.Tensor, effective_pressure: Dict) -> Tuple[tf.Tensor, tf.Tensor]:
        return _weertman(Y, effective_pressure, self.params)
    
def _weertman(Y: tf.Tensor, effective_pressure: Optional[Dict], parameters: WeertmanParams) -> Tuple[tf.Tensor, tf.Tensor]:

    """
    Computes sliding_shear_stress for the Weertman sliding law (for the loss computation in IGM).
    
    For example, for weertman sliding law, the basis vectors are U_basal and V_basal
    and the sliding law shear stress is the actual sliding law (tau_s = c * ||basal_velocity||^(s-2)*basal_velocity).
    
    You can the sliding law here (equation 17, where s = 1 + m): https://github.com/jouvetg/igm-paper/blob/main/paper.pdf
    as well as here (equation 2.16): https://www.cambridge.org/core/services/aop-cambridge-core/content/view/F2D0D3A274405887B512A474D0C64C1D/S0022112016005930a.pdf/mechanical-error-estimators-for-shallow-ice-flow-models.pdf

    Returns
    -------
    Tuple
        (basis_vectors, sliding_shear_stress)
    """

    # Manually doing sliding loss (for ground truth)
    c = parameters.coefficient
    regu_weertman = parameters.regu_weertman
    s = 1.0 + 1.0 / parameters.exponent
    
    U, V = Y_to_UV(parameters.Nz, Y)

    if parameters.staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
    
    U_basal, V_basal = get_velbase(U, V, parameters.vert_basis)
    velbase_mag = (U_basal**2 + V_basal**2 + regu_weertman**2) ** (1/2) # Assuming L2 Norm - check it against the M norm potentially...


    basis_vectors = [U_basal, V_basal]
    sliding_shear_stress = [
        c * velbase_mag ** (s - 2) * U_basal,
        c * velbase_mag ** (s - 2) * V_basal,
    ]      

    return basis_vectors, sliding_shear_stress