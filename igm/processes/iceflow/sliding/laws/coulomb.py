import tensorflow as tf
from typing import Tuple, Dict
from abc import ABC, abstractmethod

from igm.processes.iceflow.energy.utils import stag4h
from igm.processes.iceflow.utils import get_velbase, Y_to_UV

class SlidingLaw(ABC):
    @abstractmethod
    def compute_shear_stress() -> Tuple[tf.Tensor, tf.Tensor]:
        pass

class CoulombParams(tf.experimental.ExtensionType):
    coefficient: float
    exponent: float
    gamma_0: float
    epsilon: float
    Nz: int
    staggered_grid: int
    vert_basis: str
    min_effective_pressure: float

class Coulomb(SlidingLaw):
    name = "coulomb"
    
    def __init__(self, params):
        self.params = params

    @tf.function(jit_compile=True)
    def compute_shear_stress(self, Y: tf.Tensor, effective_pressure: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return _coulomb(Y, effective_pressure, self.params)

def _coulomb(Y: tf.Tensor, effective_pressure: tf.Tensor, parameters: CoulombParams) -> Tuple:

    """
    Returns a tuple of basis_vectors and sliding_shear_stress for the loss computation in IGM.
    
    https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2019GL082526

    Returns
    -------
    Tuple
        (basis_vectors, sliding_shear_stress)
    """
    
    c = parameters.coefficient
    n = parameters.exponent
    N = effective_pressure * 1e6 # convert to Pascals to be consistent with the coefficents (https://esurf.copernicus.org/articles/4/159/2016/)

    N = tf.where(N < 0.0, 0.0, N) # why is N negative?
    N = tf.where(N < parameters.min_effective_pressure, parameters.min_effective_pressure, N)
    
    gamma_0 = parameters.gamma_0
    # equivalent to A_s * C^n in regularized couluomb law (equation 3)
    # https://www.science.org/doi/10.1126/sciadv.abe7798
    
    U, V = Y_to_UV(parameters.Nz, Y)
    
    if parameters.staggered_grid:
        U = stag4h(U)
        V = stag4h(V)
        N = stag4h(N)


    U_basal, V_basal = get_velbase(U, V, parameters.vert_basis)
    
    velbase_mag = (U_basal**2 + V_basal**2) ** (1/2) # assuming l2 norm...
    basis_vectors = [U_basal, V_basal]
    
    eps = parameters.epsilon
    numerator = velbase_mag
    denominator = velbase_mag + gamma_0 * (N** n)
    
    sliding_shear_stress_u = N * c * tf.maximum(numerator / denominator, 0) ** (1/n) * (U_basal/(velbase_mag + eps)) # Pa
    sliding_shear_stress_v = N * c * tf.maximum(numerator / denominator, 0) ** (1/n) * (V_basal/(velbase_mag + eps)) # Pa

    
    sliding_shear_stress = [
        sliding_shear_stress_u * 1e-6, # convert to MPa
        sliding_shear_stress_v * 1e-6, # convert to MPa
    ]

    return basis_vectors, sliding_shear_stress