#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf  

from .weertman import SlidingLaw
from igm.processes.iceflow.utils import X_to_fieldin, Y_to_UV 
    
@tf.function(jit_compile=True)
def sliding_law_XY(X, Y, Nz, fieldin_list, dim_arrhenius, sliding_law: SlidingLaw):

    U, V = Y_to_UV(Nz, Y)

    fieldin = X_to_fieldin(X, fieldin_list, dim_arrhenius, Nz)
    
    return sliding_law(U, V, fieldin)
