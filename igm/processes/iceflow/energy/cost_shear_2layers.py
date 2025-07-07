#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
from igm.processes.iceflow.energy.utils import stag4
from igm.processes.iceflow.energy.utils import gauss_points_and_weights

@tf.function()
def compute_strainrate_Glen_twolayers_tf(U, V, dX):

    dUdx = (U[:, :, :, 1:] - U[:, :, :, :-1]) / dX[0, 0, 0]
    dVdx = (V[:, :, :, 1:] - V[:, :, :, :-1]) / dX[0, 0, 0]
    dUdy = (U[:, :, 1:, :] - U[:, :, :-1, :]) / dX[0, 0, 0]
    dVdy = (V[:, :, 1:, :] - V[:, :, :-1, :]) / dX[0, 0, 0]

    dUdx = (dUdx[:, :, :-1, :] + dUdx[:, :, 1:, :]) / 2
    dVdx = (dVdx[:, :, :-1, :] + dVdx[:, :, 1:, :]) / 2
    dUdy = (dUdy[:, :, :, :-1] + dUdy[:, :, :, 1:]) / 2
    dVdy = (dVdy[:, :, :, :-1] + dVdy[:, :, :, 1:]) / 2

    return dUdx, dVdx, dUdy, dVdy

# In the case of a 2 layers model, we assume a velcity profile is a SIA-like profile
@tf.function()
def cost_shear_2layers(thk, arrhenius, U, V, dX, exp_glen, regu_glen):

    n, w = gauss_points_and_weights(ord_gauss=3)

    dUdx, dVdx, dUdy, dVdy = compute_strainrate_Glen_twolayers_tf(U, V, dX)

    Um = stag4(U)
    Vm = stag4(V)

    # B has Unit Mpa y^(1/n)
    B = 2.0 * arrhenius ** (-1.0 / exp_glen)
    p = 1.0 + 1.0 / exp_glen

    def f(zeta):
        return ( 1 - (1 - zeta) ** (exp_glen + 1) )
    
    def fp(zeta):
        return (exp_glen + 1) * (1 - zeta) ** exp_glen

    def p_term(zeta):
  
        UDX = (dUdx[:, 0, :, :] + (dUdx[:, -1, :, :]-dUdx[:, 0, :, :]) * f(zeta))
        VDY = (dVdy[:, 0, :, :] + (dVdy[:, -1, :, :]-dVdy[:, 0, :, :]) * f(zeta))
        UDY = (dUdy[:, 0, :, :] + (dUdy[:, -1, :, :]-dUdy[:, 0, :, :]) * f(zeta))
        VDX = (dVdx[:, 0, :, :] + (dVdx[:, -1, :, :]-dVdx[:, 0, :, :]) * f(zeta))
        UDZ = (Um[:, -1, :, :]-Um[:, 0, :, :]) * fp(zeta) / tf.maximum( stag4(thk) , 1)
        VDZ = (Vm[:, -1, :, :]-Vm[:, 0, :, :]) * fp(zeta) / tf.maximum( stag4(thk) , 1)
        
        Exx = UDX
        Eyy = VDY
        Ezz = - UDX - VDY
        Exy = 0.5 * VDX + 0.5 * UDY
        Exz = 0.5 * UDZ
        Eyz = 0.5 * VDZ
    
        sr2 = 0.5 * ( Exx**2 + Exy**2 + Exy**2 + Eyy**2 + Ezz**2 + Exz**2 + Eyz**2 + Exz**2 + Eyz**2 )

        return (sr2 + regu_glen**2) ** (p / 2) / p
  
    # C_shear is unit  Mpa y^(1/n) y^(-1-1/n) * m = Mpa m/y
    C_shear = stag4(B) * stag4(thk) * sum(w[i] * p_term(n[i]) for i in range(len(n)))

    return C_shear