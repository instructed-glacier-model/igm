# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

 ## add BuelerC smb function
def params(parser):  
    pass

def initialize(params,state):

    state.x = tf.constant(np.linspace(-3950,3950,80).astype("float32"))
    state.y = tf.constant(np.linspace(-3950,3950,80).astype("float32"))

    nx = state.x.shape[0]
    ny = state.y.shape[0]
    
    state.usurf = tf.Variable(tf.zeros((ny,nx)))
    state.thk   = tf.Variable(tf.zeros((ny,nx)))
    state.topg  = tf.Variable(tf.zeros((ny,nx)))

    state.X, state.Y = tf.meshgrid(state.x, state.y)

    state.dx = state.x[1] - state.x[0]

    state.dX = tf.ones_like(state.X) * state.dx 
 
def update(params,state):
    # compute time dependent mass balance according to Bueler C 2005
    A = 1.0e-16  # [Pa^-3 year^-1]
    n = 3.
    g = 9.81
    rho = 910.
    Gamma = 2.*A*(rho*g)**n / (n+2)

    # calculate a radius from the igm coordinate grids
    R = tf.sqrt(state.X**2. + state.Y**2.)

    ## calculate dome height for time step
    lambda_B = 5.
    H_0_B = 3600.
    R_0_B = 750000.
    t_0_B = 15208.
    alpha_B = (2.-(n+1.)*lambda_B)/(5.*n+3.)
    beta_B = (1.+(2.*n+1.)*lambda_B)/(5.*n+3.)

    H_B = tf.zeros(tf.shape(R))

    if state.t > 0.:
        H_B = H_0_B*(state.t/t_0_B)**(-alpha_B) * (1.-((state.t/t_0_B)**(-beta_B) * (R/R_0_B))**((n+1.)/n))**(n/(2.*n+1.))
        H_B = tf.where(tf.math.is_nan(H_B), 0., H_B)
        smbB = (5. / state.t) * H_B

    else:
        smbB = H_B


    state.smb  = smbB
    
def finalize(params,state):
    pass
