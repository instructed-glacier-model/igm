import tensorflow as tf 

from igm.utils.math.getmag import getmag 


def get_velbase_1(U, vert_basis):
    if vert_basis.lower() in ["lagrange","sia"]:
        return U[...,0,:,:]
    elif vert_basis.lower() == "legendre":
        pm = tf.pow(-1.0, tf.range(U.shape[-3], dtype=tf.float32))
        return tf.tensordot(pm, U, axes=[[0], [-3]]) 

@tf.function(jit_compile=True)
def get_velbase(U, V, vert_basis):
    return get_velbase_1(U, vert_basis), get_velbase_1(V, vert_basis)

def get_velsurf_1(U, vert_basis):
    if vert_basis.lower() in ["lagrange","sia"]:
        return U[...,-1,:,:]
    elif vert_basis.lower() == "legendre":
        pm = tf.pow(1.0, tf.range(U.shape[-3], dtype=tf.float32))
        return tf.tensordot(pm, U, axes=[[0], [-3]])

@tf.function(jit_compile=True)
def get_velsurf(U, V, vert_basis):
    return get_velsurf_1(U, vert_basis), get_velsurf_1(V, vert_basis)

def get_velbar_1(U, vert_weight, vert_basis):
    if vert_basis.lower() == "lagrange":
        return tf.reduce_sum(U * vert_weight, axis=-3)
    elif vert_basis.lower() == "legendre":
        return U[...,0,:,:]
    elif vert_basis.lower() == "sia":
        return U[...,0,:,:]+0.8*(U[...,-1,:,:]-U[...,0,:,:])

@tf.function(jit_compile=True)
def get_velbar(U, V, vert_weight, vert_basis):
    return get_velbar_1(U, vert_weight, vert_basis), \
           get_velbar_1(V, vert_weight, vert_basis)

@tf.function(jit_compile=True)
def boundvel(velbar_mag, VEL, force_max_velbar):
    return tf.where(velbar_mag >= force_max_velbar, force_max_velbar * (VEL / velbar_mag), VEL)

@tf.function(jit_compile=True)
def clip_max_velbar(U, V, force_max_velbar, vert_basis, vert_weight):

    if vert_basis.lower() in ["lagrange","sia"]:
        velbar_mag = getmag(U, V)
        U_clipped = boundvel(velbar_mag, U, force_max_velbar)
        V_clipped = boundvel(velbar_mag, V, force_max_velbar)

    elif vert_basis.lower() == "legendre":
        velbar_mag = getmag(*get_velbar(U, V, \
                                        vert_weight, vert_basis))
        uvelbar = boundvel(velbar_mag, U[0], force_max_velbar)
        vvelbar = boundvel(velbar_mag, V[0], force_max_velbar)
        U_clipped = tf.concat([uvelbar[None,...] , U[1:]], axis=0)
        V_clipped = tf.concat([vvelbar[None,...] , V[1:]], axis=0)
        
    else:
        raise ValueError("Unknown vertical basis: " + vert_basis)
    
    return U_clipped, V_clipped