import igm, os
import tensorflow as tf
import pytest
 
def test_iceflow():
    
    state = igm.State()

    cfg = igm.load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))

    Ny,Nx = 40,30

    state.thk   = tf.Variable(tf.zeros((Ny,Nx)))
    state.usurf = tf.Variable(tf.zeros((Ny,Nx)))
    state.dX    = tf.Variable(tf.ones((Ny,Nx))*100)
    state.it    = -1
    
    igm.processes.iceflow.initialize(cfg, state)

    igm.processes.iceflow.update(cfg, state)

    igm.processes.iceflow.finalize(cfg, state)

    assert (tf.reduce_mean(state.ubar).numpy()<10*10)
