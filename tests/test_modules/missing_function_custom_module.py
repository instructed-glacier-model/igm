# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math


## add custumized smb function
def params(parser):
    parser.add_argument("--meanela", type=float, default=3000)


def initialize(params, state):
    params.meanela = np.quantile(state.usurf[state.thk > 10], 0.2)


def update(params, state):
    # perturabe the ELA with sinusional signal
    ELA = params.meanela - 750 * math.sin((state.t / 100) * math.pi)
    # compute smb linear with elevation with 2 acc & abl gradients
    state.smb = state.usurf - ELA
    state.smb *= tf.where(state.smb < 0, 0.005, 0.009)
    # cap smb by 2 m/y
    state.smb = tf.clip_by_value(state.smb, -100, 2)
    # make sure the smb is not positive outside of the mask to prevent overflow
    if hasattr(state, "icemask"):
        state.smb = tf.where((state.smb < 0) | (state.icemask > 0.5), state.smb, -10)


# ! Oops, what happens? :)
# def finalize(params,state):
#     pass
