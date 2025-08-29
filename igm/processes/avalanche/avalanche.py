#!/usr/bin/env python3

# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

def initialize(cfg, state):
    
    if "time" not in cfg.processes:
        raise ValueError("The 'time' module is required for the 'avalanche' module.")
    
    state.tlast_avalanche = tf.Variable(cfg.processes.time.start, dtype=tf.float32)

def update(cfg, state):

    if (state.t - state.tlast_avalanche) >= cfg.processes.avalanche.update_freq:
        
        if hasattr(state, "logger"):
            state.logger.info("Update AVALANCHE at time : " + str(state.t.numpy()))
 
        H = state.thk
        Zb = state.topg
        Zi = Zb + H
        
        # the elevation difference of the cells that is considered to be stable
        dHRepose = state.dx * tf.math.tan( 
            cfg.processes.avalanche.angleOfRepose * np.pi / 180.0
        )
        Ho = tf.maximum(H, 0)

        count = 0
        # volume redistributed # for documentation if needed
        # volumes = []
         
        while count <=300:
            count += 1
            
            # find out, in which direction is down (instead of doing normal gradients, we do the max of the two directions)

            dZidx_down = tf.pad(
                tf.maximum(Zi[:, 1:] - Zi[:, :-1], 0), [[0, 0], [1, 0]], "CONSTANT"
            )
            dZidx_up = tf.pad(
                tf.maximum(Zi[:, :-1] - Zi[:, 1:], 0), [[0, 0], [0, 1]], "CONSTANT"
            )
            dZidx = tf.maximum(dZidx_down, dZidx_up)

            dZidy_left = tf.pad(
                tf.maximum(Zi[1:, :] - Zi[:-1, :], 0), [[1, 0], [0, 0]], "CONSTANT"
            )
            dZidy_right = tf.pad(
                tf.maximum(Zi[:-1, :] - Zi[1:, :], 0), [[0, 1], [0, 0]], "CONSTANT"
            )
            dZidy = tf.maximum(dZidy_right, dZidy_left)

            grad = tf.math.sqrt(dZidx**2 + dZidy**2)
            gradT = dZidy_left + dZidy_right + dZidx_down + dZidx_up
            gradT = tf.where(gradT == 0, 1, gradT) # avoid devide by zero error. However, could influence the results (not checked) 
            grad = tf.where(Ho < 0.1, 0, grad)

#            Was before Andreas's update
#            mxGrad = tf.reduce_max(grad)
#            if mxGrad <= 1.1 * dHRepose:
#                break

            delH = tf.maximum(0, (grad - dHRepose) / 3.0)

            # ============ ANDREAS ADDED ===========
            # if there is less than a certain thickness to redesitribute, just redistribute the remaining thickness and stop afterwards
            # print(count, np.max(delH), np.sum(delH) / (np.shape(H)[0]*np.shape(H)[1]))
            mean_thickness = np.sum(delH) / (np.shape(H)[0]*np.shape(H)[1])

            if mean_thickness < cfg.processes.avalanche.stop_redistribution_thk:
                # for a last time, use all the thickness to redistribute and then stop
                delH = tf.maximum(0, grad - dHRepose)
                count = 2000 # set to random high number to exit the loop
                
            # volumes.append(np.sum(delH) / (np.shape(H)[0]*np.shape(H)[1]))                
            # ================================

            Htmp = Ho
            Ho = tf.maximum(0, Htmp - delH)
            delH = Htmp - Ho

            # The thickness that is redistributed to the neighboring cells based on the fraction of if it should be up, down, left or right (dZidx_**/gradT)
            delHup = tf.pad(
                delH[:, :-1] * dZidx_up[:, :-1] / gradT[:, :-1],
                [[0, 0], [1, 0]],
                "CONSTANT",
            )
            delHdn = tf.pad(
                delH[:, 1:] * dZidx_down[:, 1:] / gradT[:, 1:],
                [[0, 0], [0, 1]],
                "CONSTANT",
            )
            delHrt = tf.pad(
                delH[:-1, :] * dZidy_right[:-1, :] / gradT[:-1, :],
                [[1, 0], [0, 0]],
                "CONSTANT",
            )
            delHlt = tf.pad(
                delH[1:, :] * dZidy_left[1:, :] / gradT[1:, :],
                [[0, 1], [0, 0]],
                "CONSTANT",
            )
            
            # calculate new thickness after distribution ensuring that the thickness is always positive
            Ho = tf.maximum(0, Ho + delHdn + delHup + delHlt + delHrt)

            Zi = Zb + Ho

        # print(count)
        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow( Ho + tf.where(H<0,H,0) - state.thk ,origin='lower'); plt.colorbar()

        state.thk = Ho + tf.where(H < 0, H, 0)

        state.usurf = state.topg + state.thk

        state.tlast_avalanche.assign(state.t)


def finalize(cfg, state):
    pass
