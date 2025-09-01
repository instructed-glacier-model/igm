import igm
import tensorflow as tf
import pytest
import os
  
import numpy as np
import matplotlib.pyplot as plt

from igm.processes.enthalpy import *
from igm.processes.enthalpy.enthalpy import vertically_discretize_tf,TpmpEpmp_from_depth_tf
from igm.processes.enthalpy.enthalpy import surf_enthalpy_from_temperature_tf, compute_enthalpy_basalmeltrate, temperature_from_enthalpy_tf

# this file is avilable at https://github.com/WangYuzhe/PoLIM-Polythermal-Land-Ice-Model
# verif = scipy.io.loadmat("sol_analytic/enthA_analy_result.mat")

def test_enthalpy():

    ttf = 150000.0  # 300000
    dt = 200.0

    tim = np.arange(0, ttf, dt) + dt  # to put back to 300000
 
    cfg = igm.load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
 
    cfg.processes.iceflow.numerics.Nz = 50
    cfg.processes.iceflow.numerics.vert_spacing = 1

    cfg.processes.enthalpy.KtdivKc = 10 ** (-5)  # check this value if ok ?
    cfg.processes.enthalpy.till_wat_max = 200
    cfg.processes.enthalpy.drain_rate = 0

    thk = tf.Variable(1000 * tf.ones((1, 1)))

    depth, dz = vertically_discretize_tf(thk, cfg.processes.iceflow.numerics.Nz, cfg.processes.iceflow.numerics.vert_spacing)

    strainheat = tf.Variable(tf.zeros((cfg.processes.iceflow.numerics.Nz, 1, 1)))
    frictheat = tf.Variable(0.0 * tf.ones((1, 1)))
    geoheatflux = tf.Variable(0.042 * tf.ones((1, 1)))  # in W m-2
    tillwat = tf.Variable(0.0 * tf.ones((1, 1)))

    # Initial enthalpy field
    T = tf.Variable((-30.0 + 273.15) * tf.ones((cfg.processes.iceflow.numerics.Nz, 1, 1)))
    E = tf.Variable(cfg.processes.enthalpy.ci * (T - 223.15))
    omega = tf.Variable(tf.zeros_like(T))
    w = tf.Variable(tf.zeros_like(T))

    TB = []
    HW = []
    MR = []

    surftemp = tf.Variable((-30 + 273.15) * tf.ones((1, 1)))

    for it, t in enumerate(tim):
        if (t >= 100000.0) & (t < 150000.0):
            surftemp.assign((-5 + 273.15) * tf.ones((1, 1)))
        else:
            surftemp.assign((-30 + 273.15) * tf.ones((1, 1)))

        Tpmp, Epmp = TpmpEpmp_from_depth_tf(
            depth,
            cfg.processes.iceflow.physics.gravity_cst,
            cfg.processes.iceflow.physics.ice_density,
            cfg.processes.enthalpy.claus_clape,
            cfg.processes.enthalpy.melt_temp,
            cfg.processes.enthalpy.ci,
            cfg.processes.enthalpy.ref_temp,
        )

        surfenth = surf_enthalpy_from_temperature_tf(
            surftemp, cfg.processes.enthalpy.melt_temp, cfg.processes.enthalpy.ci, cfg.processes.enthalpy.ref_temp
        )

        E, basalMeltRate = compute_enthalpy_basalmeltrate(
            E,
            Epmp,
            dt * cfg.processes.enthalpy.spy,
            dz,
            w,
            surfenth,
            geoheatflux,
            strainheat,
            frictheat,
            tillwat,
            cfg.processes.iceflow.physics.thr_ice_thk,
            cfg.processes.enthalpy.ki,
            cfg.processes.iceflow.physics.ice_density,
            cfg.processes.enthalpy.water_density,
            cfg.processes.enthalpy.ci,
            cfg.processes.enthalpy.ref_temp,
            cfg.processes.enthalpy.Lh,
            cfg.processes.enthalpy.spy,
            cfg.processes.enthalpy.KtdivKc,
            cfg.processes.enthalpy.drain_ice_column,
        )

        T, omega = temperature_from_enthalpy_tf(
            E, Tpmp, Epmp, cfg.processes.enthalpy.ci, cfg.processes.enthalpy.ref_temp, cfg.processes.enthalpy.Lh
        )

        tillwat = tillwat + dt * (basalMeltRate - cfg.processes.enthalpy.drain_rate)
        tillwat = tf.clip_by_value(tillwat, 0, cfg.processes.enthalpy.till_wat_max)

        TB.append(T[0] - 273.15)
        HW.append(tillwat)
        MR.append(basalMeltRate)

        if it % 100 == 0:
            print(
                "time :",
                t,
                T[0, 0, 0].numpy(),
                tillwat[0, 0].numpy(),
                basalMeltRate[0, 0].numpy(),
            )
        
    last_temp = np.stack(TB)[:, 0, 0][-1]

    print("Last temperature: ", last_temp)

    # this test should pass to validate enthaly module
    assert last_temp > -2
            
    # activate this block to plot the results
    if False:

        fig = plt.figure(figsize=(8, 8))

        plt.subplot(311)
        plt.plot(tim, np.stack(TB)[:, 0, 0])
        plt.ylabel("Temperature")

        plt.subplot(312)
        plt.plot(tim, np.stack(MR)[:, 0, 0])
        # plt.plot(verif["basalMelt"][1], verif["basalMelt"][0] / 1000, "-r")
        plt.ylabel("Melt rate")

        plt.subplot(313)
        plt.plot(tim, np.stack(HW)[:, 0, 0])
        plt.ylabel("Height Water")

        plt.savefig("KleinerExpA.png", pad_inches=0)

        plt.close()
