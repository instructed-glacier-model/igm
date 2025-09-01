"""
    This class overrides the oggm.core.sia2d.Model2D class to use the ice flow model
    implemented in the IGM package. The ice flow model is called at each time step
    to compute the ice velocity and the ice thickness evolution. 
    The ice flow model is called through the python wrapper of the IGM package.
    
    Code written by: Julien Jehl, Fabien Maussion, and Guillaume Jouvet
"""

import numpy as np
import tensorflow as tf
import os

from oggm import cfg, utils
from oggm.cfg import G, SEC_IN_YEAR, SEC_IN_DAY

import igm
from oggm.core.sia2d import Model2D

class IGM_Model2D(Model2D):
    def filter_ice_border(ice_thick):
        """Sets the ice thickness at the border of the domain to zero."""
        ice_thick[0, :] = 0
        ice_thick[-1, :] = 0
        ice_thick[:, 0] = 0
        ice_thick[:, -1] = 0
        return ice_thick

    def __init__(
        self,
        bed_topo,
        init_ice_thick=None,
        dx=None,
        dy=None,
        mb_model=None,
        y0=0.0,
        mb_elev_feedback="annual",
        ice_thick_filter=filter_ice_border,
        mb_filter=None,
        x=None,
        y=None,
    ):
        super(IGM_Model2D, self).__init__(
            bed_topo,
            init_ice_thick=init_ice_thick,
            dx=dx,
            dy=dy,
            mb_model=mb_model,
            y0=y0,
            mb_elev_feedback=mb_elev_feedback,
            ice_thick_filter=ice_thick_filter,
            mb_filter=mb_filter,
        )

        """

        Parameters
        ----------    
        bed_topo : the topography (2d array)
        
        init_ice_thick : the initial ice thickness (zero everywhere by default) (2d array)
        
        dx, dy : map resolution (float)
        
        mb_model : the mass balance model to use for the simulation (funtion)
        
        y0 : the starting year (int)
        
        mb_elev_feedback : when to update the mass balance model : ’annual’, ’monthly’, ’always’
                            (’annual’ by default) (str)

        ice_thick_filter : function to apply to the ice thickness *after* each time step. (function)
        
        mb_filter : the mask of the glacier (2d array)
        
        """

        self.state = igm.State() 

        self.cfg = igm.EmptyClass()  
        self.cfg.processes = igm.load_yaml_as_cfg(os.path.join("conf","processes","iceflow.yaml"))

        # Parameter
        self.cfl = 0.25
        self.max_dt = SEC_IN_YEAR
        self.dx = dx

        self.x = x
        self.y = y

        self.cfg.processes.iceflow.retrain_iceflow_emulator_freq = 0

        # intialize
        self.state.thk = tf.Variable(self.ice_thick)
        self.state.usurf = tf.Variable(self.surface_h)
        self.state.smb = tf.Variable(tf.zeros_like(self.ice_thick))

        # define
        self.state.arrhenius = (
            tf.ones_like(self.state.thk) * cfg.PARAMS["glen_a"] * SEC_IN_YEAR * 1e18
        )
        self.state.slidingco = tf.ones_like(self.state.thk) * 0.045
        self.state.dX = tf.ones_like(self.state.thk) * self.dx

        self.state.x = tf.constant(self.x)
        self.state.y = tf.constant(self.y)

        self.state.it = -1

        self.icemask = mb_filter

        igm.processes.iceflow.iceflow.initialize(self.cfg, self.state)

    def step(self, dt):
        # recast glacier variables into igm-like variables

        self.state.thk.assign(self.ice_thick)
        self.state.usurf.assign(self.surface_h)

        # compute ubar and vbar
        igm.processes.iceflow.iceflow.update(self.cfg, self.state)

        # retrurn the divergence of the flux using upwind fluxes
        divflux = (
            igm.processes.utils.compute_divflux(
                self.state.ubar,
                self.state.vbar,
                self.state.thk,
                self.state.dX,
                self.state.dX,
            )
            / SEC_IN_YEAR
        )

        # compute max speed for the CFL stability condition
        velomax = (
            max(
                tf.math.reduce_max(tf.math.abs(self.state.ubar)),
                tf.math.reduce_max(tf.math.abs(self.state.vbar)),
            ).numpy()
            / SEC_IN_YEAR
        )

        # compute the time step that comply with CLF condition
        if velomax > 0:
            dt_cfl = min(self.cfl * self.dx / velomax, self.max_dt)
        else:
            dt_cfl = self.max_dt

        self.state.it += 1

        # compute effective time step
        dt_use = utils.clip_scalar(np.min([dt_cfl, dt]), 0, self.max_dt)

        # compute the surface mass balance
        self.state.smb.assign(self.get_mb())

        # return the updated ice thickness with oggm-like variable
        self.state.thk.assign(
            tf.maximum(self.state.thk + dt_use * (self.state.smb - divflux), 0)
        )

        self.ice_thick = self.state.thk.numpy()

        # Next step
        self.t += dt_use

        return dt_use
