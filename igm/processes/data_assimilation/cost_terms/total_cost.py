import tensorflow as tf

from .misfit_thk import misfit_thk
from .misfit_usurf import misfit_usurf
from .misfit_velsurf import misfit_velsurf
from .cost_divfluxfcz import cost_divfluxfcz
from .cost_divfluxobs import cost_divfluxobs
from .cost_vol import cost_vol
from .regu_thk import regu_thk
from .regu_slidingco import regu_slidingco
from .regu_arrhenius import regu_arrhenius

def total_cost(cfg, state, cost, i):

    # misfit between surface velocity
    if "velsurf" in cfg.processes.data_assimilation.cost_list:
        cost["velsurf"] = misfit_velsurf(cfg,state)

    # misfit between ice thickness profiles
    if "thk" in cfg.processes.data_assimilation.cost_list:
        cost["thk"] = misfit_thk(cfg, state)

    # misfit between divergence of flux
    if ("divfluxfcz" in cfg.processes.data_assimilation.cost_list):
        cost["divflux"] = cost_divfluxfcz(cfg, state, i)
    elif ("divfluxobs" in cfg.processes.data_assimilation.cost_list):
        cost["divflux"] = cost_divfluxobs(cfg, state, i)

    # misfit between top ice surfaces
    if "usurf" in cfg.processes.data_assimilation.cost_list:
        cost["usurf"] = misfit_usurf(cfg, state) 

    # add penalty terms to force obstacle constraints
    if "penalty" in cfg.processes.data_assimilation.optimization.obstacle_constraint:

        # force zero thikness outisde the mask
        if "icemask" in cfg.processes.data_assimilation.cost_list:
            cost["icemask"] = 10**10 * tf.math.reduce_mean( tf.where(state.icemaskobs > 0.5, 0.0, state.thk**2) )

        # Here one enforces non-negative ice thickness
        if "thk" in cfg.processes.data_assimilation.control_list:
            cost["thk_positive"] = \
            10**10 * tf.math.reduce_mean( tf.where(state.thk >= 0, 0.0, state.thk**2) )

        # Here one enforces non-negative slidinco
        if ("slidingco" in cfg.processes.data_assimilation.control_list) & \
            (not cfg.processes.data_assimilation.fitting.log_slidingco):
            cost["slidingco_positive"] =  \
            10**10 * tf.math.reduce_mean( tf.where(state.slidingco >= 0, 0.0, state.slidingco**2) ) 

        # Here one enforces non-negative arrhenius
        if ("arrhenius" in cfg.processes.data_assimilation.control_list):
            cost["arrhenius_positive"] =  \
            10**10 * tf.math.reduce_mean( tf.where(state.arrhenius >= 1, 0.0, state.arrhenius**2) ) 
        
    if cfg.processes.data_assimilation.cook.infer_params:
        cost["volume"] = cost_vol(cfg, state)

    # Here one adds a regularization terms for the bed toporgraphy to the cost function
    if "thk" in cfg.processes.data_assimilation.control_list:
        cost["thk_regu"] = regu_thk(cfg, state)

    # Here one adds a regularization terms for slidingco to the cost function
    if "slidingco" in cfg.processes.data_assimilation.control_list:
        cost["slid_regu"] = regu_slidingco(cfg, state)

    # Here one adds a regularization terms for arrhenius to the cost function
    if "arrhenius" in cfg.processes.data_assimilation.control_list:
        cost["arrh_regu"] = regu_arrhenius(cfg, state) 

    return tf.reduce_sum(tf.convert_to_tensor(list(cost.values())))