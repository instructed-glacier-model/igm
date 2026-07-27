import tensorflow as tf

from .misfit_thk import misfit_thk
from .misfit_thkprior import misfit_thkprior
from .misfit_usurf import misfit_usurf
from .misfit_velsurf import misfit_velsurf
from .misfit_vol import misfit_vol
from .cost_divfluxfcz import cost_divfluxfcz
from .cost_divfluxobs import cost_divfluxobs
from .cost_divfluxpen import cost_divfluxpen
from .cost_vol import cost_vol
from .regu_thk import regu_thk
from .regu_usurf import regu_usurf
from .regu_slidingco import regu_slidingco
from .regu_arrhenius import regu_arrhenius

def total_cost(cfg, state, cost, i):

    # misfit between surface velocity
    if "velsurf" in cfg.assimilations.data_assimilation.cost_list:
        cost["velsurf"] = misfit_velsurf(cfg,state)

    # misfit between ice thickness profiles
    if "thk" in cfg.assimilations.data_assimilation.cost_list:
        cost["thk"] = misfit_thk(cfg, state)

    # prior pulling thk toward thkinit within its per-pixel error tolerance —
    # prevents unconstrained thinning of unobserved glaciers
    if "thkprior" in cfg.assimilations.data_assimilation.cost_list:
        cost["thkprior"] = misfit_thkprior(cfg, state)

    # misfit between divergence of flux
    if ("divfluxfcz" in cfg.assimilations.data_assimilation.cost_list):
        cost["divflux"] = cost_divfluxfcz(cfg, state, i)
    elif ("divfluxobs" in cfg.assimilations.data_assimilation.cost_list):
        cost["divflux"] = cost_divfluxobs(cfg, state, i)
    elif ("divfluxpen" in cfg.assimilations.data_assimilation.cost_list):
        # pure smoothness penalty on divflux (no target)
        cost["divflux"] = cost_divfluxpen(cfg, state, i)

    # misfit between top ice surfaces
    if "usurf" in cfg.assimilations.data_assimilation.cost_list:
        cost["usurf"] = misfit_usurf(cfg, state) 
 
    if "volume_init" in cfg.assimilations.data_assimilation.cost_list:
        cost["volume"] = misfit_vol(cfg, state)

    # add penalty terms to force obstacle constraints
    if "penalty" in cfg.assimilations.data_assimilation.optimization.obstacle_constraint:

        # force zero thikness outisde the mask
        if "icemask" in cfg.assimilations.data_assimilation.cost_list:
            cost["icemask"] = 10**10 * tf.math.reduce_mean( tf.where(state.icemaskobs > 0.5, 0.0, state.thk**2) )

        # Here one enforces non-negative ice thickness
        if "thk" in cfg.assimilations.data_assimilation.control_list:
            cost["thk_positive"] = \
            10**10 * tf.math.reduce_mean( tf.where(state.thk >= 0, 0.0, state.thk**2) )

        # Here one enforces non-negative friction control (slidingco or tau_ref)
        fric = state.da_friction
        if (fric in cfg.assimilations.data_assimilation.control_list) & \
            (not cfg.assimilations.data_assimilation.fitting.log_slidingco):
            fric_field = getattr(state, fric)
            cost[f"{fric}_positive"] = \
            10**10 * tf.math.reduce_mean( tf.where(fric_field >= 0, 0.0, fric_field**2) )

        # Here one enforces non-negative arrhenius
        if ("arrhenius" in cfg.assimilations.data_assimilation.control_list):
            cost["arrhenius_positive"] =  \
            10**10 * tf.math.reduce_mean( tf.where(state.arrhenius >= 1, 0.0, state.arrhenius**2) ) 
        
    if cfg.assimilations.data_assimilation.cook.infer_params:
        cost["volume"] = cost_vol(cfg, state)

    # Here one adds a regularization terms for the bed toporgraphy to the cost function
    if "thk" in cfg.assimilations.data_assimilation.control_list:
        cost["thk_regu"] = regu_thk(cfg, state)

    # Smoothness of the surface-elevation deviation (usurf - usurfobs);
    # active only when usurf is a control and a positive weight is set.
    if ("usurf" in cfg.assimilations.data_assimilation.control_list
            and cfg.assimilations.data_assimilation.regularization.get("usurf", 0.0) > 0.0):
        cost["usurf_regu"] = regu_usurf(cfg, state)

    # Here one adds a regularization terms for the friction control
    # (slidingco or tau_ref) to the cost function
    if state.da_friction in cfg.assimilations.data_assimilation.control_list:
        cost["slid_regu"] = regu_slidingco(cfg, state)

    # Here one adds a regularization terms for arrhenius to the cost function
    if "arrhenius" in cfg.assimilations.data_assimilation.control_list:
        cost["arrh_regu"] = regu_arrhenius(cfg, state) 

    return tf.reduce_sum(tf.convert_to_tensor(list(cost.values())))