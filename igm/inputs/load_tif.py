
import numpy as np
import os, glob
import tensorflow as tf

from .complete_data import complete_data

from .include_icemask import include_icemask
    
def run(cfg, state):
    import rasterio

    filepath = state.original_cwd.joinpath(cfg.core.folder_data)

    files = glob.glob(os.path.join(filepath, "*.tif"))
    
    print(files)
    
    for file in files:
        var = os.path.split(file)[-1].split(".")[0]
        if os.path.exists(file):
            state.profile_tif_file = rasterio.open(file, "r").profile
            with rasterio.open(file) as src:
                vars()[var] = np.flipud(src.read(1))
                height, width = vars()[var].shape
                x = np.array(rasterio.transform.xy(src.transform, 0, np.arange(width))[0]) 
                y = np.array(rasterio.transform.xy(src.transform, np.arange(height), 0)[1])
                y = y[::-1]
            del src

    # coarsen if requested
    if cfg.inputs.load_tif.coarsen > 1:
        xx = x[:: cfg.inputs.load_tif.coarsen]
        yy = y[:: cfg.inputs.load_tif.coarsen]
        for file in files:
            var = os.path.split(file)[-1].split(".")[0]
            if (not var in ["x", "y"]) & (vars()[var].ndim == 2):
                vars()[var] = vars()[var][
                    :: cfg.inputs.load_tif.coarsen, :: cfg.inputs.load_tif.coarsen
                ]
        #                vars()[var] = RectBivariateSpline(y, x, vars()[var])(yy, xx) # does not work
        x = xx
        y = yy

    # crop if requested
    if cfg.inputs.load_tif.crop:
        i0 = max(0, int((cfg.inputs.load_tif.xmin - x[0]) / (x[1] - x[0])))
        i1 = min(int((cfg.inputs.load_tif.xmax - x[0]) / (x[1] - x[0])), x.shape[0] - 1)
        i1 = max(i0 + 1, i1)
        j0 = max(0, int((cfg.inputs.load_tif.ymin - y[0]) / (y[1] - y[0])))
        j1 = min(int((cfg.inputs.load_tif.ymax - y[0]) / (y[1] - y[0])), y.shape[0] - 1)
        j1 = max(j0 + 1, j1)
        #        i0,i1 = int((cfg.inputs.load_tif.xmin-x[0])/(x[1]-x[0])),int((cfg.inputs.load_tif.xmax-x[0])/(x[1]-x[0]))
        #        j0,j1 = int((cfg.inputs.load_tif.ymin-y[0])/(y[1]-y[0])),int((cfg.inputs.load_tif.ymax-y[0])/(y[1]-y[0]))
        for file in files:
            var = os.path.split(file)[-1].split(".")[0]
            if not var in ["x", "y"]:
                vars()[var] = vars()[var][j0:j1, i0:i1]
        y = y[j0:j1]
        x = x[i0:i1]

    # transform from numpy to tensorflow
    for file in files:
        var = os.path.split(file)[-1].split(".")[0]
        vars(state)[var] = tf.Variable(vars()[var].astype("float32"), trainable=False)

    state.x = tf.constant(x.astype("float32"))
    state.y = tf.constant(y.astype("float32"))

    complete_data(state)

    if cfg.inputs.load_tif.icemask_include:
        include_icemask(state, mask_shapefile=cfg.inputs.load_tif.icemask_shapefile, mask_invert=cfg.inputs.load_tif.icemask_invert)