import igm
import tensorflow as tf
import pytest
import numpy as np
from make_fake_ncdf import write_ncdf
import os
 
def test_load_ncdf():
    
    write_ncdf()
 
    state = igm.State()

    state.original_cwd = ""

    cfg = igm.load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
    
    cfg.core.folder_data = ""
    cfg.inputs.load_ncdf.input_file = 'input.nc'
 
    igm.inputs.load_ncdf.run(cfg, state)

    ny,nx = state.thk.shape

    mid = state.topg[int(ny/2),int(nx/2)]
 
    assert (mid>2450)&(mid<2550)

    for f in ['input.nc']:
        if os.path.exists(f):
            os.remove(f)
