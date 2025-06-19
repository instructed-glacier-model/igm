#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file
 
import os  

from igm.inputs.oggm_shop.oggm_util import oggm_util
# from .make_input_file_old import make_input_file
from igm.inputs.oggm_shop.open_gridded_data import open_gridded_data
from igm.inputs.oggm_shop.arrange_data import arrange_data
from igm.inputs.oggm_shop.make_input_file import make_input_file

def run(cfg, state):

    if cfg.inputs.oggm_shop.RGI_ID=="":
        RGI_ID = cfg.inputs.oggm_shop.RGI_IDs[0]
    else:
        RGI_ID = cfg.inputs.oggm_shop.RGI_ID

    # Get the RGI version and product from the RGI_ID
    if (RGI_ID.count('-')==4)&(RGI_ID.split('-')[1][1]=='7'):
        RGI_version = 7
        RGI_product = RGI_ID.split('-')[2]
    elif (RGI_ID.count('-')==1)&(RGI_ID.split('-')[0][3]=='6'):
        RGI_version = 6
        RGI_product = None
    else:
        print("RGI version not recognized")

    path_data = os.path.join(state.original_cwd,cfg.core.folder_data)

    if cfg.inputs.oggm_shop.RGI_ID=="":
        path_RGIs = [os.path.join(path_data,path_RGI) for path_RGI in cfg.inputs.oggm_shop.RGI_IDs]
    else:
        path_RGIs = [os.path.join(path_data,cfg.inputs.oggm_shop.RGI_ID)]

    path_file = os.path.join(path_data,cfg.inputs.oggm_shop.filename)

    if not os.path.exists(path_data):
        os.makedirs(path_data)

    # Fetch the data from OGGM if it does not exist
    if not all(os.path.exists(p) for p in path_RGIs):
        oggm_util(cfg, path_RGIs, RGI_version, RGI_product)

    # transform the data into IGM readable data if it does not exist
    if not os.path.exists(path_file):
        # make_input_file(cfg, state, path_RGIs[0], path_file, RGI_version, RGI_product)

        ds = open_gridded_data(cfg, path_RGIs[0], state)

        ds_vars = arrange_data(cfg, state, path_RGIs[0], ds, RGI_version, RGI_product)

        make_input_file(cfg, ds, ds_vars, path_file)
 