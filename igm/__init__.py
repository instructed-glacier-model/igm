#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import sys
import os
# sys.path.append(os.getcwd()) # I guess by default sys.path does not have cwd. not sure if this is the convention though...
    
from . import processes
from . import inputs, outputs

var = "test"

from . import common
# from .common import (
#     State,
#     IGM_DESCRIPTION,
#     load_modules,
#     add_logger,
#     initialize_modules,
#     update_modules,
#     finalize_modules,
#     run_outputs,
#     setup_igm_modules,
#     add_logger,
#     print_gpu_info,
#     download_unzip_and_store,
#     print_comp,
#     EmptyClass,
#     load_yaml_as_cfg,
#     load_yaml_recursive,
#     check_incompatilities_in_parameters_file
# )
