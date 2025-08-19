#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from typing import Any, Dict

from ..energy import EnergyComponent


class SlidingComponent(EnergyComponent):
    pass


def get_sliding_params_args(cfg) -> Dict[str, Any]:

    cfg_numerics = cfg.processes.iceflow.numerics
    cfg_physics = cfg.processes.iceflow.physics

    law = cfg_physics.sliding.law

    return {"vert_basis": cfg_numerics.vert_basis, **cfg_physics.sliding[law]}
