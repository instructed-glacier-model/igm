#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from typing import List

from . import (
    EnergyComponent,
    EnergyComponents,
    EnergyParams,
    get_energy_params_args,
)


def get_energy_components(cfg: DictConfig) -> List[EnergyComponent]:
    """Get the list of energy components objects."""

    cfg_physics = cfg.processes.iceflow.physics

    # Validate component-conditional inputs.
    # The floating component reads `water_level` from fieldin to detect
    # water-bordering ice cells and to compute the calving-front pressure.
    if "floating" in cfg_physics.energy_components:
        unified_inputs = list(cfg.processes.iceflow.unified.inputs)
        if "water_level" not in unified_inputs:
            raise ValueError(
                "❌ The 'floating' energy component requires 'water_level' in "
                "cfg.processes.iceflow.unified.inputs. "
                f"Current inputs: {unified_inputs}. "
                "Add 'water_level' to that list. The field is auto-populated "
                "as a uniform field from cfg.processes.thk.default_sealevel "
                "if no water_level variable is found in the input NetCDF."
            )

    energy_components = []
    for component in cfg_physics.energy_components:
        if component not in EnergyComponents:
            raise ValueError(f"❌ Unknown energy component: <{component}>.")

        # Get component and params class
        if component == "sliding":
            law = cfg_physics.sliding.law
            component_class = EnergyComponents[component][law]
            params_class = EnergyParams[component][law]
        else:
            component_class = EnergyComponents[component]
            params_class = EnergyParams[component]

        # Get args extractor
        get_params_args = get_energy_params_args[component]

        # Instantiate params and component classes
        params_args = get_params_args(cfg)
        params = params_class(**params_args)
        component_obj = component_class(params)

        # Add component to the list of components
        energy_components.append(component_obj)

    return energy_components
