from typing import List, Any
from types import ModuleType

import time

from ...core import State
from ...utilities import print_info
from ..modules.loader import load_modules


def initialize_modules(processes: List, cfg: Any, state: State) -> None:
    for module in processes:
        if cfg.core.logging:
            state.logger.info(f"Initializing module: {module.__name__.split('.')[-1]}")
        module.initialize(cfg, state)


def update_modules(processes: List, outputs: List, cfg: Any, state: State) -> None:

    state.it = 0
    state.continue_run = True
    if cfg.core.print_comp:
        state.tcomp = {
            module.__name__.split(".")[-1]: [] for module in processes + outputs
        }
    while state.continue_run:
        for module in processes:
            m = module.__name__.split(".")[-1]
            if cfg.core.print_comp:
                state.tcomp[m].append(time.time())
            module.update(cfg, state)
            if cfg.core.print_comp:
                state.tcomp[m][-1] -= time.time()
                state.tcomp[m][-1] *= -1
        run_outputs(outputs, cfg, state)
        if cfg.core.print_info:
            print_info(state)
        state.it += 1

        if not hasattr(state, "t"):
            state.continue_run = False


def finalize_modules(processes: List, cfg: Any, state: State) -> None:
    for module in processes:
        module.finalize(cfg, state)


def run_outputs(output_modules: List, cfg: Any, state: State) -> None:
    for module in output_modules:
        m = module.__name__.split(".")[-1]
        if cfg.core.print_comp:
            state.tcomp[m].append(time.time())
        module.run(cfg, state)
        if cfg.core.print_comp:
            state.tcomp[m][-1] -= time.time()
            state.tcomp[m][-1] *= -1


def setup_igm_modules(cfg, state) -> List[ModuleType]:
    return load_modules(cfg, state)
