#!/usr/bin/env python3

"""
# Copyright (C) 2021-2025 IGM authors 
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import sys
import importlib
from typing import List, Tuple
from types import ModuleType
from hydra.core.hydra_config import HydraConfig

from .utils import get_module_name, get_orders
from .validator import validate_module


def load_modules(
    cfg, state
) -> Tuple[List[ModuleType], List[ModuleType], List[ModuleType]]:
    """Returns a list of actionable modules to then apply the update, initialize, finalize functions on for IGM."""
    imported_input_modules = []
    imported_modules = []
    imported_output_modules = []

    root_foldername = (
        f"{HydraConfig.get().runtime.cwd}/{cfg.core.structure.root_foldername}"
    )

    # Add custom modules folder to sys.path
    user_input_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.input_modules_foldername}"
    user_process_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.process_modules_foldername}"
    user_output_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.output_modules_foldername}"

    custom_modules_folders = [
        user_input_modules_folder,
        user_process_modules_folder,
        user_output_modules_folder,
    ]

    for folder in custom_modules_folders:
        if folder not in sys.path:
            sys.path.append(folder)

    if "inputs" in cfg:
        load_user_modules(
            cfg=cfg,
            state=state,
            modules_list=cfg.inputs,
            imported_modules_list=imported_input_modules,
            module_folder=user_input_modules_folder,
        )
        load_modules_igm(
            cfg=cfg,
            state=state,
            modules_list=cfg.inputs,
            imported_modules_list=imported_input_modules,
            module_type="inputs",
        )

    if "processes" in cfg:
        load_user_modules(
            cfg=cfg,
            state=state,
            modules_list=cfg.processes,
            imported_modules_list=imported_modules,
            module_folder=user_process_modules_folder,
        )
        load_modules_igm(
            cfg=cfg,
            state=state,
            modules_list=cfg.processes,
            imported_modules_list=imported_modules,
            module_type="processes",
        )

    if "outputs" in cfg:
        load_user_modules(
            cfg=cfg,
            state=state,
            modules_list=cfg.outputs,
            imported_modules_list=imported_output_modules,
            module_folder=user_output_modules_folder,
        )
        load_modules_igm(
            cfg=cfg,
            state=state,
            modules_list=cfg.outputs,
            imported_modules_list=imported_output_modules,
            module_type="outputs",
        )

    # Reorder modules
    input_order, module_order, output_order = get_orders()

    input_order_dict = {name: index for index, name in enumerate(input_order)}
    imported_input_modules = sorted(
        imported_input_modules,
        key=lambda module: input_order_dict[get_module_name(module)],
    )

    modules_order_dict = {name: index for index, name in enumerate(module_order)}
    imported_modules = sorted(
        imported_modules, key=lambda module: modules_order_dict[get_module_name(module)]
    )

    output_order_dict = {name: index for index, name in enumerate(output_order)}
    imported_output_modules = sorted(
        imported_output_modules,
        key=lambda module: output_order_dict[get_module_name(module)],
    )

    if cfg.core.print_imported_modules:
        print(f"{'':-^100}")
        print(f"{'INPUTS Modules':-^100}")
        for i, input_module in enumerate(imported_input_modules):
            print(f" {i}: {input_module}")
        print(f"{'PROCESSES Modules':-^100}")
        for i, module in enumerate(imported_modules):
            print(f" {i}: {module}")
        print(f"{'OUTPUTS Modules':-^100}")
        for i, output_module in enumerate(imported_output_modules):
            print(f" {i}: {output_module}")
        print(f"{'':-^100}")

    return imported_input_modules, imported_modules, imported_output_modules


def load_user_modules(
    cfg, state, modules_list, imported_modules_list, module_folder
) -> List[ModuleType]:

    from importlib.machinery import SourceFileLoader

    for module_name in modules_list:
        # Local Directory
        try:
            module = SourceFileLoader(
                f"{module_name}", f".{module_name}.py"
            ).load_module()
        except FileNotFoundError:

            # User Modules Folder
            try:
                module = SourceFileLoader(
                    f"{module_name}", f"{module_folder}/{module_name}.py"
                ).load_module()
            except FileNotFoundError:
                # User Modules Folder Folder
                try:
                    module = SourceFileLoader(
                        f"{module_name}",
                        f"{module_folder}/{module_name}/{module_name}.py",
                    ).load_module()
                except FileNotFoundError:
                    pass
                else:
                    imported_modules_list.append(module)
            else:
                imported_modules_list.append(module)
        else:
            imported_modules_list.append(module)

    return imported_modules_list


def load_modules_igm(
    cfg, state, modules_list, imported_modules_list, module_type
) -> List[ModuleType]:

    from importlib.machinery import SourceFileLoader

    imported_modules_names = [module.__name__ for module in imported_modules_list]
    for module_name in modules_list:
        if module_name in imported_modules_names:
            continue

        module_path = f"igm.{module_type}.{module_name}"
        module = importlib.import_module(module_path)
        if module_type == "processes":
            validate_module(module)
        imported_modules_list.append(module)
