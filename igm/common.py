#!/usr/bin/env python3

"""
# Copyright (C) 2021-2025 IGM authors 
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import os, json, sys
import importlib
from typing import List, Any, Dict, Tuple
from types import ModuleType
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

import igm

IGM_DESCRIPTION = r"""
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │             Welcome to IGM, a modular, open-source, fast, and user-friendly glacier evolution model!             │
  │                                                                                                                  │
  │                                                                                                                  │
  │                         __/\\\\\\\\\\\_____/\\\\\\\\\\\\__/\\\\____________/\\\\_                                │
  │                          _\/////\\\///____/\\\//////////__\/\\\\\\________/\\\\\\_                               │
  │                           _____\/\\\______/\\\_____________\/\\\//\\\____/\\\//\\\_                              │
  │                            _____\/\\\_____\/\\\____/\\\\\\\_\/\\\\///\\\/\\\/_\/\\\_                             │
  │                             _____\/\\\_____\/\\\___\/////\\\_\/\\\__\///\\\/___\/\\\_                            │
  │                              _____\/\\\_____\/\\\_______\/\\\_\/\\\____\///_____\/\\\_                           │
  │                               _____\/\\\_____\/\\\_______\/\\\_\/\\\_____________\/\\\_                          │
  │                                __/\\\\\\\\\\\_\//\\\\\\\\\\\\/__\/\\\_____________\/\\\_                         │
  │                                 _\///////////___\////////////____\///______________\///__                        │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""


class State:
    pass


from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
import datetime

def setup_igm_modules(cfg, state) -> List[ModuleType]:
    return load_modules(cfg, state)


def initialize_modules(processes: List, cfg: Any, state: State) -> None:
    for module in processes:
        if cfg.core.logging:
            state.logger.info(f"Initializing module: {module.__name__.split('.')[-1]}")
        module.initialize(cfg, state)

def print_info(state):
 
    if state.it % 100 == 1:
        if hasattr(state, "pbar"):
            state.pbar.close()
        state.pbar = tqdm(desc=f"IGM", 
                          ascii=False, 
                          dynamic_ncols=True,
                          bar_format="{desc} {postfix}")   

    if hasattr(state, "pbar"):
        dic_postfix= { 
            "🕒": datetime.datetime.now().strftime("%H:%M:%S"),
            "🔄": f"{state.it:06.0f}",
            "⏱ Time": f"{state.t.numpy():09.1f} yr",
            "⏳ Step": f"{state.dt:04.2f} yr",
        }
        if hasattr(state, "dx"):
            dic_postfix["❄️  Volume"] = f"{np.sum(state.thk) * (state.dx**2) / 10**9:108.2f} km³"
        if hasattr(state, "particle_x"):
            dic_postfix["# Particles"] = f"{state.particle_x.shape[0]}"

#        dic_postfix["💾 GPU Mem (MB)"] = tf.config.experimental.get_memory_info("GPU:0")['current'] / 1024**2

        state.pbar.set_postfix(dic_postfix)
        state.pbar.update(1)


def update_modules(processes: List, outputs: List, cfg: Any, state: State) -> None:

    state.it = 0
    state.continue_run = True
    if cfg.core.print_comp:
        state.tcomp = {module.__name__.split('.')[-1]: [] for module in processes+outputs}
    while state.continue_run:
        for module in processes:
            m=module.__name__.split('.')[-1]
            if cfg.core.print_comp:
                state.tcomp[m].append(time.time())
            module.update(cfg, state)
            if cfg.core.print_comp:
                state.tcomp[m][-1] -= time.time() ; state.tcomp[m][-1] *= -1
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
        m=module.__name__.split('.')[-1]
        if cfg.core.print_comp:
            state.tcomp[m].append(time.time())
        module.run(cfg, state)
        if cfg.core.print_comp:
            state.tcomp[m][-1] -= time.time() ; state.tcomp[m][-1] *= -1


def add_logger(cfg, state) -> None:

    # ! Ignore logging file for now...
    # if cfg.logging_file == "":
    #     pathf = ""
    # else:
    #     pathf = cfg.logging_file

    logging.basicConfig(
        # filename=pathf,
        encoding="utf-8",
        filemode="w",
        level=cfg.core.logging_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.root.setLevel(cfg.core.logging_level)

    state.logger = logging.getLogger("igm")

def get_module_name(module):
        return module.__name__.split('.')[-1]

def get_orders():
    import yaml
    config_path = [path["path"] for path in HydraConfig.get().runtime.config_sources if path["schema"] == "file"][0]
    
    with open(f'{config_path}/experiment/{HydraConfig.get().runtime.choices.experiment}.yaml', 'r') as file:
        original_experiment_config = yaml.safe_load(file)
    
    defaults = original_experiment_config['defaults']
    input_order = modules_order = output_order = []
    for default in defaults:

        key = list(default.keys())[0] # ? Cleaner / more robust way to do this?
        if key == 'override /inputs':
            input_order = default[key]
        elif key == 'override /processes':
            modules_order = default[key]
        elif key == 'override /outputs':
            output_order = default[key]

    return input_order, modules_order, output_order
        

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

    if "inputs" in cfg:
        user_input_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.input_modules_foldername}"
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
        user_process_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.process_modules_foldername}"
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
        user_output_modules_folder = f"{root_foldername}/{cfg.core.structure.code_foldername}/{cfg.core.structure.output_modules_foldername}"
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
    imported_input_modules = sorted(imported_input_modules, key=lambda module: input_order_dict[get_module_name(module)])
    
    modules_order_dict = {name: index for index, name in enumerate(module_order)}
    imported_modules = sorted(imported_modules, key=lambda module: modules_order_dict[get_module_name(module)])
    
    output_order_dict = {name: index for index, name in enumerate(output_order)}
    imported_output_modules = sorted(imported_output_modules, key=lambda module: output_order_dict[get_module_name(module)])

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

    # print("Testing for custom modules first - then will load default IGM modules...")
    for module_name in modules_list:
        # print("module name", module_name)
        # Local Directory
        try: 
            module = SourceFileLoader(
                f"{module_name}", f".{module_name}.py"
            ).load_module()
        except FileNotFoundError:
            # print(f'{module_name} [not found] in local working directory: {HydraConfig.get().runtime.cwd}. Trying custom modules directory...')

            # User Modules Folder
            try: 
                module = SourceFileLoader(
                    f"{module_name}", f"{module_folder}/{module_name}.py"
                ).load_module()
            except FileNotFoundError:
                # User Modules Folder Folder
                try:
                    module = SourceFileLoader(
                        f"{module_name}", f"{module_folder}/{module_name}/{module_name}.py"
                    ).load_module()
                except FileNotFoundError:
                    pass
                else:
                    # validate_module(module)
                    imported_modules_list.append(module)
                # print(f'{module_name} [not found] in local directory or custom modules directory')
            else:
                # print(f'{module_name} [found] in custom modules directory: {HydraConfig.get().runtime.cwd}/{cfg.core.custom_modules_folder}')
                # validate_module(module)
                imported_modules_list.append(module)
        else:
            # print(f'{module_name} [found] in local working directory: {HydraConfig.get().runtime.cwd}')
            # validate_module(module)
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

def validate_module(module) -> None:
    """Validates that a module has the required functions to be used in IGM."""
    required_functions = ["initialize", "finalize", "update"]
    for function in required_functions:
        if not hasattr(module, function):
            raise AttributeError(
                f"Module {module} is missing the required function ({function}). If it is a custom python package, make sure to include the 3 required functions: ['initialize', 'finalize', 'update'].",
                f"Please see https://github.com/jouvetg/igm/wiki/5.-Custom-modules-(coding) for more information on how to construct custom modules.",
            )

def print_gpu_info() -> None:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(f"{'CUDA Enviroment':-^150}")
    tf.sysconfig.get_build_info().pop("cuda_compute_capabilities", None)
    print(f"{json.dumps(tf.sysconfig.get_build_info(), indent=2, default=str)}")
    print(f"{'Available GPU Devices':-^150}")
    for gpu in gpus:
        gpu_info = {"gpu_id": gpu.name, "device_type": gpu.device_type}
        device_details = tf.config.experimental.get_device_details(gpu)
        gpu_info.update(device_details)

        print(f"{json.dumps(gpu_info, indent=2, default=str)}")
    print(f"{'':-^150}")


def download_unzip_and_store(url, folder_path) -> None:
    """
    Use wget to download a ZIP file and unzip its contents to a specified folder.

    Args:
    - url (str): The URL of the ZIP file to download.
    - folder_path (str): The path of the folder where the ZIP file's contents will be extracted.
    # - folder_name (str): The name of the folder where the ZIP file's contents will be extracted.
    """

    import subprocess
    import os
    import zipfile

    # Ensure the destination folder exists
    if not os.path.exists(folder_path):  # directory exists?
        os.makedirs(folder_path)

        # Download the file with wget
        logging.info("Downloading the ZIP file with wget...")
        subprocess.run(["wget", "-O", "downloaded_file.zip", url])

        # Unzipping the file
        logging.info("Unzipping the file...")
        with zipfile.ZipFile("downloaded_file.zip", "r") as zip_ref:
            zip_ref.extractall(folder_path)

        # Clean up (delete) the zip file after extraction
        os.remove("downloaded_file.zip")
        logging.info(f"File successfully downloaded and extracted to '{folder_path}'")

    else:
        logging.info(f"The data already exists at '{folder_path}'")


def print_comp(state):
    ################################################################

    size_of_tensor = {}

    for m in state.__dict__.keys():
        try:
            size_gb = sys.getsizeof(getattr(state, m).numpy())
            if size_gb > 1024**1:
                size_of_tensor[m] = size_gb / (1024**3)
        except:
            pass

    # sort from highest to lowest
    size_of_tensor = dict(
        sorted(size_of_tensor.items(), key=lambda item: item[1], reverse=True)
    )

    print("Memory statistics report:")
    with open("memory-statistics.txt", "w") as f:
        for key, value in size_of_tensor.items():
            print("     %24s  |  size : %8.4f Gb " % (key, value), file=f)
            print("     %24s  |  size : %8.4f Gb  " % (key, value))

    _plot_memory_pie(state)

    ################################################################

    modules = list(state.tcomp.keys())

    print("Computational statistics report:")
    with open("computational-statistics.txt", "w") as f:
        for m in modules:
            CELA = ( m, np.mean(state.tcomp[m]), np.sum(state.tcomp[m]) )
            print("     %14s  |  mean time per it : %8.4f  |  total : %8.4f" % CELA, file=f)
            print("     %14s  |  mean time per it : %8.4f  |  total : %8.4f" % CELA)

    _plot_computational_pie(state)


def _plot_computational_pie(state):
    """
    Plot to the computational time of each model components in a pie
    """

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return "{:.0f}".format(val)

        return my_autopct

    total = []
    name = []

    modules = list(state.tcomp.keys())

    for m in modules:
        total.append(np.sum(state.tcomp[m][1:]))
        name.append(m)

    sumallindiv = np.sum(total)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"), dpi=200)
    wedges, texts, autotexts = ax.pie(
        total, autopct=make_autopct(total), textprops=dict(color="w")
    )
    ax.legend(
        wedges,
        name,
        title="Model components",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    plt.setp(autotexts, size=8, weight="bold")
    #    ax.set_title("Matplotlib bakery: A pie")
    plt.tight_layout()
    plt.savefig("computational-pie.png", pad_inches=0)
    plt.close("all")


def _plot_memory_pie(state):
    """
    Plot to the memory size of each model components in a pie
    """

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return "{:.0f}".format(val)

        return my_autopct

    size_of_tensor = {}

    for m in state.__dict__.keys():
        try:
            size_gb = sys.getsizeof(getattr(state, m).numpy())
            if size_gb > 1024**1:
                size_of_tensor[m] = size_gb / (1024**3)
        except:
            pass

    size_of_tensor = dict(
        sorted(size_of_tensor.items(), key=lambda item: item[1], reverse=True)
    )

    total = list(size_of_tensor.values())[:10]
    name = list(size_of_tensor.keys())[:10]

    sumallindiv = np.sum(total)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"), dpi=200)
    wedges, texts, autotexts = ax.pie(
        total, autopct=make_autopct(total), textprops=dict(color="w")
    )
    ax.legend(
        wedges,
        name,
        title="Model components",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    plt.setp(autotexts, size=8, weight="bold")
    #    ax.set_title("Matplotlib bakery: A pie")
    plt.tight_layout()
    plt.savefig("memory-pie.png", pad_inches=0)
    plt.close("all")

###########################################################

# These function permits to read specific yaml files without calling hydra
# This is not ideal yet, used for instructed_oggm and testing

class EmptyClass:
    pass

class DictToObj:
    """Recursively convert a dictionary to an object with attribute-style access."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)  # Allow dictionary-like access

    def __repr__(self):
        return str(self.__dict__)

    def to_dict(self):
        """Convert back to a dictionary."""
        return {key: value.to_dict() if isinstance(value, DictToObj) else value for key, value in self.__dict__.items()}
 
def load_yaml_as_cfg(yaml_filename):

    import os, yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
    yaml_path = os.path.join(script_dir, yaml_filename)  # Build the full path

    with open(yaml_path, 'r') as file:
        yaml_dict = yaml.safe_load(file)  # Load as dict
    
    return DictToObj(yaml_dict)  # Convert to object

##########################################################

def load_yaml_recursive(base_dir):
    config = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_dir)
                keys = relative_path.replace('.yaml', '').replace('.yml', '').split(os.sep)

                # Load the YAML file
                yaml_conf = OmegaConf.load(full_path)

                # Nest it in the config dictionary
                sub_conf = config
                for key in keys[:-1]:
                    sub_conf = sub_conf.setdefault(key, {})
                if keys[-1] in yaml_conf:
                    sub_conf[keys[-1]] = OmegaConf.merge(sub_conf.get(keys[-1], {}), yaml_conf[keys[-1]])
                else:
                    sub_conf[keys[-1]] = yaml_conf

    return OmegaConf.create(config)


# this function checks if the parameters in the config file are compatible with the ones in the igm repository 
def check_incompatilities_in_parameters_file(cfg,path):

    from difflib import get_close_matches

    def flatten_dict(d, parent_key="", sep="."):
        def recurse(obj, prefix):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield from recurse(v, f"{prefix}{sep}{k}" if prefix else k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    yield from recurse(item, f"{prefix}[{i}]")
            else:
                yield (prefix, obj)

        return dict(recurse(d, parent_key))
        
    def compare_configs(cfg, cfgo, path="", excluded_keys=["cwd", "config"]):
        for key in cfg:
            full_path = f"{path}.{key}" if path else key
            if key not in excluded_keys:
                if key not in cfgo:
                    # Get possible matches for the missing key
                    posskeys=flatten_dict(OmegaConf.to_container(cfgo, resolve=False)).keys() 
                    suggestions = get_close_matches(key, posskeys, n=5, cutoff=0.2)
                    suggestions = [path+'.'+s for s in suggestions]
                    suggestion_msg = f" Did you mean '{suggestions}'?" if suggestions else ""
                    raise ValueError(f"Parameter '{full_path}' does not exist.\n {suggestion_msg}")
                if OmegaConf.is_dict(cfg[key]):
                    if not OmegaConf.is_dict(cfgo[key]):
                        raise ValueError(f"Configuration mismatch at '{full_path}': expected a dictionary-like config.")
                    compare_configs(cfg[key], cfgo[key], full_path)

    ############################

    cfgo = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))

    addo = load_yaml_recursive(os.path.join(path,'user/conf'))

    cfgo = OmegaConf.merge(cfgo, addo)

    compare_configs(cfg, cfgo)
 