from hydra.core.hydra_config import HydraConfig
import yaml


def get_orders():
    config_path = [
        path["path"]
        for path in HydraConfig.get().runtime.config_sources
        if path["schema"] == "file"
    ][0]

    with open(
        f"{config_path}/experiment/{HydraConfig.get().runtime.choices.experiment}.yaml",
        "r",
    ) as file:
        original_experiment_config = yaml.safe_load(file)

    defaults = original_experiment_config["defaults"]
    input_order = modules_order = output_order = []
    for default in defaults:

        key = list(default.keys())[0]  # ? Cleaner / more robust way to do this?
        if key == "override /inputs":
            input_order = default[key]
        elif key == "override /processes":
            modules_order = default[key]
        elif key == "override /outputs":
            output_order = default[key]

    return input_order, modules_order, output_order


def get_module_name(module):
    return module.__name__.split(".")[-1]
