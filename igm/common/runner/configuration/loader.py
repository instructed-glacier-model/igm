import os, yaml
from omegaconf import OmegaConf


def load_yaml_as_cfg(yaml_filename):
    from .utils import DictToObj

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )  # Get the script's directory
    yaml_path = os.path.join(script_dir, yaml_filename)  # Build the full path

    with open(yaml_path, "r") as file:
        yaml_dict = yaml.safe_load(file)  # Load as dict

    return DictToObj(yaml_dict)  # Convert to object


def load_yaml_recursive(base_dir):
    config = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, base_dir)
                keys = (
                    relative_path.replace(".yaml", "").replace(".yml", "").split(os.sep)
                )

                # Load the YAML file
                yaml_conf = OmegaConf.load(full_path)

                # Nest it in the config dictionary
                sub_conf = config
                for key in keys[:-1]:
                    sub_conf = sub_conf.setdefault(key, {})
                if keys[-1] in yaml_conf:
                    sub_conf[keys[-1]] = OmegaConf.merge(
                        sub_conf.get(keys[-1], {}), yaml_conf[keys[-1]]
                    )
                else:
                    sub_conf[keys[-1]] = yaml_conf

    return OmegaConf.create(config)
