import os
import igm
from omegaconf import OmegaConf


from .loader import load_yaml_recursive


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
        return {
            key: value.to_dict() if isinstance(value, DictToObj) else value
            for key, value in self.__dict__.items()
        }


def check_incompatilities_in_parameters_file(cfg, path):

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
                    posskeys = flatten_dict(
                        OmegaConf.to_container(cfgo, resolve=False)
                    ).keys()
                    suggestions = get_close_matches(key, posskeys, n=5, cutoff=0.2)
                    suggestions = [path + "." + s for s in suggestions]
                    suggestion_msg = (
                        f" Did you mean '{suggestions}'?" if suggestions else ""
                    )
                    raise ValueError(
                        f"Parameter '{full_path}' does not exist.\n {suggestion_msg}"
                    )
                if OmegaConf.is_dict(cfg[key]):
                    if not OmegaConf.is_dict(cfgo[key]):
                        raise ValueError(
                            f"Configuration mismatch at '{full_path}': expected a dictionary-like config."
                        )
                    compare_configs(cfg[key], cfgo[key], full_path)

    ############################

    cfgo = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))

    addo = load_yaml_recursive(os.path.join(path, "user/conf"))

    cfgo = OmegaConf.merge(cfgo, addo)

    compare_configs(cfg, cfgo)
