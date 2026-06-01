import igm
import os
import pkgutil
import yaml


def test_yaml_files_exist():
    # Example for one subpackage
    data = pkgutil.get_data("igm.conf", "config.yaml")
    assert data is not None
    cfg = yaml.safe_load(data)
