import sys
import os
import importlib
import pytest

PACKAGE_NAME = "igm"

# List of submodules or subpackages to smoke-test imports
IMPORTS_TO_TEST = [
    "processes.iceflow.emulate.emulators",
    "inputs",
    "outputs",
    "common",
    "processes",
    "assimilations",
    "utils",
    "conf",
    "conf_help",
    # add more critical paths here
]

def test_imports_from_installed_package():
    """
    Smoke test: remove local repo from sys.path to ensure we are importing from
    installed wheel, then try importing critical subpackages/modules.
    """

    # Remove current working directory from sys.path to prevent cheating
    cwd = os.getcwd()
    sys.path = [p for p in sys.path if cwd not in p and p != ""]

    # Attempt imports
    for mod in IMPORTS_TO_TEST:
        full_mod_name = f"{PACKAGE_NAME}.{mod}"
        importlib.import_module(full_mod_name)