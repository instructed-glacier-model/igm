import os
from pathlib import Path
import zipfile

import pytest

ROOT_PACKAGE = "igm"  # top-level package
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT_PACKAGE_DIR = PROJECT_ROOT / ROOT_PACKAGE
DIST_DIR = PROJECT_ROOT / "dist"  # where the wheel is built


def get_all_package_names(root_dir):
    """Recursively find all package directories containing .py files."""
    package_names = []
    root_dir = Path(root_dir)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.endswith(".py") for f in filenames):
            rel_path = Path(dirpath).relative_to(root_dir)
            package_names.append(".".join([ROOT_PACKAGE, *rel_path.parts]))
    return package_names


def test_all_subpackages_have_init():
    """Ensure every subpackage directory has __init__.py"""
    missing = []
    for dirpath, dirnames, filenames in os.walk(ROOT_PACKAGE_DIR):
        if any(f.endswith(".py") for f in filenames) and "__init__.py" not in filenames:
            missing.append(os.path.relpath(dirpath, PROJECT_ROOT))
    assert not missing, f"Missing __init__.py in: {missing}"


def get_wheel_path():
    """Return the newest wheel path, or skip when no wheel has been built."""
    if not DIST_DIR.exists():
        pytest.skip(
            f"No wheel build directory found at {DIST_DIR}. "
            "Skipping wheel packaging check; build a wheel first with "
            "`python setup.py bdist_wheel`."
        )

    wheel_files = sorted(
        DIST_DIR.glob("*.whl"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    if not wheel_files:
        pytest.skip(
            f"No wheel file found in {DIST_DIR}. "
            "Skipping wheel packaging check; build a wheel first with "
            "`python setup.py bdist_wheel`."
        )
    return wheel_files[0]


def test_subpackages_in_wheel():
    """Check that every subpackage is included in the wheel"""
    wheel_path = get_wheel_path()

    # List all files in wheel
    with zipfile.ZipFile(wheel_path, "r") as whl:
        whl_files = set(whl.namelist())

    missing = []
    for package_name in get_all_package_names(ROOT_PACKAGE_DIR):
        # Construct expected path, e.g. igm/processes/iceflow/emulate/__init__.py
        pkg_path = "/".join([*package_name.split("."), "__init__.py"])
        if pkg_path not in whl_files:
            missing.append(package_name)

    assert not missing, f"Packages missing from wheel: {missing}"
