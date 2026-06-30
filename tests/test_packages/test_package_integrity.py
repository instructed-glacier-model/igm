import os
import zipfile
import pytest

ROOT_PACKAGE = "igm"  # top-level package
DIST_DIR = "dist"  # where the wheel is built


def get_all_subpackages(root_dir):
    """Recursively find all subdirectories containing .py files"""
    subpackages = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.endswith(".py") for f in filenames):
            rel_path = os.path.relpath(dirpath, root_dir)
            subpackages.append(rel_path.replace(os.sep, "."))
    return subpackages


def test_all_subpackages_have_init():
    """Ensure every subpackage directory has __init__.py"""
    missing = []
    for dirpath, dirnames, filenames in os.walk(ROOT_PACKAGE):
        if any(f.endswith(".py") for f in filenames) and "__init__.py" not in filenames:
            missing.append(dirpath)
    assert not missing, f"Missing __init__.py in: {missing}"


@pytest.mark.parametrize("subpkg", get_all_subpackages(ROOT_PACKAGE))
def test_subpackages_in_wheel(subpkg):
    """Check that every subpackage is included in the wheel"""
    # Find wheel file
    wheel_files = [f for f in os.listdir(DIST_DIR) if f.endswith(".whl")]
    assert wheel_files, f"No wheel file found in {DIST_DIR}"
    wheel_path = os.path.join(DIST_DIR, wheel_files[0])

    # List all files in wheel
    with zipfile.ZipFile(wheel_path, "r") as whl:
        whl_files = whl.namelist()

    # Construct expected path
    # e.g., igm/processes/iceflow/emulate/__init__.py
    pkg_path = os.path.join(ROOT_PACKAGE, *subpkg.split("."), "__init__.py").replace(
        os.sep, "/"
    )
    assert pkg_path in whl_files, f"Package {subpkg} missing from wheel"
