import igm
import pytest
import os


@pytest.mark.skip(reason="API deprecated - get_modules_list no longer exists")
def test_load_valid_custom_module():
    print("s", os.getcwd())
    modules_dict = igm.get_modules_list(
        "./test_modules/param_files/valid_custom_module.json"
    )
    __ = igm.load_modules(modules_dict)


@pytest.mark.skip(reason="API deprecated - get_modules_list no longer exists")
def test_load_valid_custom_module_folder():
    modules_dict = igm.get_modules_list(
        "./test_modules/param_files/valid_custom_module_folder.json"
    )
    __ = igm.load_modules(modules_dict)


@pytest.mark.skip(reason="API deprecated - get_modules_list no longer exists")
def test_load_custom_module_missing_function():
    modules_dict = igm.get_modules_list(
        "./test_modules/param_files/missing_function_custom_module.json"
    )

    with pytest.raises(AttributeError):
        __ = igm.load_modules(modules_dict)


@pytest.mark.skip(reason="API deprecated - get_modules_list no longer exists")
def test_load_custom_module_folder_missing_function():
    modules_dict = igm.get_modules_list(
        "./test_modules/param_files/invalid_custom_module_folder.json"
    )

    with pytest.raises(AttributeError):
        __ = igm.load_modules(modules_dict)
