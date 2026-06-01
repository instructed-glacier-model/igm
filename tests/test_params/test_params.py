import igm
import pytest

# from igm.common import validate_module, load_modules_from_directory
from json import JSONDecodeError


@pytest.mark.skip(reason="API deprecated - get_modules_list no longer exists")
def test_load_json_params_with_comments():
    """Tests that adding comments to the json will not fail with a decode error."""

    igm.get_modules_list("./test_params/param_files/params_comments.json")


@pytest.mark.skip(reason="API deprecated - get_modules_list no longer exists")
def test_load_yaml_params_with_comments():
    """Tests that adding comments to the json will not fail with a decode error."""

    igm.get_modules_list("./test_params/param_files/params_comments.yaml")


@pytest.mark.skip(reason="API deprecated - get_modules_list no longer exists")
def test_load_igm_modules_json():
    """Tests that the core igm modules LIST loaded from the params.json file are loaded."""
    modules_dict = igm.get_modules_list("./test_params/param_files/params.json")

    preproc_modules = modules_dict["modules_preproc"]
    process_modules = modules_dict["modules_process"]
    postproc_modules = modules_dict["modules_postproc"]

    igm_core_preproc_modules = ["load_ncdf"]
    igm_core_process_modules = [
        "iceflow",
        "time",
        "thk",
        "rockflow",
        "vert_flow",
        "particles",
    ]
    igm_core_postproc_modules = ["write_ncdf", "plot2d", "print_info", "print_comp"]

    assert set(igm_core_preproc_modules).issubset(preproc_modules)
    assert set(igm_core_process_modules).issubset(process_modules)
    assert set(igm_core_postproc_modules).issubset(postproc_modules)


@pytest.mark.skip(reason="API deprecated - get_modules_list no longer exists")
def test_load_igm_modules_yaml():
    """Tests that the core igm modules LIST loaded from the params.json file are loaded."""
    modules_dict = igm.get_modules_list("./test_params/param_files/params.yaml")

    preproc_modules = modules_dict["modules_preproc"]
    process_modules = modules_dict["modules_process"]
    postproc_modules = modules_dict["modules_postproc"]

    igm_core_preproc_modules = ["load_ncdf"]
    igm_core_process_modules = [
        "iceflow",
        "time",
        "thk",
        "rockflow",
        "vert_flow",
        "particles",
    ]
    igm_core_postproc_modules = ["write_ncdf", "plot2d", "print_info", "print_comp"]

    assert set(igm_core_preproc_modules).issubset(preproc_modules)
    assert set(igm_core_process_modules).issubset(process_modules)
    assert set(igm_core_postproc_modules).issubset(postproc_modules)


@pytest.mark.skip(reason="API deprecated - get_modules_list no longer exists")
def test_load_modules():
    """Tests that the core igm modules LIST AND the custom modules from the params.json file are loaded."""
    modules_dict = igm.get_modules_list("./test_params/param_files/params.json")
    assert modules_dict == {
        "modules_preproc": ["load_ncdf", "track_usurf_obs"],
        "modules_process": [
            "clim_aletsch",
            "smb_accmelt",
            "iceflow",
            "time",
            "thk",
            "rockflow",
            "vert_flow",
            "particles",
        ],
        "modules_postproc": ["write_ncdf", "plot2d", "print_info", "print_comp"],
    }


@pytest.mark.skip(reason="API deprecated - params_core no longer exists")
def test_params_core():
    """Tests that the core igm parameters are loaded."""
    parser = igm.params_core()
    params, __ = parser.parse_known_args()

    assert vars(params) == {
        "param_file": "params.json",
        "modules_preproc": ["oggm_shop"],
        "modules_process": ["iceflow", "time", "thk"],
        "modules_postproc": ["write_ncdf", "plot2d", "print_info"],
        "logging": False,
        "logging_level": 30,
        "logging_file": "",
        "print_params": True,
        "gpu_info": False,
        "gpu_id": 0,
        "saved_params_filename": "params_saved",
        "url_data": "",
        "folder_data": "data",
    }


# def test_params_core_overwrite():
#     state = igm.State()  # class acting as a dictionary
#     parser = igm.params_core()
#     # args = ['--working_dir', 'test', '--gpu_info']  # replace with your arguments
#     parser = get_parser(parser)
#     __, params, __ = igm.setup_igm(state=state, parser=parser)

#     assert vars(params) == {
#         "working_dir": "test",
#         "param_file": "params.json",
#         "modules_preproc": ["oggm_shop"],
#         "modules_process": ["iceflow", "time", "thk"],
#         "modules_postproc": ["write_ncdf", "plot2d", "print_info"],
#         "logging": False,
#         "logging_level": 30,
#         "logging_file": "",
#         "print_params": True,
#         "gpu_info": False,
#         "gpu_id": 0,
#         "saved_params_filename": "params_saved",
#     }

# def get_parser(parser):
#     params, _ = parser.parse_known_args()
#     params_dict = vars(params)
#     # print(params_dict)
#     params_dict.update({"working_dir": "test"})
#     # print(params_dict)

#     parser.set_defaults(**params_dict)

#     # print(parser)
#     # params = parser.parse_args()
#     return parser

# def test_params_core_overwrite():
#     state = igm.State()  # class acting as a dictionary
#     parser = igm.params_core()

#     params, _ = parser.parse_known_args()
#     params_dict = vars(params)
#     print(params_dict)
#     params_dict.update({"working_dir": "test"})
#     print(params_dict)

#     parser.set_defaults(**params_dict)

#     print(parser)
#     params = parser.parse_args()
#     # args = ['--arg1', 'value1', '--arg2', 'value2']  # replace with your arguments
#     # parsed_args = parser(args)
#     __, params, __ = igm.setup_igm(state=state, parser=parser)

#     assert vars(params) == {
#         "working_dir": "test",
#         "param_file": "params.json",
#         "modules_preproc": ["oggm_shop"],
#         "modules_process": ["iceflow", "time", "thk"],
#         "modules_postproc": ["write_ncdf", "plot2d", "print_info"],
#         "logging": False,
#         "logging_level": 30,
#         "logging_file": "",
#         "print_params": True,
#         "gpu_info": False,
#         "gpu_id": 0,
#         "saved_params_filename": "params_saved",
#     }


# def test_params_json():
#     parser = igm.params_core()
#     params, _ = parser.parse_known_args()

#     modules_dict = igm.get_modules_list(params.param_file)
#     imported_modules = igm.load_modules(modules_dict)
#     imported_modules = igm.load_dependent_modules(imported_modules)

#     for module in imported_modules:
#         module.params(parser)

#     core_and_module_params = parser.parse_args()
#     params = igm.load_user_defined_params(
#         param_file=core_and_module_params.param_file,
#         params_dict=vars(core_and_module_params),
#     )

#     parser.set_defaults(**params)
#     params = parser.parse_args()


# def test_params_cli():
#     parser = igm.params_core()
#     params, _ = parser.parse_known_args()

#     modules_dict = igm.get_modules_list(params.param_file)
#     imported_modules = igm.load_modules(modules_dict)
#     imported_modules = igm.load_dependent_modules(imported_modules)

#     for module in imported_modules:
#         module.params(parser)

#     core_and_module_params = parser.parse_args()
#     params = igm.load_user_defined_params(
#         param_file=core_and_module_params.param_file,
#         params_dict=vars(core_and_module_params),
#     )

#     parser.set_defaults(**params)
#     params = parser.parse_args()
