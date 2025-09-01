import os
import sys

import igm
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fixtures import (
    empty_core_expected,
    override_core_expected,
    override_core_expected_cli,
    override_modules_expected,
    overriden_defaults_list_expected,
)

cwd = os.getcwd()
sys.path.append(cwd)


def test_empty_core():
    """Tests to see that if an empty yaml emperiment is loaded, the default values are loaded"""

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(
        version_base=None, config_dir=igm_config_dir, job_name="test_defaults"
    ):
        cfg = compose(
            config_name="config",
            overrides=[f"hydra.searchpath=[file://{cwd}]", "+experiment=empty"],
        )
        correct = OmegaConf.create(empty_core_expected)
        # print(OmegaConf.to_yaml(cfg), OmegaConf.to_yaml(correct))

        assert cfg == correct


def test_override_core():
    """Tests to see that we can override the core values with an experiment"""

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(
        version_base=None, config_dir=igm_config_dir, job_name="test_defaults"
    ):
        cfg = compose(
            config_name="config",
            overrides=[f"hydra.searchpath=[file://{cwd}]", "+experiment=override_core"],
        )
        correct = OmegaConf.create(override_core_expected)

        print(OmegaConf.to_yaml(cfg), OmegaConf.to_yaml(correct))

        assert cfg.core == correct.core


def test_override_core_syntax():
    """Tests to see that we can override the core values with an experiment"""

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(
        version_base=None, config_dir=igm_config_dir, job_name="test_defaults"
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"hydra.searchpath=[file://{cwd}]",
                "+experiment=override_core_syntax",
            ],
        )

        # Should be equivalent to dropping some of the defalut values (not working...)
        # cfg = compose(
        #     config_name="config",
        #     overrides=[
        #         f"hydra.searchpath=[file://{cwd}]",
        #         "+experiment=override_core",
        #         # "~core", # works as intended
        #         # "~modules/load_ncdf",
        #         # "~modules/smb_simple",
        #         # "~modules/iceflow",
        #         # "~modules/print_info",
        #     ],
        # )
        
        correct = OmegaConf.create(overriden_defaults_list_expected)
        # correct_modules = OmegaConf.create(overriden_defaults_list_expected)

        # print(OmegaConf.to_yaml(cfg), OmegaConf.to_yaml(correct))
        print(OmegaConf.to_yaml(cfg))

        assert cfg == correct
        # assert cfg.modules == correct_modules.modules


def test_override_modules():
    """Tests to see that we can override the module values with an experiment"""

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(
        version_base=None, config_dir=igm_config_dir, job_name="test_defaults"
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"hydra.searchpath=[file://{cwd}]",
                "+experiment=override_modules",
            ],
        )
        correct = OmegaConf.create(override_modules_expected)

        # print(OmegaConf.to_yaml(cfg), OmegaConf.to_yaml(correct))

        assert cfg.modules == correct.modules


def test_comments():
    """Tests to see that comments do not cause any issues"""

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(
        version_base=None, config_dir=igm_config_dir, job_name="test_defaults"
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"hydra.searchpath=[file://{cwd}]",
                "+experiment=override_with_comments",
            ],
        )
        correct = OmegaConf.create(override_core_expected)

        assert cfg.core == correct.core


def test_wrong_global_package_location():
    """Tests to confirm that if the global package header is not at the very top,
    the default values are loaded instead as the yaml file created a nested group instead of overriding default values
    """

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(
        version_base=None, config_dir=igm_config_dir, job_name="test_defaults"
    ):

        # package header in wrong location
        cfg = compose(
            config_name="config",
            overrides=[
                f"hydra.searchpath=[file://{cwd}]",
                "+experiment=wrong_global_package_location",
            ],
        )
        # no package header
        cfg_no_header = compose(
            config_name="config",
            overrides=[
                f"hydra.searchpath=[file://{cwd}]",
                "+experiment=without_package_header",
            ],
        )

        correct = OmegaConf.create(override_core_expected_cli)
        correct_default = OmegaConf.create(empty_core_expected)

        assert cfg_no_header.core != correct.core
        assert cfg.core != correct.core

        assert cfg.core == correct_default.core
        assert cfg_no_header.core == correct_default.core

        # both behave the same way and have the same bug
        assert cfg_no_header == cfg


def test_command_line_override():
    """Tests to see that we can override a yaml experiment file (meaning the yaml file overrode the defaults and then we overrode the yaml file)"""

    igm_config_dir = os.path.join(igm.__path__[0], "conf")
    with initialize_config_dir(
        version_base=None, config_dir=igm_config_dir, job_name="test_defaults"
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"hydra.searchpath=[file://{cwd}]",
                "+experiment=override_core",
                "core.saved_params_filename=overriden_params_cli",
                "modules.iceflow.iceflow.iflo_fieldin=[thk, usurf]",
                "core.gpu_id=3",
            ],
        )

        correct = OmegaConf.create(override_core_expected_cli)

        assert cfg.core == correct.core
        assert (
            cfg.modules.iceflow.iceflow.iflo_fieldin
            == correct.modules.iceflow.iceflow.iflo_fieldin
        )
        assert cfg.modules == correct.modules


# def test_incorrect_module_name():

#     igm_config_dir = os.path.join(igm.__path__[0], "conf")
#     with initialize_config_dir(version_base=None, config_dir=igm_config_dir, job_name="test_defaults"):

#         with pytest.raises(ModuleNotFoundError):
#             cfg = compose(config_name="config", overrides=[f"hydra.searchpath=[file://{cwd}]", "+experiment=incorrect_module_name"])

# correct = OmegaConf.create(override_core_expected_cli)

# assert cfg.core == correct.core
# assert cfg.modules == correct.modules
# assert cfg == correct


# if you run locally, use this...
# if __name__ == "__main__":

#     cwd = os.getcwd()
#     sys.path.append(cwd)

# print(cwd)
# test_empty_core()
