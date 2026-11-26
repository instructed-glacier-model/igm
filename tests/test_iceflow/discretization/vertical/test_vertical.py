#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os

import pytest
from omegaconf import DictConfig

import igm
from igm.common.runner.configuration.loader import load_yaml_recursive
from igm.processes.iceflow.discretization.vertical.vertical import VerticalDiscr


@pytest.fixture
def cfg() -> DictConfig:
    cfg = load_yaml_recursive(os.path.join(igm.__path__[0], "conf"))
    return cfg


def test_base_class_abstract(cfg: DictConfig) -> None:
    with pytest.raises(TypeError):
        VerticalDiscr(cfg)


def test_base_class_build_matrices(cfg: DictConfig) -> None:

    class IncompleteDiscr(VerticalDiscr):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteDiscr(cfg)
