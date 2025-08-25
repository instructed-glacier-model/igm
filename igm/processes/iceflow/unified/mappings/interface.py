#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from abc import ABC, abstractmethod
from typing import Any, Dict

from igm.common.core import State


class InterfaceMapping(ABC):

    @staticmethod
    @abstractmethod
    def get_mapping_args(cfg: DictConfig, state: State) -> Dict[str, Any]:
        raise NotImplementedError(
            "❌ The get_mapping_args static method is not implemented."
        )
