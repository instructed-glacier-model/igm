#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from abc import ABC, abstractmethod
from typing import Tuple, Union

TV = Union[tf.Tensor, tf.Variable]


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions on velocity fields."""

    def __call__(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply boundary condition to velocity components (callable interface)."""
        return self.apply(U, V)

    @abstractmethod
    def apply(self, U: TV, V: TV) -> Tuple[TV, TV]:
        """Apply boundary condition to velocity components (must be implemented by subclasses)."""
        pass
