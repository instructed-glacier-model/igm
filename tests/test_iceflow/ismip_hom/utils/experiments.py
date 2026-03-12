#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional

from igm.common import State

from hydra.core.hydra_config import HydraConfig
import os


class Experiment(ABC):
    CONST_FIELDS = {"x", "y", "X", "Y"}

    @abstractmethod
    def init_fields(
        self, L: Optional[float] = None, n: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        pass

    @classmethod
    def init_state(
        cls,
        state: State,
        L: Optional[float] = None,
        n: Optional[int] = None,
        dtype: tf.DType = tf.float32,
    ) -> None:
        fields_np = cls().init_fields(L, n)

        for field, value in fields_np.items():
            tensor = (
                tf.constant(value, dtype=dtype)
                if field in cls.CONST_FIELDS
                # else tf.Variable(value)
                else tf.Variable(value, dtype=dtype)
            )
            setattr(state, field, tensor)


class ExperimentA(Experiment):
    def init_fields(self, L: float, n: int) -> Dict[str, np.ndarray]:
        nx, ny = n, n
        x = np.linspace(0.0, L, nx)
        y = np.linspace(0.0, L, ny)
        dx = x[1] - x[0]
        X, Y = np.meshgrid(x, y)
        dX = dx * np.ones_like(X)

        α_x = np.deg2rad(0.5)
        ω = 2 * np.pi / L
        z_s = -X * np.tan(α_x)
        h = 1000.0 - 500.0 * np.sin(ω * X) * np.sin(ω * Y)
        z_b = z_s - h
        C = 1.0 * np.ones_like(X)
        A = 100.0 * np.ones_like(X)

        return {
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "dx": dx,
            "dX": dX,
            "thk": h,
            "topg": z_b,
            "usurf": z_s,
            "slidingco": C,
            "arrhenius": A,
        }


class ExperimentB(Experiment):
    def init_fields(self, L: float, n: int) -> Dict[str, np.ndarray]:
        nx = n
        x = np.linspace(0.0, L, nx)
        dx = x[1] - x[0]
        y = np.array([0.0, dx])
        X, Y = np.meshgrid(x, y)
        dX = dx * np.ones_like(X)

        α_x = np.deg2rad(0.5)
        ω = 2 * np.pi / L
        z_s = -X * np.tan(α_x)
        h = 1000.0 - 500.0 * np.sin(ω * X)
        z_b = z_s - h
        C = 1.0 * np.ones_like(X)
        A = 100.0 * np.ones_like(X)

        return {
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "dx": dx,
            "dX": dX,
            "thk": h,
            "topg": z_b,
            "usurf": z_s,
            "slidingco": C,
            "arrhenius": A,
        }


class ExperimentC(Experiment):
    def init_fields(self, L: float, n: int) -> Dict[str, np.ndarray]:
        nx, ny = n, n
        x = np.linspace(0.0, L, nx)
        y = np.linspace(0.0, L, ny)
        dx = x[1] - x[0]
        X, Y = np.meshgrid(x, y)
        dX = dx * np.ones_like(X)

        α_x = np.deg2rad(0.1)
        ω = 2 * np.pi / L
        z_s = -X * np.tan(α_x)
        h = 1000.0 * np.ones_like(X)
        z_b = z_s - h
        C = 1e-3 + 1e-3 * np.sin(ω * X) * np.sin(ω * Y)
        A = 100.0 * np.ones_like(X)

        return {
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "dx": dx,
            "dX": dX,
            "thk": h,
            "topg": z_b,
            "usurf": z_s,
            "slidingco": C,
            "arrhenius": A,
        }


class ExperimentCInversion(Experiment):
    def init_fields(self, L: float, n: int) -> Dict[str, np.ndarray]:
        nx, ny = n, n
        x = np.linspace(0.0, L, nx)
        y = np.linspace(0.0, L, ny)
        dx = x[1] - x[0]
        X, Y = np.meshgrid(x, y)
        dX = dx * np.ones_like(X)

        α_x = np.deg2rad(0.1)
        ω = 2 * np.pi / L
        z_s = -X * np.tan(α_x)
        h = 1000.0 * np.ones_like(X)
        z_b = z_s - h
        C = 1e-3 + 1e-3 * np.sin(ω * X) * np.sin(ω * Y)
        A = 100.0 * np.ones_like(X)

        return {
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "dx": dx,
            "dX": dX,
            "thk": h,
            "topg": z_b,
            "usurf": z_s,
            "slidingco": tf.Variable(
                tf.ones_like(h, dtype=tf.float32), trainable=True, name="slidingco"
            ),
            "arrhenius": A,
        }


class ExperimentD(Experiment):
    def init_fields(self, L: float, n: int) -> Dict[str, np.ndarray]:
        nx = n
        x = np.linspace(0.0, L, nx)
        dx = x[1] - x[0]
        y = np.array([0.0, dx])
        X, Y = np.meshgrid(x, y)
        dX = dx * np.ones_like(X)

        α_x = np.deg2rad(0.1)
        ω = 2 * np.pi / L
        z_s = -X * np.tan(α_x)
        h = 1000.0 * np.ones_like(X)
        z_b = z_s - h
        C = 1e-3 + 1e-3 * np.sin(ω * X)
        A = 100.0 * np.ones_like(X)

        return {
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "dx": dx,
            "dX": dX,
            "thk": h,
            "topg": z_b,
            "usurf": z_s,
            "slidingco": C,
            "arrhenius": A,
        }


class ExperimentE1(Experiment):
    def init_fields(
        self, L: Optional[float] = None, n: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        path_cwd = HydraConfig.get().runtime.cwd
        path_data = os.path.join(path_cwd, "..", "data", "arolla", "arolla100.dat")
        data = np.loadtxt(path_data)
        x = data[:, 0]
        z_b = data[:, 1]
        z_s = data[:, 2]
        h = z_s - z_b

        dx = x[1] - x[0]
        y = np.array([0.0, dx])
        z_b = np.tile(z_b, (2, 1))
        z_s = np.tile(z_s, (2, 1))
        h = np.tile(h, (2, 1))
        X, Y = np.meshgrid(x, y)
        dX = dx * np.ones_like(X)

        C = 1.0 * np.ones_like(X)
        A = 100.0 * np.ones_like(X)

        return {
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "dx": dx,
            "dX": dX,
            "thk": h,
            "topg": z_b,
            "usurf": z_s,
            "slidingco": C,
            "arrhenius": A,
        }


class ExperimentE2(Experiment):
    def init_fields(
        self, L: Optional[float] = None, n: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        path_cwd = HydraConfig.get().runtime.cwd
        path_data = os.path.join(path_cwd, "..", "data", "arolla", "arolla100.dat")
        data = np.loadtxt(path_data)
        x = data[:, 0]
        z_b = data[:, 1]
        z_s = data[:, 2]
        idx_bc = data[:, 3]
        h = z_s - z_b

        dx = x[1] - x[0]
        y = np.array([0.0, dx])
        z_b = np.tile(z_b, (2, 1))
        z_s = np.tile(z_s, (2, 1))
        h = np.tile(h, (2, 1))
        X, Y = np.meshgrid(x, y)
        dX = dx * np.ones_like(X)

        C = 1.0 * np.ones_like(X)
        C[:, idx_bc == 1.0] = 0.0
        A = 100.0 * np.ones_like(X)

        return {
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "dx": dx,
            "dX": dX,
            "thk": h,
            "topg": z_b,
            "usurf": z_s,
            "slidingco": C,
            "arrhenius": A,
        }
