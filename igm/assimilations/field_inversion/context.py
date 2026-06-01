#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tensorflow as tf

from igm.processes.iceflow.utils.velocities import get_velsurf, get_velbar
from igm.processes.iceflow.utils.compute_divflux import compute_divflux
from igm.utils.math.precision import normalize_precision


_DEFAULT_MASK_CACHE_KEY = "__default_icemask__"


@dataclass
class DAEvaluationContext:
    """
    Per-objective-evaluation cache for DA quantities.

    IMPORTANT: any θ-dependent physical field MUST be accessed via da_map
    (ctx.physical(...)) to guarantee gradients flow properly.
    """

    cfg: Any
    state: Any
    da_map: Any
    U_in: tf.Tensor
    V_in: tf.Tensor
    inputs: Any

    _dtype: Optional[tf.DType] = None
    _U: Optional[tf.Tensor] = None
    _V: Optional[tf.Tensor] = None
    _dx: Optional[tf.Tensor] = None

    _uvelsurf: Optional[tf.Tensor] = None
    _vvelsurf: Optional[tf.Tensor] = None
    _ubar: Optional[tf.Tensor] = None
    _vbar: Optional[tf.Tensor] = None
    _divflux: Optional[tf.Tensor] = None

    _mask_cache: Optional[Dict[str, tf.Tensor]] = None
    _physical_cache: Optional[Dict[str, tf.Tensor]] = None

    def __post_init__(self) -> None:
        self._mask_cache = {}
        self._physical_cache = {}

    @property
    def dtype(self) -> tf.DType:
        if self._dtype is None:
            self._dtype = normalize_precision(self.cfg.processes.iceflow.numerics.precision)
        return self._dtype

    @property
    def U(self) -> tf.Tensor:
        if self._U is None:
            self._U = self.U_in[0]
        return self._U

    @property
    def V(self) -> tf.Tensor:
        if self._V is None:
            self._V = self.V_in[0]
        return self._V

    @property
    def dx(self) -> tf.Tensor:
        if self._dx is None:
            self._dx = tf.cast(self.state.dX, self.dtype)
        return self._dx

    def state_field(self, name: str) -> tf.Tensor:
        """Non-θ field from state, such as an observation or prior field."""
        if not hasattr(self.state, name):
            raise AttributeError(f"State has no field '{name}'")
        return tf.cast(tf.convert_to_tensor(getattr(self.state, name)), self.dtype)

    def physical(self, name: str) -> tf.Tensor:
        """θ-dependent physical field, cached within this objective evaluation."""
        assert self._physical_cache is not None
        if name not in self._physical_cache:
            self._physical_cache[name] = self.da_map.get_physical_field(name)
        return self._physical_cache[name]

    def model(self, name: str) -> tf.Tensor:
        """Model quantity provider."""
        if name == "uvelsurf":
            u, _ = self.velsurf()
            return u
        if name == "vvelsurf":
            _, v = self.velsurf()
            return v
        if name == "divflux":
            return self.divflux()
        return self.physical(name)

    def velsurf(self) -> Tuple[tf.Tensor, tf.Tensor]:
        if self._uvelsurf is None or self._vvelsurf is None:
            u, v = get_velsurf(self.U, self.V, self.state.iceflow.discr_v.V_s)
            self._uvelsurf = u
            self._vvelsurf = v
        return self._uvelsurf, self._vvelsurf
    
    def velbar(self) -> Tuple[tf.Tensor, tf.Tensor]:
        if self._ubar is None or self._vbar is None:
            u, v = get_velbar(
                self.U,
                self.V,
                self.state.iceflow.discr_v.V_bar,
            )
            self._ubar = u
            self._vbar = v
        return self._ubar, self._vbar

    def divflux(self) -> tf.Tensor:
        if self._divflux is None:
            thk = self.physical("thk")
            ubar, vbar = self.velbar()

            self._divflux = compute_divflux(
                ubar,
                vbar,
                thk,
                self.dx,
                self.dx,
                method="upwind",
            )
        return self._divflux

    def get_mask(self, mask_name: Optional[str]) -> tf.Tensor:
        """
        Return a boolean mask tensor.

        mask_name is None: state.icemask.
        mask_name is a string: state.<mask_name>, or AttributeError if missing.
        """
        assert self._mask_cache is not None
        key = _DEFAULT_MASK_CACHE_KEY if mask_name is None else mask_name
        if key in self._mask_cache:
            return self._mask_cache[key]

        if mask_name is None:
            mask = tf.cast(self.state.icemask, tf.bool)
        else:
            if not hasattr(self.state, mask_name):
                raise AttributeError(f"State has no mask field '{mask_name}'.")
            mask = tf.cast(getattr(self.state, mask_name), tf.bool)

        self._mask_cache[key] = mask
        return mask
