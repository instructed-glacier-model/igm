"""The frictional heating of the enthalpy module equals the dissipation of the
ice-flow sliding energy, tau_b * u_b, for any reference speed u_ref."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

import igm
from igm.processes.enthalpy.dissipation.utils import compute_friction_heat
from igm.processes.iceflow.energy.components.sliding import (
    SlidingComponents,
    SlidingParams,
    get_sliding_params_args,
)
from igm.processes.iceflow.horizontal import HorizontalDiscrs
from igm.processes.iceflow.vertical import VerticalDiscrs

DTYPE = tf.float64
SECONDS_PER_YEAR = 31556926.0
NY, NX = 6, 8


def _cfg(law, u_ref, exponent):
    defaults = OmegaConf.load(
        Path(igm.__file__).parent / "conf" / "processes" / "iceflow.yaml"
    )
    inputs = ["thk", "usurf", "arrhenius", "tau_ref", "dX", "effective_pressure"]
    return OmegaConf.merge(
        {"processes": defaults},
        {
            "processes": {
                "iceflow": {
                    "numerics": {
                        "precision": "double",
                        "basis_horizontal": "q1",
                        "basis_vertical": "lagrange",
                        "Nz": 2,
                    },
                    "physics": {
                        "sliding": {
                            "law": law,
                            "use_mask_gr": False,
                            "u_ref": u_ref,
                            "exponent": exponent,
                        }
                    },
                    "unified": {"inputs": inputs},
                }
            }
        },
    )


@pytest.mark.unit
@pytest.mark.parametrize("law", ["weertman", "mohr_coulomb"])
@pytest.mark.parametrize("u_ref", [1.0, 100.0, 500.0])
@pytest.mark.parametrize("speed", [30.0, 700.0])
def test_friction_heat_equals_sliding_dissipation(law, u_ref, speed):
    exponent = 3.0
    cfg = _cfg(law, u_ref, exponent)
    shape = (NY, NX)
    fields = {
        "thk": tf.fill(shape, tf.constant(500.0, DTYPE)),
        "usurf": tf.fill(shape, tf.constant(600.0, DTYPE)),
        "topg": tf.fill(shape, tf.constant(100.0, DTYPE)),
        "water_level": tf.zeros(shape, DTYPE),
        "arrhenius": tf.fill(shape, tf.constant(78.0, DTYPE)),
        "tau_ref": tf.fill(shape, tf.constant(0.2, DTYPE)),
        "effective_pressure": tf.fill(shape, tf.constant(0.3, DTYPE)),
        "dX": tf.fill(shape, tf.constant(1000.0, DTYPE)),
    }
    numerics = cfg.processes.iceflow.numerics
    discr_h = HorizontalDiscrs[numerics.basis_horizontal](cfg)
    discr_v = VerticalDiscrs[numerics.basis_vertical](cfg)
    U = tf.fill((numerics.Nz, NY, NX), tf.constant(speed, DTYPE))
    V = tf.zeros_like(U)

    component = SlidingComponents[law](
        SlidingParams[law](**get_sliding_params_args(cfg))
    )
    fieldin = {k: v[None] for k, v in fields.items()}
    energy = component.cost(U[None], V[None], fieldin, discr_h, discr_v)
    dissipation = (1.0 + 1.0 / exponent) * float(np.mean(energy.numpy()))

    state = SimpleNamespace(
        **fields, U=U, V=V, iceflow=SimpleNamespace(discr_v=discr_v)
    )
    heat = compute_friction_heat(cfg, state).numpy()[1:-1, 1:-1]
    heat_per_year = heat * SECONDS_PER_YEAR / 1.0e6

    np.testing.assert_allclose(heat_per_year, dissipation, rtol=1e-6)
