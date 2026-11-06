import numpy as np
import pytest
import tensorflow as tf
from omegaconf import OmegaConf

from igm.processes.iceflow.horizontal import HorizontalDiscrs


@pytest.mark.parametrize("name", sorted(HorizontalDiscrs))
@pytest.mark.parametrize("shape", [(3, 9, 11), (3, 2, 9, 11)])
def test_samples_of_a_batch_are_discretized_independently(name, shape):
    """A batch of tiles of one grid must give each tile the result it gets alone, with the
    quadrature axis right after the batch, for 2D [B, Ny, Nx] and 3D [B, Nz, Ny, Nx] fields."""
    cfg = OmegaConf.create({"processes": {"iceflow": {"numerics": {"precision": "single"}}}})
    discr = HorizontalDiscrs[name](cfg)
    rng = np.random.default_rng(0)
    X = tf.constant(rng.standard_normal(shape), tf.float32)
    dX = tf.fill((shape[0],) + shape[-2:], 50.0)

    batched = [discr.interp_h(X), *discr.grad_h(X, dX)]
    for i in range(shape[0]):
        alone = [discr.interp_h(X[i : i + 1]), *discr.grad_h(X[i : i + 1], dX[i : i + 1])]
        for b, a in zip(batched, alone):
            assert b.shape[0] == shape[0] and b.shape[1] == a.shape[1]
            np.testing.assert_allclose(b[i : i + 1].numpy(), a.numpy(), rtol=1e-6, atol=1e-6)
