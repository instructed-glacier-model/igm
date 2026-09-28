#!/usr/bin/env python3

# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Constants, configuration checks, segment reductions and neighbour stencils."""

import warnings
from typing import List, Sequence, Tuple, Union

import tensorflow as tf
from omegaconf import DictConfig

SECONDS_PER_YEAR = 31556926.0  # as in the enthalpy process


def require_process(cfg: DictConfig, process: str, law: str) -> None:
    """Raise unless ``process`` is active, naming the melt law that needs it."""
    if process not in cfg.processes:
        raise ValueError(
            f"The bmb melt law {law!r} needs the {process!r} process; add it "
            "to the processes of the experiment."
        )


def densities(cfg: DictConfig, law: str) -> Tuple[float, float]:
    """Ice and sea-water densities (kg m-3), shared with the ice flow.

    The melt laws are calibrated with sea water (about 1028 kg m-3); the ice
    flow's default is fresh water (1000 kg m-3), hence the warning.
    """
    require_process(cfg, "iceflow", law)
    physics = cfg.processes.iceflow.physics
    ice_density, water_density = float(physics.ice_density), float(
        physics.water_density
    )
    if water_density < 1020.0:
        warnings.warn(
            f"The bmb melt law {law!r} uses the ice-flow water density "
            f"{water_density:g} kg m-3; set cfg.processes.iceflow.physics."
            "water_density to that of sea water (about 1028) for ocean melt."
        )
    return ice_density, water_density


def segment_means(
    fields: Sequence[tf.Tensor],
    ids: tf.Tensor,
    mask: tf.Tensor,
    num_segments: Union[int, tf.Tensor],
) -> Tuple[List[tf.Tensor], tf.Tensor]:
    """Means of ``fields`` over the cells of ``mask``, per segment.

    ``ids`` are int32 segment ids in ``[0, num_segments)``; cells outside
    ``mask`` do not contribute. Returns, at every cell, the means of its
    segment (0 for a segment without cells in ``mask``) and the number of
    cells of ``mask`` in it. All fields are reduced in one scatter and
    broadcast back in one gather. The sums are accumulated in float64: in
    float32, a sum over millions of cells loses the mean.
    """
    dtype = fields[0].dtype
    data = tf.stack(
        [tf.where(mask, tf.cast(f, tf.float64), 0.0) for f in fields]
        + [tf.cast(mask, tf.float64)],
        axis=-1,
    )
    total = tf.math.unsorted_segment_sum(data, ids, num_segments)
    count = total[:, -1:]
    table = tf.concat([tf.math.divide_no_nan(total[:, :-1], count), count], axis=-1)
    at_cell = tf.cast(tf.gather(table, ids), dtype)
    return tf.unstack(at_cell[..., :-1], axis=-1), at_cell[..., -1]


EDGE = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def shifted(padded: tf.Tensor, di: int, dj: int) -> tf.Tensor:
    """View of a once-padded field at the neighbour offset ``(di, dj)``."""
    rows = slice(1 + di, di - 1 if di < 1 else None)
    cols = slice(1 + dj, dj - 1 if dj < 1 else None)
    return padded[rows, cols]


def neighbours(
    x: tf.Tensor, offsets: Sequence[Tuple[int, int]], fill: Union[int, float, bool] = 0
) -> List[tf.Tensor]:
    """Views of the 2-D ``x`` shifted to each offset, ``fill`` beyond the edges."""
    p = tf.pad(x, [[1, 1], [1, 1]], constant_values=fill)
    return [shifted(p, di, dj) for di, dj in offsets]


def any_neighbour(mask: tf.Tensor, offsets: Sequence[Tuple[int, int]]) -> tf.Tensor:
    """True where any neighbour at ``offsets`` is True."""
    return tf.reduce_any(tf.stack(neighbours(mask, offsets, False)), axis=0)
