#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import random

import tensorflow as tf

from .io_tfrecords import load_metadata, list_shards, make_datasets


@dataclass(frozen=True)
class TFRecordDatasets:
    metadata: dict
    H: int
    W: int
    Cx: int
    train_ds: tf.data.Dataset
    val_ds: tf.data.Dataset


def validate_dataset_matches_inputs(
    inputs: Sequence[str],
    Cx: int,
    metadata_inputs: Optional[Sequence[str]] = None,
    *,
    dataset_label: str = "TFRecord dataset",
) -> None:
    expected_inputs = tuple(str(x) for x in inputs)

    if Cx != len(expected_inputs):
        raise ValueError(
            f"{dataset_label} has C={Cx} input channels, but cfg.processes.iceflow.unified.inputs "
            f"has {len(expected_inputs)} entries: {expected_inputs}. These must match in count and order."
        )

    if metadata_inputs is None:
        return

    actual_inputs = tuple(str(x) for x in metadata_inputs)
    if actual_inputs != expected_inputs:
        raise ValueError(
            f"{dataset_label} declares input_names={actual_inputs}, but "
            f"cfg.processes.iceflow.unified.inputs={expected_inputs}. These must match exactly in count and order."
        )


def build_velocity_data_loss(
    *,
    loss_type: str = "huber",
    huber_delta: float = 50.0,
):
    loss_name = str(loss_type).lower()
    if loss_name not in ("mse", "huber"):
        raise ValueError(f"loss_type must be 'mse' or 'huber', got {loss_type!r}")

    if loss_name == "huber":
        huber = tf.keras.losses.Huber(
            delta=float(huber_delta),
            reduction=tf.keras.losses.Reduction.NONE,
        )

        @tf.function(reduce_retracing=True, jit_compile=False)
        def velocity_data_loss(U: tf.Tensor, V: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
            Ut = tf.cast(y_batch[..., 0], U.dtype)
            Vt = tf.cast(y_batch[..., 1], V.dtype)
            return tf.reduce_mean(huber(Ut, U) + huber(Vt, V))

        return velocity_data_loss

    @tf.function(reduce_retracing=True, jit_compile=False)
    def velocity_data_loss(U: tf.Tensor, V: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        Ut = tf.cast(y_batch[..., 0], U.dtype)
        Vt = tf.cast(y_batch[..., 1], V.dtype)
        return tf.reduce_mean(tf.square(U - Ut) + tf.square(V - Vt))

    return velocity_data_loss


def build_tfrecord_datasets_for_nz(
    tfrecord_root: str | Path,
    *,
    nz: int,
    inputs: Sequence[str],
    batch_size: int,
    compression: str = "GZIP",
    shuffle_buffer: int = 2048,
    split_seed: int = 0,
    val_seed: int = 1234,
) -> TFRecordDatasets:
    tfrecord_root = Path(tfrecord_root)
    metadata = load_metadata(tfrecord_root)

    if "example_shapes_by_nz" not in metadata or str(nz) not in metadata["example_shapes_by_nz"]:
        raise ValueError(f"TFRecord metadata at {tfrecord_root} has no entry for Nz={nz}.")

    shapes = metadata["example_shapes_by_nz"][str(nz)]
    H, W, Cx = shapes["x"]

    validate_dataset_matches_inputs(
        inputs=inputs,
        Cx=Cx,
        metadata_inputs=metadata.get("x_channel_names"),
        dataset_label=f"TFRecord dataset at {tfrecord_root}",
    )

    train_files = list_shards(tfrecord_root, nz, split="train")
    val_files = list_shards(tfrecord_root, nz, split="val")

    rng = random.Random(int(split_seed))
    rng.shuffle(train_files)

    train_ds, val_ds = make_datasets(
        train_files=train_files,
        val_files=val_files,
        H=H,
        W=W,
        Nz=nz,
        compression=compression,
        batch_size=int(batch_size),
        Cx=Cx,
        shuffle_buffer=int(shuffle_buffer),
        val_seed=int(val_seed),
    )

    return TFRecordDatasets(
        metadata=metadata,
        H=H,
        W=W,
        Cx=Cx,
        train_ds=train_ds,
        val_ds=val_ds,
    )

def _anchor_loss(current_vars, ref_vars, eps: float = 1e-12):
    if not current_vars:
        return tf.constant(0.0, dtype=tf.float32)

    dtype = current_vars[0].dtype
    diff_vals = [
        tf.reduce_mean(tf.square(tf.cast(v, dtype) - tf.cast(v0, dtype)))
        for v, v0 in zip(current_vars, ref_vars)
    ]
    ref_vals = [
        tf.reduce_mean(tf.square(tf.cast(v0, dtype)))
        for v0 in ref_vars
    ]

    diff_mean = tf.add_n(diff_vals) / tf.cast(len(diff_vals), dtype)
    ref_mean = tf.add_n(ref_vals) / tf.cast(len(ref_vals), dtype)
    return diff_mean / (ref_mean + tf.cast(eps, dtype))