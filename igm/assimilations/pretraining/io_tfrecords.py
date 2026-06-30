#!/usr/bin/env python3
# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Optional
import tensorflow as tf

def load_metadata(tfrecord_root: Path) -> dict:
    meta_path = tfrecord_root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json at {meta_path}")
    return json.loads(meta_path.read_text())

def list_shards(tfrecord_root: Path, nz: int, split: str) -> List[str]:
    shard_dir = tfrecord_root / split / f"nz{nz}"
    if not shard_dir.exists():
        raise FileNotFoundError(f"Missing Nz folder for split '{split}': {shard_dir}")

    # be robust to filename prefix changes
    files = sorted(str(p) for p in shard_dir.glob("*.tfrecord"))
    if not files:
        raise FileNotFoundError(f"No TFRecord shards found in {shard_dir}")
    return files

def parse_example(
    serialized: tf.Tensor,
    H: int,
    W: int,
    Nz: int,
    Cx: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    feat = {
        "seed": tf.io.FixedLenFeature([], tf.int64),
        "t": tf.io.FixedLenFeature([], tf.float32),
        "nz": tf.io.FixedLenFeature([], tf.int64),
        "x": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(serialized, feat)

    x = tf.io.parse_tensor(ex["x"], out_type=tf.float32)
    y = tf.io.parse_tensor(ex["y"], out_type=tf.float32)

    x = tf.ensure_shape(x, [H, W, Cx])
    y = tf.ensure_shape(y, [Nz, H, W, 2])  # (U,V)
    return x, y


def make_datasets(
    train_files: List[str],
    val_files: List[str],
    H: int,
    W: int,
    Nz: int,
    compression: str,
    batch_size: int,
    Cx: int,
    shuffle_buffer: int = 2048,
    val_seed: int = 1234,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    files = tf.data.Dataset.from_tensor_slices(train_files)
    files = files.shuffle(len(train_files), reshuffle_each_iteration=True).repeat()

    train_ds = files.interleave(
        lambda f: tf.data.TFRecordDataset(f, compression_type=compression),
        cycle_length=16,
        block_length=4,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    train_ds = train_ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    train_ds = train_ds.map(lambda s: parse_example(s, H, W, Nz, Cx),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    # --- VAL (finite each epoch): shuffle (seeded) -> parse -> batch -> prefetch
    val_ds = tf.data.TFRecordDataset(
        val_files,
        compression_type=compression,
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.shuffle(
        buffer_size=shuffle_buffer,
        seed=val_seed,
        reshuffle_each_iteration=True,  # new order each epoch if you recreate the iterator
    )
    val_ds = val_ds.map(
        lambda s: parse_example(s, H, W, Nz, Cx),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.batch(batch_size, drop_remainder=True)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

