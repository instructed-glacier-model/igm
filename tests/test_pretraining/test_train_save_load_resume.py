"""
Integration test for the pretraining module.

Exercises the full lifecycle:
  1. Train a lightweight CNN for one epoch via igm.igm_run.main()
  2. Verify checkpoint and emulator.keras were written
  3. Load the saved emulator artifact and verify its properties
  4. Restore from the checkpoint and verify the start epoch

The training itself runs through the real igm.igm_run.main() stack so that
all glue code in pretraining.initialize (normalizer adaptation, Trainer
construction, etc.) is exercised. iceflow is included in the processes list
because pretraining reads its config; iceflow.initialize detects the
pretraining key and skips glacier-field setup (iceflow.py:65-67).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

# ---------------------------------------------------------------------------
# TFRecord fixture helper
# ---------------------------------------------------------------------------

def _write_fake_tfrecords(
    root: Path,
    H: int,
    W: int,
    Nz: int,
    Cx: int,
    n_train: int,
    n_val: int,
    input_names: list[str],
) -> None:
    """Write minimal TFRecord data compatible with build_tfrecord_datasets_for_nz."""
    rng = np.random.default_rng(42)

    metadata = {
        "example_shapes_by_nz": {
            str(Nz): {"x": [H, W, Cx], "y": [Nz, H, W, 2]}
        },
        "x_channel_names": input_names,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata))

    def _make_example(seed: int) -> tf.train.Example:
        x = rng.random((H, W, Cx), dtype=np.float32)
        y = rng.random((Nz, H, W, 2), dtype=np.float32)
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "seed": tf.train.Feature(int64_list=tf.train.Int64List(value=[seed])),
                    "t": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0])),
                    "nz": tf.train.Feature(int64_list=tf.train.Int64List(value=[Nz])),
                    "x": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.constant(x)).numpy()]
                        )
                    ),
                    "y": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(tf.constant(y)).numpy()]
                        )
                    ),
                }
            )
        )

    opts = tf.io.TFRecordOptions(compression_type="GZIP")

    for split, n in [("train", n_train), ("val", n_val)]:
        shard_dir = root / split / f"nz{Nz}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        with tf.io.TFRecordWriter(str(shard_dir / "shard.tfrecord"), opts) as writer:
            for i in range(n):
                writer.write(_make_example(i).SerializeToString())


# ---------------------------------------------------------------------------
# Constants shared across test phases
# ---------------------------------------------------------------------------

_H, _W, _NZ, _CX = 8, 8, 2, 3
_INPUT_NAMES = ["thk", "usurf", "tau_ref"]
_EXPERIMENT_NAME = "test_pretraining_run"


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_train_save_load_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full pretraining lifecycle: train → verify files → load → resume."""

    # ------------------------------------------------------------------ #
    # Phase 1: run a real IGM job with pretraining                        #
    # ------------------------------------------------------------------ #
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = str(tmp_path / "data")
    out_dir = str(tmp_path / "out")

    _write_fake_tfrecords(
        root=Path(data_dir),
        H=_H, W=_W, Nz=_NZ, Cx=_CX,
        n_train=8, n_val=8,
        input_names=_INPUT_NAMES,
    )

    # _write_fake_tfrecords ran TF ops (tf.io.serialize_tensor) which
    # initialized the TF eager context. igm_run.main() then tries to call
    # tf.config.experimental.set_memory_growth on the already-initialized
    # context, which raises RuntimeError. Patching both list_physical_devices
    # and list_logical_devices to return no GPUs causes main() to skip all
    # GPU setup code and use the default (CPU/single-device) strategy.
    _orig_lpd = tf.config.list_physical_devices
    _orig_lld = tf.config.list_logical_devices
    monkeypatch.setattr(
        tf.config,
        "list_physical_devices",
        lambda device_type=None: (
            [] if device_type == "GPU" else _orig_lpd(device_type)
        ),
    )
    monkeypatch.setattr(
        tf.config,
        "list_logical_devices",
        lambda device_type=None: (
            [] if device_type == "GPU" else _orig_lld(device_type)
        ),
    )

    # Hydra's searchpath is set to cwd (see igm/conf/config.yaml) so it can
    # locate experiment/params.yaml from the test directory.
    monkeypatch.chdir(test_dir)
    monkeypatch.setattr(sys, "argv", [
        "igm_run.py",
        "+experiment=params",
        f"assimilations.pretraining.data_dir={data_dir}",
        f"assimilations.pretraining.out_dir={out_dir}",
        f"hydra.run.dir={str(tmp_path / 'hydra_output')}",
    ])

    from igm.igm_run import main
    main()

    # ------------------------------------------------------------------ #
    # Phase 2: verify that training wrote the expected files              #
    # ------------------------------------------------------------------ #
    artifact_dir = Path(out_dir) / _EXPERIMENT_NAME

    assert (artifact_dir / "emulator.keras").exists(), \
        "emulator.keras not found — pretraining did not save the artifact"
    assert (artifact_dir / "history.yaml").exists(), \
        "history.yaml not found — training loop did not complete"
    assert any((artifact_dir / "checkpoints").glob("ckpt-*")), \
        "No checkpoint files found — Trainer did not save a checkpoint"

    # ------------------------------------------------------------------ #
    # Phase 3: load the saved emulator artifact and verify its properties #
    # ------------------------------------------------------------------ #
    from omegaconf import OmegaConf

    from igm.processes.iceflow.emulate.utils.artifacts import (
        EmulatorArtifact,
        load_emulator_artifact,
    )

    # Mirror the cfg used during training (iceflow.yaml defaults + params.yaml overrides).
    _load_cfg = OmegaConf.create({
        "processes": {"iceflow": {
            "physics": {"sliding": {"u_ref": 100.0}},
            "numerics": {"Nz": _NZ, "basis_vertical": "Lagrange", "basis_horizontal": "central"},
            "unified": {"inputs": list(_INPUT_NAMES)},
        }}
    })
    loaded = load_emulator_artifact(
        artifact_dir, cfg=_load_cfg, expected_inputs=list(_INPUT_NAMES)
    )

    assert isinstance(loaded, EmulatorArtifact), \
        f"Expected EmulatorArtifact, got {type(loaded)}"
    assert loaded.Nz == _NZ, \
        f"Loaded model has Nz={loaded.Nz}, expected {_NZ}"
    assert loaded.nb_inputs == _CX, \
        f"Loaded model has nb_inputs={loaded.nb_inputs}, expected {_CX}"
    assert loaded.input_names == _INPUT_NAMES, \
        f"Loaded model input_names={loaded.input_names}, expected {_INPUT_NAMES}"

    # Quick forward-pass smoke check: output shape should be [B, H, W, 2*Nz]
    dummy_x = tf.zeros((1, _H, _W, _CX), dtype=tf.float32)
    output = loaded(dummy_x, training=False)
    expected_output_shape = (1, _H, _W, 2 * _NZ)
    assert tuple(output.shape) == expected_output_shape, \
        f"Forward pass shape {tuple(output.shape)} != expected {expected_output_shape}"

    # ------------------------------------------------------------------ #
    # Phase 4: restore from checkpoint and verify start epoch             #
    # ------------------------------------------------------------------ #
    from igm.processes.iceflow.unified.mappings.network import MappingNetwork
    from igm.assimilations.pretraining.trainer import Trainer
    from igm.assimilations.pretraining.training_utils import build_tfrecord_datasets_for_nz

    cfg = OmegaConf.create({
        "assimilations": {
            "pretraining": {
                "steps_per_epoch": 2,
                "val_steps": 2,
                "lambda_ema": 0.99,
                "lambda_min": 0.001,
                "lambda_max": 100.0,
                "lambda_max_change": 2.0,
                "lambda_update_every": 100,
                "warmup_steps": 10000,
                "learning_rate": 1e-4,
                "loss_type": "mse",
                "huber_delta": 50.0,
            }
        }
    })

    datasets = build_tfrecord_datasets_for_nz(
        Path(data_dir),
        nz=_NZ,
        inputs=_INPUT_NAMES,
        batch_size=2,
        compression="GZIP",
    )

    mapping = MappingNetwork(
        bcs=[],
        network=loaded,
        Nz=_NZ,
        output_scale=1.0,
        precision="float32",
    )

    ckpt_dir = artifact_dir / "checkpoints"
    resume_trainer = Trainer(
        cfg=cfg,
        model=loaded,
        mapping=mapping,
        physics_cost_fn=lambda U, V, x: tf.zeros((), dtype=U.dtype),
        train_ds=datasets.train_ds,
        val_ds=datasets.val_ds,
        out_dir=artifact_dir,
        ckpt_dir=ckpt_dir,
        fig_dir=artifact_dir / "figures",
        n_epochs=2,
        accum_steps=1,
        Nz=_NZ,
        save_model=True,
        make_plots=False,
    )

    start_epoch = resume_trainer.restore_from_checkpoint()
    assert start_epoch == 1, \
        f"restore_from_checkpoint() returned {start_epoch}, expected 1"
