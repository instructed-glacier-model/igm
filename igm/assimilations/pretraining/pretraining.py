#!/usr/bin/env python3
# Copyright (C) 2021-2026 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Tuple

import tensorflow as tf

from igm.processes.iceflow.unified.mappings import Mappings, InterfaceMappings
from igm.processes.iceflow.unified.mappings.interfaces.network import (
    mapping_args_for_model,
)
from igm.processes.iceflow.emulate.utils.artifacts import (
    load_emulator_artifact,
    save_emulator_artifact,
    wrap_emulator_artifact,
)
from igm.assimilations.pretraining.cost_tmp import get_cost_fn
from igm.utils.math.precision import normalize_precision

from .trainer import Trainer
from .training_utils import build_tfrecord_datasets_for_nz


def update(cfg, state):
    pass


def finalize(cfg, state):
    pass


def _prepare_run_dirs(
    out_dir: Path, *, resume: bool, save_model: bool, make_plots: bool
) -> Tuple[Path, Path]:

    ckpt_dir = out_dir / "checkpoints"
    fig_dir = out_dir / "figures"

    if save_model:
        if resume:
            if not out_dir.exists():
                raise FileNotFoundError(
                    f"resume=True but experiment directory does not exist: {out_dir}"
                )
            if not ckpt_dir.exists():
                raise FileNotFoundError(
                    f"resume=True but checkpoints directory missing: {ckpt_dir}"
                )
        else:
            if ckpt_dir.exists() and any(ckpt_dir.glob("ckpt-*")):
                raise FileExistsError(
                    f"Experiment already has checkpoints at {ckpt_dir} but resume=False. "
                    "Set cfg.assimilations.pretraining.resume=true or use a new experiment_name."
                )
            ckpt_dir.mkdir(parents=True, exist_ok=True)

    if make_plots:
        fig_dir.mkdir(parents=True, exist_ok=True)

    return ckpt_dir, fig_dir


def _resolve_accum_steps(cfg_pretraining) -> Tuple[int, int, int]:
    """Validate batch_size / micro_batch_size and return (effective_bs, micro_bs, accum_steps)."""
    effective_bs = int(cfg_pretraining.batch_size)
    micro_bs = int(cfg_pretraining.micro_batch_size)

    if micro_bs <= 0 or effective_bs <= 0:
        raise ValueError(
            f"batch_size and micro_batch_size must be > 0, "
            f"got batch_size={effective_bs}, micro_batch_size={micro_bs}"
        )
    if micro_bs > effective_bs:
        raise ValueError(
            f"micro_batch_size ({micro_bs}) cannot exceed batch_size ({effective_bs})"
        )
    if effective_bs % micro_bs != 0:
        raise ValueError(
            f"batch_size ({effective_bs}) must be divisible by micro_batch_size "
            f"({micro_bs}) for clean accumulation"
        )
    return effective_bs, micro_bs, effective_bs // micro_bs


def initialize(cfg, state):
    tf.config.optimizer.set_jit(False)

    cfg_pretraining = cfg.assimilations.pretraining
    cfg_iceflow = cfg.processes.iceflow

    # pretraining is an /assimilations module, so it initialises *before* the
    # /processes (iceflow included). We write state.iceflow.mapping below, and
    # that namespace is created by iceflow.initialize, so initialise the
    # forward model here first. iceflow.initialize is idempotent (guarded by
    # state.iceflow_initialized), so the later /processes pass is a no-op.
    importlib.import_module("igm.processes.iceflow").initialize(cfg, state)

    make_plots = bool(cfg_pretraining.make_plots)
    save_model = bool(cfg_pretraining.save_model)
    resume = bool(cfg_pretraining.resume) if save_model else False

    out_dir = Path(cfg_pretraining.out_dir) / cfg_pretraining.experiment_name
    if make_plots or save_model:
        out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir, fig_dir = _prepare_run_dirs(
        out_dir, resume=resume, save_model=save_model, make_plots=make_plots
    )

    effective_bs, micro_bs, accum_steps = _resolve_accum_steps(cfg_pretraining)
    print(
        f"[grad-accum] effective_bs={effective_bs} "
        f"micro_bs={micro_bs} accum_steps={accum_steps}"
    )

    inputs = tuple(cfg_iceflow.unified.inputs)
    Nz = int(cfg_iceflow.numerics.Nz)
    datasets = build_tfrecord_datasets_for_nz(
        Path(cfg_pretraining.data_dir),
        nz=Nz,
        inputs=inputs,
        batch_size=micro_bs,
        compression="GZIP",
        split_seed=int(cfg_pretraining.split_seed),
    )
    train_ds, val_ds = datasets.train_ds, datasets.val_ds
    H, W, Cx = datasets.H, datasets.W, datasets.Cx
    desired_dtype = normalize_precision(cfg_iceflow.numerics.precision)
    dummy_x = tf.zeros((1, H, W, Cx), dtype=desired_dtype)

    if resume:
        state.iceflow_model = load_emulator_artifact(
            artifact_dir=out_dir, cfg=cfg, expected_inputs=inputs
        )
        mapping_args = mapping_args_for_model(cfg, state, state.iceflow_model)
        print("[resume] loaded emulator.keras; checkpoint will restore weights/optimizer state")
    else:
        mapping_args = InterfaceMappings["network"].get_mapping_args(cfg, state)
        state.iceflow_model = wrap_emulator_artifact(state.iceflow_model)
        state.iceflow_model.basis_vertical = str(cfg_iceflow.numerics.basis_vertical)
        state.iceflow_model.basis_horizontal = str(cfg_iceflow.numerics.basis_horizontal)
        state.iceflow_model.input_normalizer.adapt(
            train_ds.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE).take(2000)
        )
        state.iceflow_model.build(dummy_x.shape)
        mapping_args["network"] = state.iceflow_model
        if save_model:
            # First save: writes architecture + adapted normalizer to disk so
            # that resume runs can reload them before the checkpoint restore.
            path = save_emulator_artifact(artifact_dir=out_dir, model=state.iceflow_model)
            print(f"[artifact] wrote {path}")

    mapping = Mappings["network"](**mapping_args)
    state.iceflow.mapping = mapping

    trainer = Trainer(
        cfg=cfg,
        model=state.iceflow_model,
        mapping=mapping,
        physics_cost_fn=get_cost_fn(cfg, state),
        train_ds=train_ds,
        val_ds=val_ds,
        out_dir=out_dir,
        ckpt_dir=ckpt_dir,
        fig_dir=fig_dir,
        n_epochs=int(cfg_pretraining.epochs),
        accum_steps=accum_steps,
        Nz=Nz,
        save_model=save_model,
        make_plots=make_plots,
    )

    start_epoch = trainer.restore_from_checkpoint() if resume else 0
    trainer.run(start_epoch=start_epoch)

    if save_model:
        path = save_emulator_artifact(artifact_dir=out_dir, model=state.iceflow_model)
        print(f"[export] saved emulator artifact to {path}")

    state.score = trainer.final_score()
