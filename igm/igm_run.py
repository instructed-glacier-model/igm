#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file
import os

import tensorflow as tf

from igm.common import (
    State,
    initialize_modules,
    update_modules,
    finalize_modules,
    setup_igm_modules,
    print_gpu_info,
    add_logger,
    write_igm_version,
    download_unzip_and_store,
    print_comp,
    check_incompatilities_in_parameters_file,
)

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd  # , to_absolute_path
import hydra
import json

OmegaConf.register_new_resolver("get_cwd", lambda x: os.getcwd())
from datetime import datetime


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    state = State()  # class acting as a dictionary

    state.original_cwd = Path(get_original_cwd())
    state.saveresult = True
    state.start_time = datetime.now()

    write_igm_version(Path(os.getcwd()))
    
    if cfg.core.check_compat_params:
        check_incompatilities_in_parameters_file(cfg, state.original_cwd)

    if cfg.core.hardware.gpu_info:
        # print([gpus[i] for i in cfg.core.hardware.visible_gpus])
        print_gpu_info()

    gpus = tf.config.list_physical_devices("GPU")
    for gpu_instance in gpus:
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    if gpus:
        print([gpus[i] for i in cfg.core.hardware.visible_gpus])

        try:
            selected_visible_gpus = [gpus[i] for i in cfg.core.hardware.visible_gpus]
            tf.config.set_visible_devices(selected_visible_gpus, "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    if len(tf.config.list_logical_devices("GPU")) > 1:
        raise NotImplementedError(
            "Strategies for multiple GPUs are not yet implemented. Please make only one GPU visible."
        )
        # strategy = tf.distribute.MirroredStrategy()
    else:
        # if there is only one visible GPU, the id will be 0! Even when choosing a GPU that has index 4, it will only be 0 after configuring visible devices!
        # However, apply_gradients is having issues... so we have to update that first!
        # strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        strategy = tf.distribute.get_strategy()

    if cfg.core.logging:
        add_logger(cfg=cfg, state=state)
        tf.get_logger().setLevel(cfg.core.tf_logging_level)
    
    if cfg.core.print_params:
        print(OmegaConf.to_yaml(cfg))

    if not cfg.core.url_data == "":
        folder_path = state.original_cwd.joinpath(cfg.core.folder_data)
        download_unzip_and_store(cfg.core.url_data, folder_path)

    (
        imported_inputs_modules,
        imported_processes_modules,
        imported_outputs_modules,
    ) = setup_igm_modules(cfg, state)

    for input_method in imported_inputs_modules:
        input_method.run(cfg, state)

    for output_method in imported_outputs_modules:
        # TODO: would be cleaner to have inside setup_igm_modules...
        if not hasattr(output_method, "initialize"):
            raise ValueError(
                "Output methods must have an 'initialize' method defined (in addition to a 'run' method)."
            )
        output_method.initialize(cfg, state)

    with strategy.scope():
        initialize_modules(imported_processes_modules, cfg, state)
        update_modules(imported_processes_modules, imported_outputs_modules, cfg, state)
        finalize_modules(imported_processes_modules, cfg, state)

    if cfg.core.print_comp:
        print_comp(state)

    state.end_time = datetime.now()
    state.runtime = (state.end_time - state.start_time).total_seconds()

    if hasattr(state, "score"):
        score = state.score
        with open("_igm_score.json", "w") as f:
            json.dump(score, f)
    else:
        score = float("inf")

    return score


if __name__ == "__main__":
    main()
