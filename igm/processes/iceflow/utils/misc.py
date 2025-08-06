#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 
from tqdm import tqdm
import datetime
from omegaconf import DictConfig

class EarlyStopping:
    def __init__(self, relative_min_delta=1e-3, patience=10):
        """
        Args:
            relative_min_delta (float): Minimum relative improvement required.
            patience (int): Number of consecutive iterations with no significant improvement allowed.
        """
        self.relative_min_delta = relative_min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = None

    def should_stop(self, current_loss):
        if self.best_loss is None:
            # Initialize best_loss during the first call
            self.best_loss = current_loss
            return False
        
        # Compute relative improvement
        relative_improvement = (self.best_loss - current_loss) / abs(self.best_loss)

        if relative_improvement > self.relative_min_delta:
            # Significant improvement: update best_loss and reset wait
            self.best_loss = current_loss
            self.wait = 0
            return False
        else:
            # No significant improvement: increment wait
            self.wait += 1
            if self.wait >= self.patience:
                return True
            
def initialize_iceflow_fields(cfg, state):

    # here we initialize variable parmaetrizing ice flow
    if not hasattr(state, "arrhenius"):
        if cfg.processes.iceflow.physics.dim_arrhenius == 3:
            state.arrhenius = \
                tf.ones((cfg.processes.iceflow.numerics.Nz, state.thk.shape[0], state.thk.shape[1])) \
                * cfg.processes.iceflow.physics.init_arrhenius * cfg.processes.iceflow.physics.enhancement_factor
        else:
            state.arrhenius = tf.ones_like(state.thk) * cfg.processes.iceflow.physics.init_arrhenius * cfg.processes.iceflow.physics.enhancement_factor

    if not hasattr(state, "slidingco"):
        state.slidingco = tf.ones_like(state.thk) * cfg.processes.iceflow.physics.init_slidingco

    # here we create a new velocity field
    if not hasattr(state, "U"):
        state.U = tf.zeros((cfg.processes.iceflow.numerics.Nz, state.thk.shape[0], state.thk.shape[1])) 
        state.V = tf.zeros((cfg.processes.iceflow.numerics.Nz, state.thk.shape[0], state.thk.shape[1])) 
            

def print_info(state, it, cfg, energy_mean_list, velsurf_mag):
 
    if it % 100 == 1:
        if hasattr(state, "pbar_train"):
            state.pbar_train.close()
        state.pbar_train = tqdm(desc=f" Phys assim.", ascii=False, dynamic_ncols=True, bar_format="{desc} {postfix}")

    if hasattr(state, "pbar_train"):
        dic_postfix = {}
        dic_postfix["🕒"] = datetime.datetime.now().strftime("%H:%M:%S")
        dic_postfix["🔄"] = f"{it:04.0f}"
        for i, f in enumerate(cfg.processes.iceflow.physics.energy_components):
            dic_postfix[f] = f"{energy_mean_list[i]:06.3f}"
        dic_postfix["glen"] = f"{np.sum(energy_mean_list):06.3f}"
        dic_postfix["Max vel"] = f"{velsurf_mag:06.1f}"
#       dic_postfix["💾 GPU Mem (MB)"] = tf.config.experimental.get_memory_info("GPU:0")['current'] / 1024**2

        state.pbar_train.set_postfix(dic_postfix)
        state.pbar_train.update(1)

def is_retrain(iteration, cfg: DictConfig) -> bool:

    # run_it = False
    if cfg.processes.iceflow.emulator.retrain_freq > 0:
        run_it = iteration % cfg.processes.iceflow.emulator.retrain_freq == 0

    warm_up = int(iteration <= cfg.processes.iceflow.emulator.warm_up_it)
    
    return run_it or warm_up