#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import importlib
import tensorflow as tf


def _impl(cfg):
    method = cfg.processes.smb.method.lower()
    return importlib.import_module(f"igm.processes.smb.{method}")


def get_active_submodule(cfg):
    return _impl(cfg)


def initialize(cfg, state):
    _impl(cfg).initialize(cfg, state)
    if not hasattr(state, "smb"):
        state.smb = tf.zeros_like(state.usurf) if hasattr(state, "usurf") else None


def update(cfg, state):
    _impl(cfg).update(cfg, state)


def finalize(cfg, state):
    _impl(cfg).finalize(cfg, state)
