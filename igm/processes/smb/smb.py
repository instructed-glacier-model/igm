#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import importlib


def _impl(cfg):
    method = cfg.processes.smb.method.lower()
    return importlib.import_module(f"igm.processes.smb.{method}")


def initialize(cfg, state):
    _impl(cfg).initialize(cfg, state)


def update(cfg, state):
    _impl(cfg).update(cfg, state)


def finalize(cfg, state):
    _impl(cfg).finalize(cfg, state)
