#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Checkpoint / restart of an IGM run.

A checkpoint is a snapshot of the whole ``state`` object taken at the end of a
time iteration (after the output modules ran). It is written in the Hydra run
directory as ``<folder>/state.npz`` (all arrays) plus ``<folder>/manifest.json``
(the description of every entry and the run bookkeeping). Re-running the very
same ``igm_run`` command in the same run directory resumes the run from that
snapshot; the user gets the same results as if the run had never been stopped.

Entries are classified by their Python type, which keeps the mechanism
independent of the modules: anything a module stores on ``state`` is saved,
provided it is one of

* ``tf.Variable`` / eager ``tf.Tensor``        (fields, scalars, counters)
* ``numpy.ndarray`` / numpy scalar
* Python ``int``/``float``/``bool``/``str``/``None`` and lists/tuples of them
  (lists of NumPy scalars keep their dtype)
* ``dict`` of the above (e.g. ``state.particle``)
* ``tf.keras.Model``                             (weights)
* ``tf.keras.optimizers.Optimizer``              (slots, iteration counter)
* plain objects (``state.iceflow``): their attributes are walked recursively
  for the types above, so ``state.iceflow.optimizer.optim_adam`` is covered.

Everything else (loggers, matplotlib figures, xarray datasets, functions,
modules, ...) is recreated by the modules' ``initialize`` on restart and is
listed under ``skipped`` in the manifest so the user can check nothing
important was left out.

A graceful stop is requested either by a signal (``SIGUSR1``/``SIGTERM`` by
default, e.g. sent by the cluster scheduler before the wall-time limit) or by
creating a file named ``STOP`` in the run directory. The current iteration is
finished, the checkpoint written, and the process exits.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf

MANIFEST_VERSION = 1

# Bookkeeping attributes that describe *this* process rather than the run.
_NEVER_SAVE = {
    "logger",
    "pbar",
    "original_cwd",
    "start_time",
    "end_time",
    "runtime",
    "tcomp",
    "ckpt_stop_requested",
    "ckpt_restarted",
    "ckpt_restarting",
    "ckpt_last_wall",
    "ckpt_n_saves",
    "ckpt_stopped",
}

_PY_SCALARS = (int, float, bool, str, type(None))
_MAX_DEPTH = 4


# ─── classification ─────────────────────────────────────────────────────────


def _is_keras_model(obj) -> bool:
    return isinstance(obj, tf.keras.Model)


def _is_keras_optimizer(obj) -> bool:
    opt_types = tuple(
        t
        for t in (
            getattr(tf.keras.optimizers, "Optimizer", None),
            getattr(getattr(tf.keras.optimizers, "legacy", None), "Optimizer", None),
        )
        if t is not None
    )
    return isinstance(obj, opt_types)


def _is_py_scalar_list(obj) -> bool:
    if not isinstance(obj, (list, tuple)):
        return False
    return all(isinstance(v, _PY_SCALARS + (np.generic,)) for v in obj)


def _to_py(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def _collect(path: str, obj: Any, arrays: dict, entries: list, skipped: list,
             seen: set, depth: int) -> None:
    """Classify ``obj`` found at ``path`` and record it."""
    if depth > _MAX_DEPTH:
        skipped.append((path, "too deep"))
        return

    if isinstance(obj, tf.Variable):
        key = f"{len(arrays):05d}"
        arrays[key] = obj.numpy()
        entries.append(
            {"path": path, "kind": "variable", "key": key,
             "trainable": bool(obj.trainable), "dtype": obj.dtype.name}
        )
    elif isinstance(obj, tf.Tensor):
        if not hasattr(obj, "numpy"):  # symbolic tensor: nothing to save
            skipped.append((path, "symbolic tensor"))
            return
        key = f"{len(arrays):05d}"
        arrays[key] = obj.numpy()
        entries.append({"path": path, "kind": "tensor", "key": key,
                        "dtype": obj.dtype.name})
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            skipped.append((path, "object ndarray"))
            return
        key = f"{len(arrays):05d}"
        arrays[key] = obj
        entries.append({"path": path, "kind": "ndarray", "key": key})
    elif isinstance(obj, np.generic):
        entries.append({"path": path, "kind": "py", "value": obj.item(),
                        "np_dtype": obj.dtype.name})
    elif isinstance(obj, _PY_SCALARS):
        entries.append({"path": path, "kind": "py", "value": obj})
    elif _is_py_scalar_list(obj):
        np_dtypes = {v.dtype for v in obj if isinstance(v, np.generic)}
        if obj and len(np_dtypes) == 1 and all(isinstance(v, np.generic) for v in obj):
            # list of NumPy scalars (e.g. np.float32 statistics): keep the dtype,
            # a float64 round trip would change later reductions in the last bits
            key = f"{len(arrays):05d}"
            arrays[key] = np.array(obj)
            entries.append({"path": path, "kind": "nplist", "key": key,
                            "tuple": isinstance(obj, tuple)})
        else:
            entries.append({"path": path, "kind": "py",
                            "value": [_to_py(v) for v in obj],
                            "tuple": isinstance(obj, tuple)})
    elif _is_keras_model(obj):
        if id(obj) in seen:
            entries.append({"path": path, "kind": "alias"})
            return
        seen.add(id(obj))
        keys = []
        for w in obj.get_weights():
            key = f"{len(arrays):05d}"
            arrays[key] = w
            keys.append(key)
        entries.append({"path": path, "kind": "keras_model", "keys": keys})
    elif _is_keras_optimizer(obj):
        if id(obj) in seen:
            entries.append({"path": path, "kind": "alias"})
            return
        seen.add(id(obj))
        keys = []
        for v in obj.variables:
            key = f"{len(arrays):05d}"
            arrays[key] = v.numpy()
            keys.append(key)
        # shapes of the variables the optimizer was built on (Keras 3): lets the
        # restore rebuild the slot variables of a not-yet-built fresh optimizer
        built_on = getattr(obj, "_trainable_variables", None) or []
        entries.append({"path": path, "kind": "keras_optimizer", "keys": keys,
                        "var_shapes": [list(v.shape) for v in built_on]})
    elif isinstance(obj, dict):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if not all(isinstance(k, str) for k in obj):
            skipped.append((path, "dict with non-string keys"))
            return
        entries.append({"path": path, "kind": "dict"})
        for k, v in obj.items():
            _collect(f"{path}.{k}", v, arrays, entries, skipped, seen, depth + 1)
    elif callable(obj) or isinstance(obj, type):
        skipped.append((path, type(obj).__name__))
    elif isinstance(obj, tf.experimental.ExtensionType):
        # Immutable parameter bundles rebuilt from the configuration.
        skipped.append((path, "ExtensionType"))
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("igm"):
        # Plain IGM container (state.iceflow, unified optimizer, mapping, ...):
        # walk its attributes for the types above.
        if id(obj) in seen:
            return
        seen.add(id(obj))
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            _collect(f"{path}.{k}", v, arrays, entries, skipped, seen, depth + 1)
    else:
        skipped.append((path, type(obj).__name__))


def collect_state(state) -> tuple[dict, list, list]:
    """Return (arrays, entries, skipped) describing the whole ``state``."""
    arrays: dict = {}
    entries: list = []
    skipped: list = []
    seen: set = set()
    for name, value in vars(state).items():
        if name in _NEVER_SAVE or name.startswith("__"):
            continue
        _collect(name, value, arrays, entries, skipped, seen, depth=0)
    return arrays, entries, skipped


# ─── paths / helpers ────────────────────────────────────────────────────────


def _cfg(cfg):
    return cfg.core.checkpoint


def checkpoint_dir(cfg) -> Path:
    # Hydra already changed into the run directory.
    return Path(os.getcwd()) / _cfg(cfg).folder


def checkpoint_exists(cfg) -> bool:
    d = checkpoint_dir(cfg)
    return (d / "manifest.json").exists() and (d / "state.npz").exists()


def _log(state, msg: str, level=logging.INFO) -> None:
    if hasattr(state, "logger"):
        state.logger.log(level, msg)
    print(f"[checkpoint] {msg}")


def _resolve(root, path: str):
    """Return (parent, last_key, exists) for a dotted path on ``root``."""
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        if isinstance(obj, dict):
            if p not in obj:
                return None, parts[-1], False
            obj = obj[p]
        else:
            if not hasattr(obj, p):
                return None, parts[-1], False
            obj = getattr(obj, p)
        if obj is None:
            # container not created yet on the fresh state (e.g. set by the first
            # update): exists=None tells the caller to skip silently
            return None, parts[-1], None
    last = parts[-1]
    if isinstance(obj, dict):
        return obj, last, last in obj
    return obj, last, hasattr(obj, last)


def _find_trainable_variables(state, manifest, shapes):
    """Trainable variables of the checkpointed Keras model whose shapes match."""
    for e in manifest["entries"]:
        if e["kind"] != "keras_model":
            continue
        parent, key, exists = _resolve(state, e["path"])
        if parent is None or not exists:
            continue
        model = _get(parent, key)
        tv = list(getattr(model, "trainable_variables", []))
        if [list(v.shape) for v in tv] == [list(x) for x in shapes]:
            return tv
    return None


def _get(parent, key):
    return parent[key] if isinstance(parent, dict) else getattr(parent, key)


def _set(parent, key, value):
    if isinstance(parent, dict):
        parent[key] = value
    else:
        setattr(parent, key, value)


# ─── save ───────────────────────────────────────────────────────────────────


def save_checkpoint(cfg, state, reason: str = "periodic") -> Path:
    """Write the checkpoint atomically and return its directory."""
    t0 = time.time()
    d = checkpoint_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)

    arrays, entries, skipped = collect_state(state)
    token = uuid.uuid4().hex

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "token": token,
        "written": datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
        "it": int(getattr(state, "it", 0)),
        "t": float(state.t.numpy()) if hasattr(state, "t") else None,
        "n_saves": int(getattr(state, "ckpt_n_saves", 0)) + 1,
        "entries": entries,
        "skipped": [{"path": p, "type": t} for p, t in skipped],
        "config": OmegaConf.to_yaml(cfg, resolve=False),
    }

    # Arrays first, manifest last, each through a temporary file: a reader
    # accepts the pair only if both carry the same token.
    tmp_npz = d / "state.tmp.npz"
    with open(tmp_npz, "wb") as f:
        np.savez(f, __token__=np.array(token), **arrays)
    os.replace(tmp_npz, d / "state.npz")

    tmp_json = d / "manifest.tmp.json"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp_json, d / "manifest.json")

    state.ckpt_n_saves = manifest["n_saves"]
    state.ckpt_last_wall = time.time()
    size_mb = (d / "state.npz").stat().st_size / 2**20
    _log(
        state,
        f"saved ({reason}) at it={manifest['it']} t={manifest['t']}: "
        f"{len(entries)} entries, {size_mb:.1f} MB, {time.time() - t0:.2f} s "
        f"-> {d}",
    )
    return d


def remove_checkpoint(cfg, state) -> None:
    d = checkpoint_dir(cfg)
    for name in ("state.npz", "manifest.json"):
        try:
            (d / name).unlink()
        except FileNotFoundError:
            pass
    try:
        d.rmdir()
    except OSError:
        pass


# ─── restore ────────────────────────────────────────────────────────────────


def load_manifest(cfg) -> dict:
    d = checkpoint_dir(cfg)
    with open(d / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def restore_checkpoint(cfg, state) -> dict:
    """Overlay the checkpoint on an initialized ``state``.

    Must be called after the modules' ``initialize``: the modules have then
    created the containers (Keras models, optimizers, ``state.iceflow``) that
    the checkpoint fills in.
    """
    t0 = time.time()
    d = checkpoint_dir(cfg)
    manifest = load_manifest(cfg)
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise RuntimeError(
            f"Checkpoint manifest version {manifest.get('manifest_version')} "
            f"not supported (expected {MANIFEST_VERSION})."
        )
    data = np.load(d / "state.npz", allow_pickle=False)
    if str(data["__token__"]) != manifest["token"]:
        raise RuntimeError(
            f"Checkpoint in {d} is inconsistent (state.npz and manifest.json "
            "come from different writes). Remove the folder to start afresh."
        )

    if _cfg(cfg).check_config:
        current = OmegaConf.to_yaml(cfg, resolve=False)
        if current != manifest["config"]:
            _log(
                state,
                "configuration differs from the one saved in the checkpoint; "
                "resuming anyway (set core.checkpoint.check_config=false to "
                "silence this warning).",
                logging.WARNING,
            )

    problems = []
    n_applied = 0
    for e in manifest["entries"]:
        path, kind = e["path"], e["kind"]
        parent, key, exists = _resolve(state, path)
        if parent is None:
            if exists is not None:
                problems.append(f"{path}: container missing on the fresh state")
            continue

        if kind == "variable":
            arr = data[e["key"]]
            cur = _get(parent, key) if exists else None
            if (
                isinstance(cur, tf.Variable)
                and cur.shape == arr.shape
                and cur.dtype.name == e["dtype"]
            ):
                cur.assign(arr)
            else:
                _set(parent, key, tf.Variable(arr, trainable=e["trainable"],
                                              dtype=e["dtype"]))
            n_applied += 1
        elif kind == "tensor":
            _set(parent, key, tf.constant(data[e["key"]], dtype=e["dtype"]))
            n_applied += 1
        elif kind == "ndarray":
            _set(parent, key, np.array(data[e["key"]]))
            n_applied += 1
        elif kind == "nplist":
            seq = list(data[e["key"]])  # NumPy scalars of the saved dtype
            _set(parent, key, tuple(seq) if e.get("tuple") else seq)
            n_applied += 1
        elif kind == "py":
            v = e["value"]
            if "np_dtype" in e:
                v = np.dtype(e["np_dtype"]).type(v)
            elif e.get("tuple"):
                v = tuple(v)
            try:
                _set(parent, key, v)
                n_applied += 1
            except AttributeError as err:
                problems.append(f"{path}: {err}")
        elif kind == "dict":
            if not exists:
                _set(parent, key, {})
        elif kind == "keras_model":
            if not exists:
                problems.append(f"{path}: Keras model not created by initialize")
                continue
            model = _get(parent, key)
            weights = [data[k] for k in e["keys"]]
            try:
                model.set_weights(weights)
                n_applied += 1
            except ValueError as err:
                problems.append(f"{path}: cannot set weights ({err})")
        elif kind == "keras_optimizer":
            if not exists:
                problems.append(f"{path}: optimizer not created by initialize")
                continue
            opt = _get(parent, key)
            values = [data[k] for k in e["keys"]]
            if (
                not getattr(opt, "built", True)
                and len(opt.variables) < len(values)
                and e.get("var_shapes")
            ):
                # fresh optimizer not built yet (Keras builds the slot variables at
                # the first apply_gradients): build it on the model it was trained on
                target = _find_trainable_variables(state, manifest, e["var_shapes"])
                if target is not None:
                    try:
                        opt.build(target)
                    except Exception as err:  # pragma: no cover
                        problems.append(f"{path}: cannot build optimizer ({err})")
            if len(opt.variables) != len(values):
                if len(values) == 0:
                    continue  # was not built when saved either
                problems.append(
                    f"{path}: optimizer has {len(opt.variables)} variables, "
                    f"checkpoint has {len(values)} (not restored)"
                )
                continue
            for v, arr in zip(opt.variables, values):
                v.assign(arr)
            n_applied += 1
        elif kind == "alias":
            pass
        else:
            problems.append(f"{path}: unknown kind {kind}")

    data.close()
    state.ckpt_restarted = True
    state.ckpt_n_saves = manifest.get("n_saves", 0)
    state.ckpt_last_wall = time.time()
    _log(
        state,
        f"restored it={manifest['it']} t={manifest['t']} "
        f"(written {manifest['written']}, {manifest['reason']}): "
        f"{n_applied} entries in {time.time() - t0:.2f} s",
    )
    for p in problems:
        _log(state, "restore: " + p, logging.WARNING)
    return manifest


# ─── graceful stop ──────────────────────────────────────────────────────────


def install_signal_handlers(cfg, state) -> None:
    state.ckpt_stop_requested = False

    def _handler(signum, frame):
        name = signal.Signals(signum).name
        state.ckpt_stop_requested = True
        print(
            f"\n[checkpoint] {name} received: finishing the current iteration, "
            "writing a checkpoint, then exiting."
        )

    for name in _cfg(cfg).signals:
        sig = getattr(signal, str(name), None)
        if sig is None:
            _log(state, f"unknown signal {name!r} ignored", logging.WARNING)
            continue
        signal.signal(sig, _handler)


def stop_requested(cfg, state) -> bool:
    if getattr(state, "ckpt_stop_requested", False):
        return True
    stop_file = _cfg(cfg).stop_file
    if stop_file and Path(os.getcwd()).joinpath(stop_file).exists():
        state.ckpt_stop_requested = True
        _log(state, f"stop file {stop_file!r} found: stopping gracefully.")
        return True
    return False


def _periodic_due(cfg, state) -> bool:
    c = _cfg(cfg)
    it = int(getattr(state, "it", 0))
    if c.every_iter > 0 and it % c.every_iter == 0:
        return True
    if c.every_save > 0 and getattr(state, "saveresult", False):
        n = int(getattr(state, "itsave", -1))
        if n >= 0 and n % c.every_save == 0:
            return True
    if c.every_seconds > 0:
        last = getattr(state, "ckpt_last_wall", None)
        if last is None:
            state.ckpt_last_wall = time.time()
        elif time.time() - last >= c.every_seconds:
            return True
    return False


def end_of_iteration(cfg, state) -> None:
    """Called by the runner at the end of every iteration.

    Writes a periodic checkpoint when due, and on a stop request writes the
    checkpoint and ends the loop (``state.ckpt_stopped`` is then set so the
    caller skips ``finalize`` and exits with ``core.checkpoint.exit_code``).
    """
    # Test harnesses may build a partial `cfg` without the core.checkpoint block.
    if "checkpoint" not in cfg.core or not _cfg(cfg).enabled:
        return
    if stop_requested(cfg, state):
        save_checkpoint(cfg, state, reason="stop")
        state.ckpt_stopped = True
        state.continue_run = False
        return
    if state.continue_run and _periodic_due(cfg, state):
        save_checkpoint(cfg, state, reason="periodic")
