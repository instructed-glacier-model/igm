import tensorflow as tf
import numpy as np
import json
import re
from pathlib import Path
from typing import Iterable, Dict, Any, Union, List
import os


def get_device():
    """Get the available device (GPU if available, else CPU)."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"You are working on GPU device: {gpus[0].name}")
        return '/GPU:0'
    else:
        print("You are working on CPU device")
        return '/CPU:0'

device = get_device()


def print_peak_gpu_memory():
    """Print peak GPU memory usage."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            peak_memory = memory_info.get('peak', 0) / (1024 ** 2)
            print(f"Peak GPU memory used: {peak_memory:.2f} MB.")
        except Exception as e:
            print(f"Could not get GPU memory info: {e}")
    else:
        print("No GPU available.")


def print_gpu_utilization():
    """Print current GPU memory utilization."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            current_memory = memory_info.get('current', 0) / (1024 ** 2)
            print(f"GPU memory occupied: {current_memory:.0f} MB.")
        except Exception as e:
            print(f"Could not get GPU memory info: {e}")
    else:
        print("No GPU available.")


def metrics_to_str(metrics):
    """Convert metrics dictionary to string representation."""
    out = []
    for year, vals in metrics.items():
        rmse = float(vals["rmse"].numpy() if hasattr(vals["rmse"], 'numpy') else vals["rmse"])
        mae = float(vals["mae"].numpy() if hasattr(vals["mae"], 'numpy') else vals["mae"])
        bias = float(vals["bias"].numpy() if hasattr(vals["bias"], 'numpy') else vals["bias"])
        std = float(vals["std"].numpy() if hasattr(vals["std"], 'numpy') else vals["std"])
        n = int(vals["area"].numpy() if hasattr(vals["area"], 'numpy') else vals["area"])
        out.append(f"{year}: rmse={rmse:.2f}, mae={mae:.2f}, bias={bias:.2f}, std={std:.2f}, area in Km^2={n}")
    return " | ".join(out)


ITER_RE = re.compile(
    r"""^
        Iter\s*(?P<iter>\d+):\s*
        loss=(?P<loss>[-\d.]+),\s*
        metrics\s*=\s*(?P<metrics>.+)
        $
    """,
    re.VERBOSE,
)


GROUP_RE = re.compile(
    r"""
        ^\s*
        (?P<id>\d+):\s*
        rmse=(?P<rmse>[-\d.]+),\s*
        mae=(?P<mae>[-\d.]+),\s*
        bias=(?P<bias>[-\d.]+),\s*
        std=(?P<std>[-\d.]+),\s*
        area\sin\sKm\^2=(?P<area>[-\d.]+)
        \s*$
    """,
    re.VERBOSE,
)


def iterations_to_jsonl(lines: Iterable[str], args):
    """Parse iteration logs and write to JSONL file."""
    out_path = Path(os.path.join(args.outdir, "training_metrics.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parsed: List[Dict[str, Any]] = []

    with out_path.open("w", encoding="utf-8") as f:
        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            m = ITER_RE.match(line)
            if not m:
                continue

            iter_num = int(m.group("iter"))
            loss_val = float(m.group("loss"))
            metrics_blob = m.group("metrics")

            groups = [g.strip() for g in metrics_blob.split("|") if g.strip()]

            metrics: Dict[str, Dict[str, float]] = {}
            for g in groups:
                gm = GROUP_RE.match(g)
                if not gm:
                    continue
                gid = gm.group("id")
                metrics[gid] = {
                    "rmse": float(gm.group("rmse")),
                    "mae": float(gm.group("mae")),
                    "bias": float(gm.group("bias")),
                    "std": float(gm.group("std")),
                    "area_km2": float(gm.group("area")),
                }

            record = {"iter": iter_num, "loss": loss_val, "metrics": metrics}
            parsed.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
