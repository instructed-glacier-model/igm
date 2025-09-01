import tensorflow as tf
import json
import numpy as np
import sys
from tqdm import tqdm
import datetime

from .visualizers import _plot_memory_pie, _plot_computational_pie


def print_comp(state):
    ################################################################

    size_of_tensor = {}

    for m in state.__dict__.keys():
        try:
            size_gb = sys.getsizeof(getattr(state, m).numpy())
            if size_gb > 1024**1:
                size_of_tensor[m] = size_gb / (1024**3)
        except:
            pass

    # sort from highest to lowest
    size_of_tensor = dict(
        sorted(size_of_tensor.items(), key=lambda item: item[1], reverse=True)
    )

    print("Memory statistics report:")
    with open("memory-statistics.txt", "w") as f:
        for key, value in size_of_tensor.items():
            print("     %24s  |  size : %8.4f Gb " % (key, value), file=f)
            print("     %24s  |  size : %8.4f Gb  " % (key, value))

    _plot_memory_pie(state)

    ################################################################

    modules = list(state.tcomp.keys())

    print("Computational statistics report:")
    with open("computational-statistics.txt", "w") as f:
        for m in modules:
            CELA = (m, np.mean(state.tcomp[m]), np.sum(state.tcomp[m]))
            print(
                "     %14s  |  mean time per it : %8.4f  |  total : %8.4f" % CELA,
                file=f,
            )
            print("     %14s  |  mean time per it : %8.4f  |  total : %8.4f" % CELA)

    _plot_computational_pie(state)


def print_gpu_info() -> None:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(f"{'CUDA Enviroment':-^150}")
    tf.sysconfig.get_build_info().pop("cuda_compute_capabilities", None)
    print(f"{json.dumps(tf.sysconfig.get_build_info(), indent=2, default=str)}")
    print(f"{'Available GPU Devices':-^150}")
    for gpu in gpus:
        gpu_info = {"gpu_id": gpu.name, "device_type": gpu.device_type}
        device_details = tf.config.experimental.get_device_details(gpu)
        gpu_info.update(device_details)

        print(f"{json.dumps(gpu_info, indent=2, default=str)}")
    print(f"{'':-^150}")


def print_info(state):

    if state.it % 100 == 1:
        if hasattr(state, "pbar"):
            state.pbar.close()
        state.pbar = tqdm(
            desc=f"IGM", ascii=False, dynamic_ncols=True, bar_format="{desc} {postfix}"
        )

    if hasattr(state, "pbar"):
        dic_postfix = {
            "🕒": datetime.datetime.now().strftime("%H:%M:%S"),
            "🔄": f"{state.it:06.0f}",
            "⏱ Time": f"{state.t.numpy():09.1f} yr",
            "⏳ Step": f"{state.dt:04.2f} yr",
        }
        if hasattr(state, "dx"):
            dic_postfix["❄️  Volume"] = (
                f"{np.sum(state.thk) * (state.dx**2) / 10**9:108.2f} km³"
            )
        if hasattr(state, "particle"):
            dic_postfix["# Particles"] = str(state.particle["x"].shape[0])

        #        dic_postfix["💾 GPU Mem (MB)"] = tf.config.experimental.get_memory_info("GPU:0")['current'] / 1024**2

        state.pbar.set_postfix(dic_postfix)
        state.pbar.update(1)
