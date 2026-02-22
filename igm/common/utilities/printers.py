import tensorflow as tf
import json
import numpy as np
import sys
from tqdm import tqdm
import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
import io
import re
import tensorflow as tf
import numpy as np


from .visualizers import _plot_memory_pie, _plot_computational_pie


PRINTER_COLORS = {
    "primary": "cyan",
    "secondary": "magenta",
    "accent": "green",
    "model_title": "bold cyan",
    "model_border": "blue",
    "model_header": "bold magenta",
    "model_layer": "cyan",
    "model_shape": "green",
    "model_params": "yellow",
    "model_params_zero": "dim",
    "model_none": "bold red",
    "input_title": "bold cyan",
    "input_border": "green",
    "input_header": "bold magenta",
    "input_variable": "cyan bold",
    "input_status": "white",
    "input_mean": "yellow",
    "input_variance": "yellow",
    "input_min": "blue",
    "input_median": "white",
    "input_max": "red",
    "status_found": "bold green",
    "status_missing": "bold red",
    "status_na": "dim",
    "label": "bold white",
    "value": "cyan",
    "value_alt": "green",
    "warning": "bold yellow",
    "info": "bold blue",
    "panel_border": "cyan",
    "panel_title": "bold",
}


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


def print_model_with_inputs(
    model,
    input_names,
    normalization_method="standardization",
    title="Model Architecture",
):
    """
    Print TensorFlow model summary alongside input variable information.

    Args:
        model: TensorFlow/Keras model
        input_names: List of variable names corresponding to channels
        normalization_method: String describing the normalization (e.g., "standardization", "min-max [0,1]", "channel-wise scaling")
        title: Title for the display
    """
    console = Console()

    # ===== MODEL TABLE =====
    string_buffer = io.StringIO()
    model.summary(print_fn=lambda x: string_buffer.write(x + "\n"))
    summary_str = string_buffer.getvalue()
    lines = summary_str.split("\n")

    model_table = Table(
        title=f"[bold cyan]{model.name}[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
        title_style="bold cyan",
        expand=False,
    )

    model_table.add_column("Layer (type)", style="cyan", no_wrap=True, width=20)
    model_table.add_column("Output Shape", style="green", width=22)
    model_table.add_column("Params", style="yellow", justify="right", width=10)

    # Parse model layers
    in_layers = False
    for line in lines:
        if line.strip().startswith("=") or line.strip().startswith("_"):
            continue
        if "Layer (type)" in line:
            in_layers = True
            continue
        if "Total params:" in line:
            in_layers = False

        if in_layers and line.strip():
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 3:
                layer_name = parts[0]
                output_shape = parts[1].replace("None", "[bold red]None[/bold red]")
                param_count = parts[2]

                if param_count != "0":
                    param_count = f"[bold yellow]{param_count}[/bold yellow]"
                else:
                    param_count = f"[dim]{param_count}[/dim]"

                model_table.add_row(layer_name, output_shape, param_count)

    # Get parameter summary
    total_params = trainable_params = None
    for line in lines:
        if "Total params:" in line:
            total_params = line.split("Total params:")[1].strip()
        elif "Trainable params:" in line:
            trainable_params = line.split("Trainable params:")[1].strip()

    model_table.caption = f"[bold white]Total:[/bold white] [cyan]{total_params}[/cyan] | [bold white]Trainable:[/bold white] [green]{trainable_params}[/green]"

    # ===== INPUT VARIABLES TABLE =====
    input_table = Table(
        title="[bold cyan]Input Variables[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="green",
        title_style="bold cyan",
        expand=False,
    )

    input_table.add_column("#", style="dim", justify="right", width=4)
    input_table.add_column("Variable", style="cyan bold", width=18)

    # Add each variable
    for i, var_name in enumerate(input_names):
        input_table.add_row(str(i), var_name)

    # Add normalization info as caption
    input_table.caption = (
        f"[bold white]Channels:[/bold white] [cyan]{len(input_names)}[/cyan] | "
        f"[bold white]Normalization:[/bold white] [yellow]{normalization_method}[/yellow]"
    )

    # ===== DISPLAY SIDE BY SIDE =====
    console.print()
    console.print(
        Panel(
            Columns([model_table, input_table], equal=False, expand=True),
            title=f"[bold]{title}[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def print_model_with_inputs_detailed(
    model, input_data, cfg_inputs, normalization_method, title="Model Information"
):
    """
    Extended version with standard deviation and percentiles.

    Args:
        model: TensorFlow/Keras model
        input_data: Dictionary of input fields {field_name: array_data}
        cfg_inputs: List of input variable names from cfg.unified.inputs
        normalization_method: String describing normalization method
        title: Title for the display
    """
    console = Console()

    # ===== MODEL TABLE =====
    string_buffer = io.StringIO()
    model.summary(print_fn=lambda x: string_buffer.write(x + "\n"))
    summary_str = string_buffer.getvalue()
    lines = summary_str.split("\n")

    model_table = Table(
        title=f"[{PRINTER_COLORS['model_title']}]{model.name}[/{PRINTER_COLORS['model_title']}]",
        show_header=True,
        header_style=PRINTER_COLORS["model_header"],
        border_style=PRINTER_COLORS["model_border"],
        title_style=PRINTER_COLORS["model_title"],
        expand=False,
    )

    model_table.add_column(
        "Layer (type)", style=PRINTER_COLORS["model_layer"], no_wrap=True, width=20
    )
    model_table.add_column(
        "Output Shape", style=PRINTER_COLORS["model_shape"], width=22
    )
    model_table.add_column(
        "Params", style=PRINTER_COLORS["model_params"], justify="right", width=10
    )

    in_layers = False
    for line in lines:
        if line.strip().startswith("=") or line.strip().startswith("_"):
            continue
        if "Layer (type)" in line:
            in_layers = True
            continue
        if "Total params:" in line:
            in_layers = False

        if in_layers and line.strip():
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 3:
                layer_name = parts[0]
                output_shape = parts[1].replace(
                    "None",
                    f"[{PRINTER_COLORS['model_none']}]None[/{PRINTER_COLORS['model_none']}]",
                )
                param_count = parts[2]

                if param_count != "0":
                    param_count = f"[{PRINTER_COLORS['model_params']}]{param_count}[/{PRINTER_COLORS['model_params']}]"
                else:
                    param_count = f"[{PRINTER_COLORS['model_params_zero']}]{param_count}[/{PRINTER_COLORS['model_params_zero']}]"

                model_table.add_row(layer_name, output_shape, param_count)

    total_params = trainable_params = None
    for line in lines:
        if "Total params:" in line:
            total_params = line.split("Total params:")[1].strip()
        elif "Trainable params:" in line:
            trainable_params = line.split("Trainable params:")[1].strip()

    model_table.caption = f"[{PRINTER_COLORS['label']}]Total:[/{PRINTER_COLORS['label']}] [{PRINTER_COLORS['value']}]{total_params}[/{PRINTER_COLORS['value']}] | [{PRINTER_COLORS['label']}]Trainable:[/{PRINTER_COLORS['label']}] [{PRINTER_COLORS['value_alt']}]{trainable_params}[/{PRINTER_COLORS['value_alt']}]"

    # ===== DETAILED INPUT VARIABLES TABLE =====
    input_table = Table(
        title=f"[{PRINTER_COLORS['input_title']}]Input Variables Statistics[/{PRINTER_COLORS['input_title']}]",
        show_header=True,
        header_style=PRINTER_COLORS["input_header"],
        border_style=PRINTER_COLORS["input_border"],
        title_style=PRINTER_COLORS["input_title"],
        expand=False,
    )

    input_table.add_column("Variable", style=PRINTER_COLORS["input_variable"], width=14)
    input_table.add_column(
        "Status", style=PRINTER_COLORS["input_status"], justify="center", width=12
    )
    input_table.add_column(
        "Mean", style=PRINTER_COLORS["input_mean"], justify="right", width=10
    )
    input_table.add_column(
        "Variance", style=PRINTER_COLORS["input_variance"], justify="right", width=10
    )
    input_table.add_column(
        "Min", style=PRINTER_COLORS["input_min"], justify="right", width=10
    )
    input_table.add_column(
        "Median", style=PRINTER_COLORS["input_median"], justify="right", width=10
    )
    input_table.add_column(
        "Max", style=PRINTER_COLORS["input_max"], justify="right", width=10
    )

    # Check which variables are available and requested
    available_keys = set(input_data.keys())
    requested_keys = set(cfg_inputs)

    # Iterate through requested inputs and check availability
    for var_name in cfg_inputs:
        if var_name in available_keys:
            # Variable is available - compute statistics
            field_data = input_data[var_name]

            # Convert to numpy if needed
            if isinstance(field_data, tf.Tensor):
                field_data = field_data.numpy()

            # Flatten for statistics
            flat_data = field_data.flatten()

            mean_val = np.mean(flat_data)
            # var_val = np.std(flat_data)**2
            var_val = np.mean((flat_data - mean_val) ** 2)
            min_val = np.min(flat_data)
            median_val = np.median(flat_data)
            max_val = np.max(flat_data)

            input_table.add_row(
                var_name,
                f"[{PRINTER_COLORS['status_found']}]✓ Found[/{PRINTER_COLORS['status_found']}]",
                f"{mean_val:.3e}",
                f"{var_val:.3e}",
                f"{min_val:.3e}",
                f"{median_val:.3e}",
                f"{max_val:.3e}",
            )
        else:
            # Variable is requested but not available
            input_table.add_row(
                var_name,
                f"[{PRINTER_COLORS['status_missing']}]✗ Missing[/{PRINTER_COLORS['status_missing']}]",
                f"[{PRINTER_COLORS['status_na']}]N/A[/{PRINTER_COLORS['status_na']}]",
                f"[{PRINTER_COLORS['status_na']}]N/A[/{PRINTER_COLORS['status_na']}]",
                f"[{PRINTER_COLORS['status_na']}]N/A[/{PRINTER_COLORS['status_na']}]",
                f"[{PRINTER_COLORS['status_na']}]N/A[/{PRINTER_COLORS['status_na']}]",
                f"[{PRINTER_COLORS['status_na']}]N/A[/{PRINTER_COLORS['status_na']}]",
            )

    # Get shape information from first available field
    if available_keys:
        first_key = list(available_keys)[0]
        sample_data = input_data[first_key]

        if isinstance(sample_data, tf.Tensor):
            sample_data = sample_data.numpy()

        if sample_data.ndim == 2:  # (N, features)
            n_samples, n_features = sample_data.shape
            shape_str = f"({n_samples}, {n_features})"
            total_points = n_samples * n_features
        else:
            shape_str = str(sample_data.shape)
            total_points = sample_data.size

        input_table.caption = (
            f"[{PRINTER_COLORS['label']}]Shape:[/{PRINTER_COLORS['label']}] [{PRINTER_COLORS['value']}]{shape_str}[/{PRINTER_COLORS['value']}] | "
            f"[{PRINTER_COLORS['label']}]Total points:[/{PRINTER_COLORS['label']}] [{PRINTER_COLORS['value_alt']}]{total_points:,}[/{PRINTER_COLORS['value_alt']}] | "
            f"[{PRINTER_COLORS['label']}]Normalization:[/{PRINTER_COLORS['label']}] [{PRINTER_COLORS['value']}]{normalization_method}[/{PRINTER_COLORS['value']}]"
        )
    else:
        input_table.caption = f"[{PRINTER_COLORS['label']}]Normalization:[/{PRINTER_COLORS['label']}] [{PRINTER_COLORS['value']}]{normalization_method}[/{PRINTER_COLORS['value']}]"

    # Check for missing or extra variables
    missing_vars = requested_keys - available_keys
    extra_vars = available_keys - requested_keys

    if missing_vars or extra_vars:
        console.print()
        if missing_vars:
            console.print(
                f"[{PRINTER_COLORS['warning']}]⚠ Warning:[/{PRINTER_COLORS['warning']}] Missing variables in input_data: {', '.join(missing_vars)}"
            )
        if extra_vars:
            console.print(
                f"[{PRINTER_COLORS['info']}]ℹ Info:[/{PRINTER_COLORS['info']}] Extra variables in input_data (not used): {', '.join(extra_vars)}"
            )

    # ===== DISPLAY SIDE BY SIDE =====
    console.print()
    console.print(
        Panel(
            Columns([model_table, input_table], equal=False, expand=True),
            title=f"[{PRINTER_COLORS['panel_title']}]{title}[/{PRINTER_COLORS['panel_title']}]",
            border_style=PRINTER_COLORS["panel_border"],
            padding=(1, 2),
        )
    )
    console.print()
