from typing import List, Any
from types import ModuleType
from pathlib import Path

import time
import yaml

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from ...core import State
from ...utilities import print_info
from ..modules.loader import load_modules

from ....utils.profiling import profile_range

_console = Console(stderr=True)


def _load_module_meta(module) -> dict | None:
    if not hasattr(module, "__file__") or module.__file__ is None:
        return None
    name = module.__name__.split(".")[-1]
    meta_path = Path(module.__file__).parent / f"{name}.yaml"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return yaml.safe_load(f)


def _build_provider_map() -> dict[str, list[str]]:
    """Map each state variable to the module(s) that write it, from all per-module YAMLs."""
    providers: dict[str, list[str]] = {}
    processes_root = Path(__file__).parent.parent.parent.parent / "processes"
    if not processes_root.is_dir():
        return providers
    for mod_dir in sorted(processes_root.iterdir()):
        if not mod_dir.is_dir():
            continue
        name = mod_dir.name
        yaml_path = mod_dir / f"{name}.yaml"
        if not yaml_path.exists():
            continue
        try:
            meta = yaml.safe_load(yaml_path.read_text()) or {}
            for var in meta.get("updates", []):
                providers.setdefault(var, []).append(name)
        except Exception:
            pass
    return providers


def initialize_modules(processes: List, cfg: Any, state: State) -> None:
    for module in processes:
        if cfg.core.logging:
            state.logger.info(f"Initializing module: {module.__name__.split('.')[-1]}")
        module.initialize(cfg, state)


def update_modules(processes: List, outputs: List, cfg: Any, state: State) -> None:

    state.it = 0
    state.continue_run = True
    if cfg.core.print_comp:
        state.tcomp = {
            module.__name__.split(".")[-1]: [] for module in processes + outputs
        }
    while state.continue_run:
        for module in processes:
            m = module.__name__.split(".")[-1]
            if cfg.core.print_comp:
                state.tcomp[m].append(time.time())

            with profile_range(f"{m}", enabled=cfg.core.hardware.profile):
                module.update(cfg, state)

            if cfg.core.print_comp:
                state.tcomp[m][-1] -= time.time()
                state.tcomp[m][-1] *= -1
        run_outputs(outputs, cfg, state)
        if cfg.core.print_info:
            print_info(state)
        state.it += 1

        if not hasattr(state, "t"):
            state.continue_run = False


def finalize_modules(processes: List, cfg: Any, state: State) -> None:
    for module in processes:
        module.finalize(cfg, state)


def run_outputs(output_modules: List, cfg: Any, state: State) -> None:
    for module in output_modules:
        m = module.__name__.split(".")[-1]
        if cfg.core.print_comp:
            state.tcomp[m].append(time.time())
        module.run(cfg, state)
        if cfg.core.print_comp:
            state.tcomp[m][-1] -= time.time()
            state.tcomp[m][-1] *= -1


def setup_igm_modules(cfg, state) -> List[ModuleType]:
    return load_modules(cfg, state)


def check_module_needs(processes: List, state: State) -> None:
    errors: list[tuple[str, list[str]]] = []
    for module in processes:
        meta = _load_module_meta(module)
        if not meta or "needs" not in meta:
            continue
        # If a module declares outputs but none are on state after init, it ran
        # in a bypassed/reduced mode (e.g. iceflow in pretraining) — skip it.
        declared_updates = meta.get("updates", [])
        if declared_updates and not any(hasattr(state, v) for v in declared_updates):
            continue
        name = module.__name__.split(".")[-1]
        missing = [v for v in meta["needs"] if not hasattr(state, v)]
        if missing:
            errors.append((name, missing))

    if not errors:
        return

    provider_map = _build_provider_map()

    n_vars = sum(len(m) for _, m in errors)
    n_mods = len(errors)
    n_vars_str = f"{n_vars} state variable{'s' if n_vars != 1 else ''}"
    n_mods_str = f"{n_mods} module{'s' if n_mods != 1 else ''}"

    summary = Text.from_markup(
        f"[bold red]{n_vars_str}[/bold red] missing across [bold]{n_mods_str}[/bold].\n"
        "[dim]State variables are written by process modules — add the providing "
        "module(s) to your experiment configuration.[/dim]"
    )

    table = Table(
        box=box.SIMPLE_HEAD, padding=(0, 2), show_edge=False, header_style="dim"
    )
    table.add_column("Module", style="bold", no_wrap=True)
    table.add_column("Missing variable", no_wrap=True)
    table.add_column("Provided by", no_wrap=True)

    for mod_name, missing_vars in errors:
        for i, var in enumerate(missing_vars):
            providers = provider_map.get(var, [])
            provider_str = (
                ", ".join(f"[green]{p}[/green]" for p in providers)
                if providers
                else "[dim]unknown[/dim]"
            )
            table.add_row(
                mod_name if i == 0 else "",
                f"[red]✗[/red]  [bold red]{var}[/bold red]",
                provider_str,
            )

    _console.print(
        Panel(
            Group(summary, Text(""), table),
            title="[bold red]⚠  IGM — unmet module dependencies[/bold red]",
            border_style="red",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    raise RuntimeError(
        f"{n_vars_str} missing for {n_mods_str}: "
        + ", ".join(f"{mod}({', '.join(vars_)})" for mod, vars_ in errors)
    )
