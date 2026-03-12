# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""Shared Optuna utilities for the IGM sweeper."""

import logging

logger = logging.getLogger(__name__)


def create_study(optimize_cfg):
    """Create an Optuna study from optimize.yaml settings."""
    import optuna

    direction = optimize_cfg["direction"]

    # Resolve sampler
    sampler_cfg = optimize_cfg.get("sampler", {})
    sampler_cls = getattr(
        optuna.samplers, sampler_cfg.get("method", "TPESampler")
    )
    sampler_kwargs = {k: v for k, v in sampler_cfg.items() if k != "method"}
    sampler = sampler_cls(**sampler_kwargs)

    # Resolve pruner (optional)
    pruner = None
    pruner_cfg = optimize_cfg.get("pruner")
    if pruner_cfg and pruner_cfg.get("method"):
        pruner_cls = getattr(optuna.pruners, pruner_cfg["method"])
        pruner_kwargs = {k: v for k, v in pruner_cfg.items() if k != "method"}
        pruner = pruner_cls(**pruner_kwargs)

    storage = optimize_cfg.get("storage")
    study_name = optimize_cfg.get("study_name")

    if isinstance(direction, list):
        return optuna.create_study(
            study_name=study_name,
            directions=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
    else:
        return optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )


def report_results(study, output_dir, objective_names=None):
    """Write study results to CSV and print summary."""
    import csv
    from pathlib import Path

    is_multi_objective = hasattr(study, "directions") and isinstance(
        study.directions, (list, tuple)
    )

    if not is_multi_objective:
        best = study.best_trial
        print(f"\n=== Best trial ===")
        print(f"  Number: {best.number}")
        print(f"  Cost:   {best.value}")
        print(f"  Params: {best.params}")
    else:
        print(f"\n=== Multi-objective study: {len(study.best_trials)} Pareto-optimal trials ===")
        for t in study.best_trials:
            if objective_names:
                named = {n: v for n, v in zip(objective_names, t.values)}
                print(f"  Trial {t.number}: {named}, params={t.params}")
            else:
                print(f"  Trial {t.number}: values={t.values}, params={t.params}")

    # CSV export
    csv_path = Path(output_dir) / "optimization_results.csv"
    trials = study.trials
    if trials:
        if is_multi_objective and objective_names:
            value_fields = list(objective_names)
        else:
            value_fields = ["value"]
        fieldnames = ["number"] + value_fields + ["state"] + list(trials[0].params.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in trials:
                row = {"number": t.number, "state": t.state.name}
                if is_multi_objective and objective_names and t.values:
                    for name, val in zip(objective_names, t.values):
                        row[name] = val
                else:
                    row["value"] = t.value if not is_multi_objective else t.values
                row.update(t.params)
                writer.writerow(row)
        print(f"\nResults written to {csv_path}")
