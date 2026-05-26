"""
DEPRECATED — `igm.processes.clim_oggm` has been moved into the new
`climate` umbrella module at `igm.processes.climate.oggm`.

How to migrate your experiment YAML:

    defaults:
      - override /processes:
        - climate          # was: clim_oggm

    processes:
      climate:
        method: oggm       # selects igm.processes.climate.oggm
        oggm:
          update_freq: 1.0           # was: processes.clim_oggm.update_freq
          file: file.txt
          clim_trend_array:
            - ["time", "delta_temp", "prec_scal"]
            - [1900, 0.0, 1.0]
            - [2020, 0.0, 1.0]
          ref_period: [1960, 1990]
          seed_par: 123

Parameter path moved:
    cfg.processes.clim_oggm.X   ->   cfg.processes.climate.oggm.X

The implementation now lives at `igm/processes/climate/oggm/oggm.py`.
"""


def _migrated():
    raise RuntimeError(
        "igm.processes.clim_oggm has been replaced by the `climate` umbrella "
        "(method: oggm). Update your experiment YAML — see "
        "igm/processes/clim_oggm/clim_oggm.py for the migration recipe."
    )


def initialize(cfg, state):
    _migrated()


def update(cfg, state):
    _migrated()


def finalize(cfg, state):
    _migrated()
