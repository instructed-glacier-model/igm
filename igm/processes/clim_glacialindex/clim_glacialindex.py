"""
DEPRECATED — `igm.processes.clim_glacialindex` has been moved into the new
`climate` umbrella module at `igm.processes.climate.glacialindex`.

How to migrate your experiment YAML:

    defaults:
      - override /processes:
        - climate              # was: clim_glacialindex

    processes:
      climate:
        method: glacialindex   # selects igm.processes.climate.glacialindex
        glacialindex:
          update_freq: 100.0           # was: processes.clim_glacialindex.update_freq
          climate_0_file: data/climate.nc
          climate_1_file: data/climate1.nc
          signal_file: data/GI.dat
          vertical_lapse_rate_0: 6.0
          vertical_lapse_rate_1: 5.74
          temporal_resampling: 12

Parameter path moved:
    cfg.processes.clim_glacialindex.X   ->   cfg.processes.climate.glacialindex.X

The implementation now lives at `igm/processes/climate/glacialindex/glacialindex.py`.
"""


def _migrated():
    raise RuntimeError(
        "igm.processes.clim_glacialindex has been replaced by the `climate` "
        "umbrella (method: glacialindex). Update your experiment YAML — see "
        "igm/processes/clim_glacialindex/clim_glacialindex.py for the migration recipe."
    )


def initialize(cfg, state):
    _migrated()


def update(cfg, state):
    _migrated()


def finalize(cfg, state):
    _migrated()
