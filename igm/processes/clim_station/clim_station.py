"""
DEPRECATED — `igm.processes.clim_station` has been moved into the new
`climate` umbrella module at `igm.processes.climate.station`.

How to migrate your experiment YAML:

    defaults:
      - override /processes:
        - climate           # was: clim_station

    processes:
      climate:
        method: station     # selects igm.processes.climate.station
        station:
          update_freq: 1.0           # was: processes.clim_station.update_freq
          zero_degree_isotherm: 2500.0
          adiabatic_lapse_rate: 0.0058
          # ... (see igm/conf/processes/climate.yaml for the full default block)

Parameter path moved:
    cfg.processes.clim_station.X   ->   cfg.processes.climate.station.X

The implementation now lives at `igm/processes/climate/station/station.py`.
"""


def _migrated():
    raise RuntimeError(
        "igm.processes.clim_station has been replaced by the `climate` "
        "umbrella (method: station). Update your experiment YAML — see "
        "igm/processes/clim_station/clim_station.py for the migration recipe."
    )


def initialize(cfg, state):
    _migrated()


def update(cfg, state):
    _migrated()


def finalize(cfg, state):
    _migrated()
