"""
DEPRECATED — `igm.processes.smb_accpdd` has been moved into the new
`smb` umbrella module at `igm.processes.smb.accpdd`.

How to migrate your experiment YAML:

    defaults:
      - override /processes:
        - smb              # was: smb_accpdd

    processes:
      smb:
        method: accpdd     # selects igm.processes.smb.accpdd
        accpdd:
          update_freq: 1.0          # was: processes.smb_accpdd.update_freq
          refreeze_factor: 0.6
          thr_temp_snow: 0.0
          thr_temp_rain: 2.0
          melt_factor_snow: 1.095726596343
          melt_factor_ice:  2.921937590248
          shift_hydro_year: 0.75
          ice_density: 910.0
          wat_density: 1000.0
          smb_maximum_accumulation: 6.0

Parameter path moved:
    cfg.processes.smb_accpdd.X   ->   cfg.processes.smb.accpdd.X

The implementation now lives at `igm/processes/smb/accpdd/accpdd.py`.
"""


def _migrated():
    raise RuntimeError(
        "igm.processes.smb_accpdd has been replaced by the `smb` umbrella "
        "(method: accpdd). Update your experiment YAML — see "
        "igm/processes/smb_accpdd/smb_accpdd.py for the migration recipe."
    )


def initialize(cfg, state):
    _migrated()


def update(cfg, state):
    _migrated()


def finalize(cfg, state):
    _migrated()
