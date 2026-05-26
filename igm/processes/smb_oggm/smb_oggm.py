"""
DEPRECATED — `igm.processes.smb_oggm` has been moved into the new
`smb` umbrella module at `igm.processes.smb.oggm`.

How to migrate your experiment YAML:

    defaults:
      - override /processes:
        - smb              # was: smb_oggm

    processes:
      smb:
        method: oggm       # selects igm.processes.smb.oggm
        oggm:
          update_freq: 1.0          # was: processes.smb_oggm.update_freq
          ice_density: 910.0
          wat_density: 1000.0
          melt_enhancer: 1.0

Parameter path moved:
    cfg.processes.smb_oggm.X   ->   cfg.processes.smb.oggm.X

The implementation now lives at `igm/processes/smb/oggm/oggm.py`.
"""


def _migrated():
    raise RuntimeError(
        "igm.processes.smb_oggm has been replaced by the `smb` umbrella "
        "(method: oggm). Update your experiment YAML — see "
        "igm/processes/smb_oggm/smb_oggm.py for the migration recipe."
    )


def initialize(cfg, state):
    _migrated()


def update(cfg, state):
    _migrated()


def finalize(cfg, state):
    _migrated()
