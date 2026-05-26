"""
DEPRECATED — `igm.processes.smb_simple` has been moved into the new
`smb` umbrella module at `igm.processes.smb.simple`.

How to migrate your experiment YAML:

    defaults:
      - override /processes:
        - smb              # was: smb_simple

    processes:
      smb:
        method: simple     # selects igm.processes.smb.simple
        simple:
          update_freq: 1.0          # was: processes.smb_simple.update_freq
          file: param.txt
          array: []

Parameter path moved:
    cfg.processes.smb_simple.X   ->   cfg.processes.smb.simple.X

The implementation now lives at `igm/processes/smb/simple/simple.py`.
"""


def _migrated():
    raise RuntimeError(
        "igm.processes.smb_simple has been replaced by the `smb` umbrella "
        "(method: simple). Update your experiment YAML — see "
        "igm/processes/smb_simple/smb_simple.py for the migration recipe."
    )


def initialize(cfg, state):
    _migrated()


def update(cfg, state):
    _migrated()


def finalize(cfg, state):
    _migrated()
