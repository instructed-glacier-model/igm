IGM_DESCRIPTION = r"""
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │             Welcome to IGM, a modular, open-source, fast, and user-friendly glacier evolution model!             │
  │                                                                                                                  │
  │                                                                                                                  │
  │                         __/\\\\\\\\\\\_____/\\\\\\\\\\\\__/\\\\____________/\\\\_                                │
  │                          _\/////\\\///____/\\\//////////__\/\\\\\\________/\\\\\\_                               │
  │                           _____\/\\\______/\\\_____________\/\\\//\\\____/\\\//\\\_                              │
  │                            _____\/\\\_____\/\\\____/\\\\\\\_\/\\\\///\\\/\\\/_\/\\\_                             │
  │                             _____\/\\\_____\/\\\___\/////\\\_\/\\\__\///\\\/___\/\\\_                            │
  │                              _____\/\\\_____\/\\\_______\/\\\_\/\\\____\///_____\/\\\_                           │
  │                               _____\/\\\_____\/\\\_______\/\\\_\/\\\_____________\/\\\_                          │
  │                                __/\\\\\\\\\\\_\//\\\\\\\\\\\\/__\/\\\_____________\/\\\_                         │
  │                                 _\///////////___\////////////____\///______________\///__                        │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""


class State:
    _aliases: dict[str, str] = {}  # alias_name → canonical_name

    def __getattr__(self, name: str):
        # Only called when `name` is not found via normal __dict__ lookup.
        # Canonical names (topg, usurf, …) never reach here — zero overhead.
        canonical = State._aliases.get(name)
        if canonical is not None:
            try:
                return self.__dict__[canonical]
            except KeyError:
                pass
        raise AttributeError(f"'State' has no attribute '{name}'")

    def __setattr__(self, name: str, value) -> None:
        canonical = State._aliases.get(name)
        object.__setattr__(self, canonical if canonical is not None else name, value)

    @classmethod
    def register_aliases(cls, mapping: dict[str, str]) -> None:
        """Register alias_name → canonical_name pairs.

        Registering once at startup covers all State instances.
        IGM canonical names (topg, usurf, …) are the canonical side; descriptive
        or convention-specific names are the alias side so existing code needs
        no changes.
        """
        cls._aliases.update(mapping)