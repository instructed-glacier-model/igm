import tensorflow as tf


def build_network_params(cfg, arch_cls) -> dict:
    node = cfg.processes.iceflow.emulator.network
    params_node = getattr(node, "params", None)
    params = dict(params_node) if params_node is not None else {}
    if not params and hasattr(arch_cls, "_LEGACY_FIELDS"):
        params = {
            k: getattr(node, k)
            for k in arch_cls._LEGACY_FIELDS
            if getattr(node, k, None) is not None
        }
        precision = getattr(cfg.processes.iceflow.numerics, "precision", None)
        if precision is not None:
            params.setdefault("precision", str(precision))
    return params
