import tensorflow as tf


def build_network_params(network_node, arch_cls, precision=None) -> dict:
    """Build the architecture's ``network_params`` dict from a config node.

    *network_node* is the network config block
    (``cfg.processes.iceflow.unified.network``). Its ``params`` sub-dict holds
    the architecture hyperparameters; when it is empty, architectures that
    declare ``_LEGACY_FIELDS`` read the flat fields of the node instead.
    """
    node = network_node
    params_node = getattr(node, "params", None)
    params = dict(params_node) if params_node is not None else {}
    # When the architecture declares its accepted keys, filter the merged config
    # to those keys only.  This prevents cross-architecture config bleed when
    # switching architectures without fully resetting the params subtree.
    if params and hasattr(arch_cls, "_DEFAULTS"):
        params = {k: v for k, v in params.items() if k in arch_cls._DEFAULTS}
    if not params and hasattr(arch_cls, "_LEGACY_FIELDS"):
        params = {
            k: getattr(node, k)
            for k in arch_cls._LEGACY_FIELDS
            if getattr(node, k, None) is not None
        }
        if precision is not None:
            params.setdefault("precision", str(precision))
    return params
