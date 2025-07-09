def validate_module(module) -> None:
    """Validates that a module has the required functions to be used in IGM."""
    required_functions = ["initialize", "finalize", "update"]
    for function in required_functions:
        if not hasattr(module, function):
            raise AttributeError(
                f"Module {module} is missing the required function ({function}). If it is a custom python package, make sure to include the 3 required functions: ['initialize', 'finalize', 'update'].",
                f"Please see https://github.com/jouvetg/igm/wiki/5.-Custom-modules-(coding) for more information on how to construct custom modules.",
            )
