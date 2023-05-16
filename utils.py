def single_arg_constant_function(value):
    return lambda x: value


def ensure_single_arg_constant_function(value):
    if not callable(value):
        return single_arg_constant_function(value)
    return value