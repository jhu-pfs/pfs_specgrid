def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def get_arg(name, old_value, args):
    if name in args and args[name] is not None:
        return args[name]
    else:
        return old_value

def is_arg(name, args):
    return name in args and args[name] is not None