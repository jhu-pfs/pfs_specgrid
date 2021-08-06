import sys

def func_fullname(f):
    m = sys.modules[f.__module__]
    return '{}.{}'.format(m.__name__, f.__qualname__)