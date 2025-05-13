import numpy as np

def float_format(num, precision=10, trim='-'):
    return np.format_float_positional(num, precision=precision, trim=trim)
