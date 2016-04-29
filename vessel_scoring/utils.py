import numpy as np
import numpy.lib.recfunctions
import os.path
import math
import sys

# Define some usefull functions
def clamp(x, low, high):
    return np.where(x <= high,
        np.where(x >= low, x, low),
        high)

def center_hist_bins(histres):
    y = histres[0]
    x = histres[1]
    x = (x[:-1] + x[1:]) / 2.0
    return y, x

def polynomial(x, *args):
    res = 0
    for ind, arg in enumerate(args):
        res += arg * x**ind
    return res

def linear(x, *args):
    res = 0.0
    for idx, arg in enumerate(args):
        res += x[idx]*arg
    return res

def mpolynomial(x, *args):
    res = 0
    for col, colargs in zip(x, np.array_split(args, len(x))):
        res += polynomial(col, *colargs)
    return res

def zigmoid(z):
    z = np.where(z < -100, -100, z)
    return 1. / (1. + np.exp(-z))

def zmpolynomial(x, *args):
    return zigmoid(mpolynomial(x, *args))

def cached(path):
    def cached(fn):
        def cached(*arg, **kw):
            if os.path.exists(path):
                return np.load(path)['cached']
            else:
                res = fn(*arg, **kw)
                np.savez_compressed(path, cached = res)
                return res
        return cached
    return cached

def is_fishy(x):
    return x["classification"] >= 0.5

def is_ambiguous(x):
    return abs(x["classification"] - 0.5) < 0.15 # ~between 1/3 and 2/3

def fishy(x):
    return x[is_fishy(x)]

def nonfishy(x):
    return x[~is_fishy(x)]



def get_polynomial_cols(x, windows):
    colnames = []
    #colnames.append("speed")
    for window in windows:
        colnames.append('measure_speedavg_%s' % window)
        colnames.append('measure_speedstddev_%s_log' % window)
        colnames.append('measure_coursestddev_%s_log' % window)

    cols = [x[col] for col in colnames]

    return cols


def get_windows(x):
    all_windows = [int(name[len("measure_speedavg_"):])
                   for name in x.dtype.names
                   if name.startswith("measure_speedavg_")]
    all_windows.sort()
    return all_windows


def get_cols_by_name(data, names, dtype=float, **kwargs):
    """get columns from recarray `data` with `names` and return as `dtype`

    kwargs is substititued into each name using str.format before
    using the name to extract an item.
    """
    features = np.zeros([len(data), len(names)], dtype=dtype)
    for i, name in enumerate(names):
        name = name.format(**kwargs)
        features[:,i] = data[name]
    return features

def clone_subset(x, dtype):
    """copy only the portions of x in dtype to a new array"""
    new = np.zeros(x.shape, dtype=dtype)
    for name in dtype.names:
        new[name] = x[name]
    return new

