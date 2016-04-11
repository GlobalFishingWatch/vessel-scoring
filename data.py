import numpy as np
import numpy.lib.recfunctions
import os.path
import math
import scipy.optimize
import matplotlib.pyplot
import sys
import graph_score
import graph_precall
from utils import *

def load_dataset(path, size = 20000):
    # Load a dataset and extract a train, cross validation and test dataset
    #
    # * We need roughly the same amount of fishing and non-fishing
    #   rows to get good predictions, but the source data for some
    #   vessel types contain mostly non-fishing rows, so we randomly
    #   select 1000 fishing rows and the same number of non-fishing
    #   rows
    # * We add the log of the stddev columns, since their values are
    #   exponentially distributed

    x = np.load(path)['x']

    x = x[np.isinf(x['classification']) != True]

    all_windows = get_windows(x)

    for window in all_windows:
        x = np.lib.recfunctions.append_fields(x, 'measure_speedstddev_%s_log' % window, [], dtypes='<f8', fill_value=0.0)
        x['measure_speedstddev_%s_log' % window] = np.log10(x['measure_speedstddev_%s' % window]+0.001)

        x = np.lib.recfunctions.append_fields(x, 'measure_coursestddev_%s_log' % window, [], dtypes='<f8', fill_value=0.0)
        x['measure_coursestddev_%s_log' % window] = np.log10(x['measure_coursestddev_%s' % window]+0.001)

    x = np.lib.recfunctions.append_fields(x, 'score', [], dtypes='<f8', fill_value=0.0)

    xuse = numpy.copy(x)
    np.random.shuffle(xuse)
    size = min(fishy(xuse).shape[0], nonfishy(xuse).shape[0], size/2)
    xuse = np.concatenate((fishy(xuse)[:size], nonfishy(xuse)[:size]))
    np.random.shuffle(xuse)

    length = xuse.shape[0]
    xtrain = xuse[:length / 2]
    xcross = xuse[length/2:length*3/4]
    xtest = xuse[length*3/4:]

    return x, xtrain, xcross, xtest

def _subsample_matched(x0, mmsi, n):
    # Create a mask that is true whenever mmsi is one of the mmsi
    # passed in
    mask = np.zeros([len(x0)], dtype=bool)
    for m in mmsi:
        mask |= (x0['mmsi'] == m)
    x = x0[mask]
    # Pick half the values from fishy rows and half from nonfishy rows.
    f = fishy(x)
    nf = nonfishy(x)
    f = np.random.choice(f, n//2, replace=False)
    nf = np.random.choice(nf, n//2, replace=False)
    ss = np.concatenate([f, nf])
    np.random.shuffle(ss)
    return ss

def _subsample_proportional(x0, mmsi, n):
    # Create a mask that is true whenever mmsi is one of the mmsi
    # passed in
    mask = np.zeros([len(x0)], dtype=bool)
    for m in mmsi:
        mask |= (x0['mmsi'] == m)
    x = x0[mask]
    # Pick half the values from fishy rows and half from nonfishy rows.
    ss = np.random.choice(x, n, replace=False)
    np.random.shuffle(ss)
    return ss

def load_dataset_by_vessel(path, size = 20000, matched=True):
    # Load a dataset and extract a train, cross validation and test dataset
    #
    # * We need roughly the same amount of fishing and non-fishing
    #   rows to get good predictions, but the source data for some
    #   vessel types contain mostly non-fishing rows, so we randomly
    #   select 1000 fishing rows and the same number of non-fishing
    #   rows
    # * We add the log of the stddev columns, since their values are
    #   exponentially distributed

    x = np.load(path)['x']

    x = x[np.isinf(x['classification']) != True]

    all_windows = get_windows(x)

    for window in all_windows:
        x = np.lib.recfunctions.append_fields(x, 'measure_speedstddev_%s_log' % window, [], dtypes='<f8', fill_value=0.0)
        x['measure_speedstddev_%s_log' % window] = np.log10(x['measure_speedstddev_%s' % window]+0.001)

        x = np.lib.recfunctions.append_fields(x, 'measure_coursestddev_%s_log' % window, [], dtypes='<f8', fill_value=0.0)
        x['measure_coursestddev_%s_log' % window] = np.log10(x['measure_coursestddev_%s' % window]+0.001)

    x = np.lib.recfunctions.append_fields(x, 'score', [], dtypes='<f8', fill_value=0.0)

    mmsi = list(set(x['mmsi']))
    np.random.shuffle(mmsi)
    n_mmsi = len(mmsi)

    subsample = _subsample_matched if matched else _subsample_proportional

    xtrain = subsample(x, mmsi[:n_mmsi//2], size//2)
    xcross = _subsample_proportional(x, mmsi[n_mmsi//2:3*n_mmsi//4], size//4)
    xtest = _subsample_proportional(x, mmsi[3*n_mmsi//4:], size//4)

    return x, xtrain, xcross, xtest