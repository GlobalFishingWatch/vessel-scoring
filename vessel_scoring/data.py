
from __future__ import print_function, division
import numpy as np
import numpy.lib.recfunctions
import os.path
import math
import scipy.optimize
import sys
from vessel_scoring.utils import *

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
    x = x[~np.isinf(x['classification']) & ~np.isnan(x['classification']) & ~np.isnan(x['timestamp']) & ~np.isnan(x['speed']) & ~np.isnan(x['course'])]

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

def _subsample_even(x0, mmsi, n):
    """Return `n` subsamples from `x0`

    - all samples have given `mmsi`

    - samples are evenly divided between fishing and nonfishing
    """
    # Create a mask that is true whenever mmsi is one of the mmsi
    # passed in
    mask = np.zeros([len(x0)], dtype=bool)
    for m in mmsi:
        mask |= (x0['mmsi'] == m)
    x = x0[mask]
    # Pick half the values from fishy rows and half from nonfishy rows.
    f = fishy(x)
    nf = nonfishy(x)
    if n//2 > len(f) or n//2 > len(nf):
        print("Warning, inufficient items to sample, returning fewer",
              file=sys.stderr)
    f = np.random.choice(f, min(n//2, len(f)), replace=False)
    nf = np.random.choice(nf, min(n//2, len(nf)), replace=False)
    ss = np.concatenate([f, nf])
    np.random.shuffle(ss)
    return ss

def _subsample_proportional(x0, mmsi, n):
    """Return `n` subsamples from `x0`

    - all samples have given `mmsi`

    - samples are random, so should have ~same be in the same proportions
      as the x0 for the given mmsi.
    """
    # Create a mask that is true whenever mmsi is one of the mmsi
    # passed in
    mask = np.zeros([len(x0)], dtype=bool)
    for m in mmsi:
        mask |= (x0['mmsi'] == m)
    x = x0[mask]
    # Pick values randomly
    # Pick values randomly
    if n > len(x):
        print("Warning, inufficient items to sample, returning", len(x),
               file=sys.stderr)
        n = len(x)
    ss = np.random.choice(x, n, replace=False)
    np.random.shuffle(ss)
    return ss

def add_log_measures(x):
    """Add log versions of each of speedstddev and coursestdev
    """
    all_windows = get_windows(x)
    for window in all_windows:
        x = np.lib.recfunctions.append_fields(x,
                'measure_speedstddev_%s_log' % window, [],
                 dtypes='<f8', fill_value=0.0)
        x['measure_speedstddev_%s_log' % window] = np.log10(
                x['measure_speedstddev_%s' % window]+0.001)

        x = np.lib.recfunctions.append_fields(x,
                'measure_coursestddev_%s_log' % window, [],
                dtypes='<f8', fill_value=0.0)
        x['measure_coursestddev_%s_log' % window] = np.log10(
                x['measure_coursestddev_%s' % window]+0.001)
    return np.lib.recfunctions.append_fields(x, 'score', [],
                                    dtypes='<f8', fill_value=0.0)

def load_dataset_by_vessel(path, size = 20000, even_split=None, seed=4321):
    """Load a dataset from `path` and return train, valid and test sets

    path - path to the dataset
    size - number of samples to return in total, divided between the
           three sets as (size//2, size//4, size//4)
    even_split - if True, use 50/50 fishing/nonfishing split for training
                  data, otherwise sample the data randomly.

    The data at path is first randomly divided by divided into
    training (1/2), validation (1/4) and test(1/4) data sets.
    These sets are chosen so that MMSI values are not shared
    across the datasets.

    The validation and test data are sampled randomly to get the
    requisite number of points. The training set is sampled randomly
    if `even_split` is False, otherwise it is chose so that half the
    points are fishing.

    """
    # Set the seed so that we can reproduce results consistently
    np.random.seed(seed)

    # Load the dataset and strip out any points that aren't classified
    # (has classification == Inf)
    x = np.load(path)['x']
    x = x[~np.isinf(x['classification']) & ~np.isnan(x['classification']) & ~np.isnan(x['timestamp']) & ~np.isnan(x['speed']) & ~np.isnan(x['course'])]

    if size > len(x):
        print("Warning, insufficient items to sample, returning all")
        size = len(x)

    # Get the list of MMSI and shuffle them. The compute the cumulative
    # lengths so that we can divide the points ~ evenly. Use search
    # sorted to find the division points
    mmsi = list(set(x['mmsi']))
    if even_split is None:
        even_split = x['classification'].sum() > 1 and x['classification'].sum() < len(x)
    if even_split:
        base_mmsi = mmsi
        # Exclude mmsi that don't have at least one fishing or nonfishing point
        mmsi = []
        for m in base_mmsi:
            subset = x[x['mmsi'] == m]
            fishing_count = subset['classification'].sum()
            if fishing_count == 0 or fishing_count == len(subset):
                continue
            mmsi.append(m)
    np.random.shuffle(mmsi)
    nx = len(x)
    sums = np.cumsum([(x['mmsi'] == m).sum() for m in mmsi])
    n1, n2 = np.searchsorted(sums, [nx//2, 3*nx//4])
    if n2 == n1:
        n2 += 1

    train_subsample = _subsample_even if even_split else _subsample_proportional

    try:
        xtrain = train_subsample(x, mmsi[:n1], size//2)
        xcross = _subsample_proportional(x, mmsi[n1:n2], size//4)
        xtest = _subsample_proportional(x, mmsi[n2:], size//4)
    except Exception as e:
        print("Warning: broken data in", path, file=sys.stderr)
        import pdb, sys
        sys.last_traceback = sys.exc_info()[2]
        pdb.set_trace()

    return x, xtrain, xcross, xtest
