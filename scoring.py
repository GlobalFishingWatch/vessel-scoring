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
from sklearn.ensemble import RandomForestClassifier


def fit_score(x, windows, seed=0):
    features = np.transpose(get_polynomial_cols(x, windows))
    y = x['classification'].astype(int)
    # In addition to cranking up n_estimators, there are a lot of
    # paramaters that can be tweaked on the RF classifier. See
    # http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    classifier = RandomForestClassifier(n_estimators=100,
                                        random_state=seed)
    classifier.fit(features, y)
    return classifier


def predict(classifier, data, windows):
    features = np.transpose(get_polynomial_cols(data, windows))
    return classifier.predict_proba(features)[:,1]


def optimize_window_size(xtrain, xcross):
    print "Fitting an optimal window size for avg/stddev columns"

    windows = np.array(get_windows(xtrain))
    err_train = np.zeros(windows.shape[0])
    err_cross = np.zeros(windows.shape[0])

    for idx, window in enumerate(windows):
        print "%s: Fitting..." % window,
        sys.stdout.flush()
        classifier = fit_score(xtrain, [window])
        print "Scoring...",
        sys.stdout.flush()
        xtrain['score'][:] = predict(classifier, xtrain, [window])
        xcross['score'][:] = predict(classifier, xcross, [window])
        print "Calc.err...",
        sys.stdout.flush()
        xtrainclassified = xtrain[xtrain["classification"] != np.Inf]
        err_train[idx] = np.sum((xtrainclassified['score'] - xtrainclassified['classification'])**2)/xtrainclassified.shape[0]
        xcrossclassified = xcross[xcross["classification"] != np.Inf]
        err_cross[idx] = np.sum((xcrossclassified['score'] - xcrossclassified['classification'])**2)/xcrossclassified.shape[0]
        print "train=%s, cross=%s" % (err_train[idx], err_cross[idx])
        sys.stdout.flush()

    min_window = windows[np.argmin(err_cross)]

    matplotlib.pyplot.figure(figsize=(20,5))
    matplotlib.pyplot.plot(windows, err_train, label="err train")
    matplotlib.pyplot.plot(windows, err_cross, label="err cross")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("Window size in seconds")
    matplotlib.pyplot.show()

    print "Best window size: %s" % min_window

    return min_window


def optimize_multi_window_sizes(xtrain, xcross):
    print "Fitting an optimal window size set for avg/stddev columns"
    print "For now uses a set from minimum window size up to X"

    all_windows = np.array(get_windows(xtrain))
    err_train = np.zeros(all_windows.shape[0])
    err_cross = np.zeros(all_windows.shape[0])

    windows = []
    for idx, window in enumerate(all_windows):
        windows.append(window)
        print "%s: Fitting..." % windows,
        sys.stdout.flush()
        classifier = fit_score(xtrain, windows)
        print "Scoring...",
        sys.stdout.flush()
        xtrain['score'][:] = predict(classifier, xtrain, windows)
        xcross['score'][:] = predict(classifier, xcross, windows)
        print "Calc.err...",
        sys.stdout.flush()
        xtrainclassified = xtrain[xtrain["classification"] != np.Inf]
        # Note that we are using accuracy rather than squared error as
        # our metric here.
        err_train[idx] = 1.0 * np.sum(((xtrainclassified['score'] > 0.5) != xtrainclassified['classification']))/xtrainclassified.shape[0]
        xcrossclassified = xcross[xcross["classification"] != np.Inf]
        err_cross[idx] = 1.0 * np.sum(((xcrossclassified['score'] > 0.5) != xcrossclassified['classification']))/xcrossclassified.shape[0]
        print "train=%s, cross=%s" % (err_train[idx], err_cross[idx])
        sys.stdout.flush()

    best_windows = windows[:np.argmin(err_cross)+1]

    matplotlib.pyplot.figure(figsize=(20,5))
    matplotlib.pyplot.plot(windows, err_train, label="err train")
    matplotlib.pyplot.plot(windows, err_cross, label="err cross")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("Max window size in seconds")
    matplotlib.pyplot.show()

    print "Best window sizes: %s" % best_windows

    return best_windows


def train_and_score(xtrain, xtest, windows):
    classifier = fit_score(xtrain, windows)
    xtest['score'][:] = predict(classifier, xtest, windows)
    return classifier

def evaluate_score(xtrain, xtest, windows):
    classifier = train_and_score(xtrain, xtest, windows)
    graph_score.graph_score(xtest, "score")
    graph_precall.graph_precall(xtest, "score")
    print "Score window:", windows
#     print "Score polynomial:", score_args
