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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def augment(x, grade):
    x = np.transpose(x)

    return x

    n, m = x.shape
    m1 = m + m*(m-1)//2
    a = np.zeros([n, grade * m1])
    a[:,:m] = x
    ndx = m
    for i in range(m):
        for j in range(i+1,m):
            a[:,ndx] = x[:,i] * x[:,j]
            ndx += 1
    for i in range(2, grade+1):
        a[:,ndx:ndx+m1] = a[:,:m1] ** i
        ndx += m1
    assert ndx == a.shape[-1], (ndx, a.shape, m1, grade)
    return a

def fit_score(x, windows, grade=4, lambda_val=0.0001):
    cols = augment(get_polynomial_cols(x, windows), grade)
    y = x['classification'].astype(int)
    # Not sure of the exact relationship between lambda and C
    # except that they have an inverse relationship
#     classifier = LogisticRegression(C=1/lambda_val)
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(cols, y)
    return classifier

def optimize_window_size(xtrain, xcross, grade = 4):
    print "Fitting an optimal window size for avg/stddev columns"

    windows = np.array(get_windows(xtrain))
    err_train = np.zeros(windows.shape[0])
    err_cross = np.zeros(windows.shape[0])

    for idx, window in enumerate(windows):
        print "%s: Fitting..." % window,
        sys.stdout.flush()
        classifier = fit_score(xtrain, [window], grade)
        print "Scoring...",
        sys.stdout.flush()
        xtrain['score'][:] = classifier.predict_proba(
            augment(get_polynomial_cols(xtrain, [window]), grade))[:,1]
        xcross['score'][:] = classifier.predict_proba(
            augment(get_polynomial_cols(xcross, [window]), grade))[:,1]

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

def optimize_polynomial_degree(xtrain, xcross, min_window, max_grade = 10):
    print "Fitting an optimal degree of polynomial"

    windows = [min_window]
    grades = range(2, max_grade)
    err_train = np.zeros(len(grades))
    err_cross = np.zeros(len(grades))

    for idx, grade in enumerate(grades):
        print "%s: Fitting..." % grade,
        sys.stdout.flush()
        classifier = fit_score(xtrain, windows, grade)
        print "Scoring...",
        sys.stdout.flush()
        xtrain['score'][:] = classifier.predict_proba(
            augment(get_polynomial_cols(xtrain, [min_window]), grade))[:,1]
        xcross['score'][:] = classifier.predict_proba(
            augment(get_polynomial_cols(xcross, [min_window]), grade))[:,1]
        print "Calc.err...",
        sys.stdout.flush()
        xtrainclassified = xtrain[xtrain["classification"] != np.Inf]
        err_train[idx] = np.sum((xtrainclassified['score'] - xtrainclassified['classification'])**2)/xtrainclassified.shape[0]
        xcrossclassified = xcross[xcross["classification"] != np.Inf]
        err_cross[idx] = np.sum((xcrossclassified['score'] - xcrossclassified['classification'])**2)/xcrossclassified.shape[0]
        print "train=%s, cross=%s" % (err_train[idx], err_cross[idx])
        sys.stdout.flush()
    grades = np.array(grades)

    matplotlib.pyplot.figure(figsize=(20,5))
    matplotlib.pyplot.plot(grades, err_train, label="err train")
    matplotlib.pyplot.plot(grades, err_cross, label="err cross")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.xlabel("Polynomial degree (for each variable)")
    matplotlib.pyplot.show()

def train_and_score(xtrain, xtest, grade, min_window):
    windows = [min_window]
    classifier = fit_score(xtrain, windows, grade)
    xtest['score'][:] = classifier.predict_proba(
        augment(get_polynomial_cols(xtest, [min_window]), grade))[:,1]
    return classifier

def evaluate_score(xtrain, xtest, grade, min_window):
    classifier = train_and_score(xtrain, xtest, grade, min_window)
    graph_score.graph_score(xtest, "score")
    graph_precall.graph_precall(xtest, "score")
    print "Score window:", min_window
#     print "Score polynomial:", score_args
