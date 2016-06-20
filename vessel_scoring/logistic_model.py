import numpy as np
from sklearn.linear_model import LogisticRegression
from vessel_scoring.utils import get_polynomial_cols, zigmoid
import vessel_scoring.base_model
import vessel_scoring.colspec

def make_features(base_cols, order, cross):
    n_base_cols = len(base_cols)
    cols = []
    for total_order in range(1, order+1):
        for i in range(n_base_cols):
            cols.append(base_cols[i]**total_order)
        if total_order <= cross:
            for i in range(n_base_cols):
                for j in range(i+1, n_base_cols):
                    # Loop from i+1 up, so that we only get
                    # off diagonal terms once.
                    for sub_order in range(1, total_order):
                        # sub_order ranges from 1-total_order-1
                        # so only when i==j do we get columns
                        # with total order.
                        cols.append(base_cols[i] ** sub_order *
                                    base_cols[j] ** (total_order - sub_order))
    chunks = [x.reshape(-1,1) for x in cols]
    return np.concatenate(chunks, axis=1)


class LogisticModel(LogisticRegression, vessel_scoring.base_model.BaseModel):

    def __init__(self, coef=None, intercept=None, order=4, cross=0,
                        colspec={}, random_state=4321):
        """

        The first to arguments are here to make interface consistent
        with LogisiticScorer:

        ceof - feature coeficients to initialize the model with
        intercept - intercept value to initialize the model with

        order - maximum order of polynomial terms
        cross - maximum order of cross terms (2 is minimum for any effect)
        colspec - specification of what cols to use

        Note that this uses only cross terms from two features at
        a time.
        """
        LogisticRegression.__init__(self, random_state=random_state)
        assert order >= 2, "order must be at least 2"
        self.order = order
        self.cross = cross
        self.colspec = vessel_scoring.colspec.Colspec(**colspec)
        if coef is not None:
            self.coef_ = np.array(coef)
        if intercept is not None:
            self.intercept_ = np.array(intercept)

    def fit(self, X, y):
        """Fit model bease on features `X` and labels `y`"""
        X = self._make_features(X)
        return LogisticRegression.fit(self, X, y)

    def predict_proba(self, X):
        """Predict probabilities based on feature vector `X`"""
        X = self._make_features(X)
        return LogisticRegression.predict_proba(self, X)

    def _make_features(self, data):
        """Convert dataset into feature matrix suitable for model"""
        return make_features(
            np.array(self.colspec.get_cols(data)),
            self.order, self.cross)

    def dump_arg_dict(self):
        return {'coef' : [list(item) for item in self.coef_],
                'intercept' : list(self.intercept_),
                'colspec' : self.colspec.dump_arg_dict(),
                'order' : self.order,
                'cross' : self.cross}



class LogisticScorer(vessel_scoring.base_model.BaseModel):
    """
    Reimplementation of the prediction part of Sklearn's LogisticRegression
    class. Idea is that we can optimize it once we stuff it in the pipe
    line, where we wouldn't be able to do that with sklearn.
    """

    def __init__(self, coef, intercept, order, cross, colspec):
        self.coef = coef
        self.intercept = intercept
        self.order = order
        self.cross = cross
        self.colspec = colspec

    def predict(self, X):
        """predict is_fishing based on feature vector `X`"""
        return self.predict_proba(X) > 0.5

    def predict_proba(self, X):
        """Predict probabilities based on feature vector `X`

        X is n_predictions x n_features
        """
        X = self._make_features(X)
        z = (self.coef * X).sum(axis=1) + self.intercept
        score = zigmoid(z)
        proba = np.zeros([len(X), 2])
        proba[:, 0] = 1 - score # Probability not fishing
        proba[:, 1] = score
        return proba

    def fishing_score(self, X):
        return self.predict_proba(X)[:,1]

    def _make_features(self, data):
        """Convert dataset into feature matrix suitable for model"""
        return make_features(
            np.array(self.colspec.get_cols(data)),
            self.order, self.cross)

