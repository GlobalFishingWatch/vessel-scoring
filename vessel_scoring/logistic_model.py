import numpy as np
from sklearn.linear_model import LogisticRegression
from vessel_scoring.utils import get_polynomial_cols, zigmoid


def make_features(data, windows, order, cross):
    base = np.array(get_polynomial_cols(data, windows))
    n_base_cols = len(base)
    cols = []
    for total_order in range(1, order+1):
        for i in range(n_base_cols):
            cols.append(base[i]**total_order)
        if total_order <= cross:
            for i in range(n_base_cols):
                for j in range(i+1, n_base_cols):
                    # Loop from i+1 up, so that we only get
                    # off diagonal terms once.
                    for sub_order in range(1, total_order):
                        # sub_order ranges from 1-total_order-1
                        # so only when i==j do we get columns
                        # with total order.
                        cols.append(base[i] ** sub_order *
                                    base[j] ** (total_order - sub_order))
    chunks = [x.reshape(-1,1) for x in cols]
    return np.concatenate(chunks, axis=1)


class LogisticModel(LogisticRegression):

    def __init__(self, order=4, cross=0, windows=['3600'],
                    random_state=4321):
        """
        order - maximum order of polynomial terms
        cross - maximum order of cross terms (2 is minimum for any effect)
        windows - list of window sizes to use in features

        Note that this uses only cross terms from two features at
        a time.
        """
        LogisticRegression.__init__(self, random_state=random_state)
        assert order >= 2, "order must be at least 2"
        self.order = order
        self.cross = cross
        self.windows = windows

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
        return make_features(data, self.windows, self.order, self.cross)

    def dump_dict(self):
        return {'coef' : [list(item) for item in self.coef_],
                'intercept' : list(self.intercept_),
                'windows' : list(self.windows),
                'order' : self.order,
                'cross' : self.cross}



class LogisticScorer:
    """
    Reimplementation of the prediction part of Sklearn's LogisticRegression
    class. Idea is that we can optimize it once we stuff it in the pipe
    line, where we wouldn't be able to do that with sklearn.
    """

    def __init__(self, coef, intercept, order, cross, windows):
        self.coef = coef
        self.intercept = intercept
        self.order = order
        self.cross = cross
        self.windows = windows

    def predict(self, X):
        """predict is_fishing based on feature vector `X`"""
        return self.predict_proba(X) > 0.5

    def predict_proba(self, X):
        """Predict probabilities based on feature vector `X`

        X is n_predictions x n_features
        """
        X = self._make_features(X)
        z = (self.coef * X).sum(axis=1) + self.intercept
        print z.max(), z.min(), abs(X).max(), abs(self.coef).max(), self.intercept
        score = zigmoid(z)
        proba = np.zeros([len(X), 2])
        proba[:, 0] = 1 - score # Probability not fishing
        proba[:, 1] = score
        return proba

    def fishing_score(self, X):
        return self.predict_proba(X)[:,1]

    def _make_features(self, data):
        """Convert dataset into feature matrix suitable for model"""
        return make_features(data, self.windows, self.order, self.cross)

