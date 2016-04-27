import numpy as np
from vessel_scoring.utils import get_cols_by_name

COURSE_STD = 0
SPEED_STD = 1
SPEED_AVG = 2
SHORE_DIST = 3

# TODO: make this inherit from appropriate sklearn classes
class LegacyHeuristicModel:

    def __init__(self, window='3600'):
        """
        window - window size to use in features
        """
        self.window = window

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        X = self._make_features(X)
        score = (X[:,COURSE_STD] + X[:,SPEED_STD] + X[:,SPEED_AVG]) * 2.0 / 3.0
        score = np.clip(score, 0, 1)
        # Port behavior is hard to distinguish from fishing,
        # so supress score close to shore...
#         not_close_to_shore = X[:,SHORE_DIST] >= 3
#         score *= not_close_to_shore
        proba = np.zeros([len(X), 2])
        proba[:, 0] = 1 - score # Probability not fishing
        proba[:, 1] = score
        return proba

    def _make_features(self, data):
        """Convert dataset into feature matrix suitable for model"""
        return get_cols_by_name(data,
                    ['measure_coursestddev_{window}' ,
                    'measure_speedstddev_{window}',
                    'measure_speedavg_{window}',
                    # 'distance_to_shore' # XXX waiting for this feature to be re-added
                    ], window=self.window)
