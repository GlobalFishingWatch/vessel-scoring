import numpy as np
from sklearn.ensemble import RandomForestClassifier
from vessel_scoring.utils import get_polynomial_cols
import vessel_scoring.base_model
import vessel_scoring.colspec

class RandomForestModel(RandomForestClassifier, vessel_scoring.base_model.BaseModel):

    def __init__(self, colspec, random_state=4321,
                        n_estimators=200):
        """
        windows - list of window sizes to use in features
        See RandomForestClassifier docs for other parameters.
        """
        RandomForestClassifier.__init__(self,
                                        n_estimators=n_estimators,
                                        random_state=random_state)
        self.colspec = vessel_scoring.colspec.Colspec(**colspec)

    def _make_features(self, data):
        """Convert dataset into feature matrix suitable for model"""
        return np.transpose(self.colspec.get_cols(data))

    def predict_proba(self, X):
        X = self._make_features(X)
        return RandomForestClassifier.predict_proba(self, X)

    def fit(self, X, y):
        X = self._make_features(X)
        return RandomForestClassifier.fit(self, X, y)


