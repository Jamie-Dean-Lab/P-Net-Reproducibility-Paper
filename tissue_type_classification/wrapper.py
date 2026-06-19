import numpy as np


class ProbaWrapper:
    def __init__(self, estimator, args):
        self.model = estimator
        self.args = args

    def fit(self, X, y):
        self.n_classes = y.shape[1] if y.ndim == 2 else len(np.unique(y))
        self.predictor = self.model(**self.args)
        self.predictor.fit(X, y.argmax(axis=1) if y.ndim == 2 else y)

    def predict(self, X):
        if hasattr(self.predictor, 'predict_proba'):
            proba = self.predictor.predict_proba(X)
            if proba.shape[1] == self.n_classes:
                return proba
            # Some classes were absent from the training fold — pad with zero probability
            full_proba = np.zeros((proba.shape[0], self.n_classes))
            full_proba[:, self.predictor.classes_] = proba
            return full_proba
        scores = self.predictor.decision_function(X)
        if scores.ndim == 1 or scores.shape[1] == self.n_classes:
            return scores
        # Some classes were absent from the training fold — pad below the minimum real score
        full_scores = np.full((scores.shape[0], self.n_classes), scores.min() - 1.0)
        full_scores[:, self.predictor.classes_] = scores
        return full_scores
