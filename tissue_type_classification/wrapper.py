import numpy as np


class ProbaWrapper:
    def __init__(self, estimator, args):
        self.model = estimator
        self.args = args

    def fit(self, X, y):
        self.n_classes = y.shape[1] if y.ndim == 2 else len(np.unique(y))
        self.predictor = self.model(**self.args)
        labels = y.argmax(axis=1) if y.ndim == 2 else y
        # Store which original classes are present so predict() can pad correctly.
        self.train_classes_ = np.unique(labels)
        # Remap to contiguous 0..k-1 — required by estimators such as XGBoost.
        if not np.array_equal(self.train_classes_, np.arange(len(self.train_classes_))):
            remap = {c: i for i, c in enumerate(self.train_classes_)}
            labels = np.array([remap[l] for l in labels])
        self.predictor.fit(X, labels)

    def predict(self, X):
        if hasattr(self.predictor, 'predict_proba'):
            proba = self.predictor.predict_proba(X)
            # If all class scores are ~0 the sum is 0 → NaN. Replace those rows with a uniform distribution.
            nan_rows = np.isnan(proba).any(axis=1)
            if nan_rows.any():
                proba[nan_rows] = 1.0 / proba.shape[1]
            if proba.shape[1] == self.n_classes:
                return proba
            # Some classes were absent from the training fold — pad with zero probability
            full_proba = np.zeros((proba.shape[0], self.n_classes))
            full_proba[:, self.train_classes_] = proba
            return full_proba
        scores = self.predictor.decision_function(X)
        if scores.ndim == 1 or scores.shape[1] == self.n_classes:
            return scores
        # Some classes were absent from the training fold — pad below the minimum real score
        full_scores = np.full((scores.shape[0], self.n_classes), scores.min() - 1.0)
        full_scores[:, self.train_classes_] = scores
        return full_scores
