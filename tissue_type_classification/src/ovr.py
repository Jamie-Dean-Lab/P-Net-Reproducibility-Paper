from sklearn.multiclass import OneVsRestClassifier

class OVRWrapper:
    def __init__(self, estimator, args):
        self.model = estimator
        self.args = args
    def fit(self, X, y):
        self.predictor = OneVsRestClassifier(self.model(**self.args))
        self.predictor.fit(X, y)
    def predict(self, X):
        return self.predictor.decision_function(X)