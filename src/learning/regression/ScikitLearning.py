import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import explained_variance_score

__author__ = 'Kern'


def regression_score(true_data, predict_data):
    assert (true_data.shape == predict_data.shape)
    if len(true_data.shape) == 1 or true_data.shape[1] == 1:
        return explained_variance_score(true_data, predict_data)
    else:
        return np.mean([explained_variance_score(true_data[:, index], predict_data[:, index]) for index in
                        range(true_data.shape[1])])


class ScikitModel:
    def __init__(self, model, s_id):
        self.serialize_id = s_id
        self.model = model

    def train(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        self.model.fit(x_data, y_data)

    def score(self, x_data, y_data):
        return self.model.score(x_data, y_data)

    def predict(self, x_data):
        return self.model.predict(x_data)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
