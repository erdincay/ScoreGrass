import theanets

import numpy as np

from src.learning.regression.ScikitLearning import regression_score

__author__ = 'Kern'


class TheanetsANNs:
    def __init__(self, x_dim, y_dim, hidden_size, s_id):
        self.serialize_id = s_id
        self.exp = theanets.Experiment(theanets.Regressor, layers=(x_dim, hidden_size, y_dim))

    @staticmethod
    def __trans_on_dimensions(data):
        if len(data.shape) == 1:
            data_matrix = np.matrix(data).T
        else:
            data_matrix = data.values
        return data_matrix

    @staticmethod
    def _prepare_dataset(x_data, y_data):
        return x_data, TheanetsANNs.__trans_on_dimensions(y_data)

    def train(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        return self.exp.train(TheanetsANNs._prepare_dataset(x_data, y_data))

    def score(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        true_data = TheanetsANNs.__trans_on_dimensions(y_data)
        predict_data = self.predict(x_data)
        return regression_score(true_data, predict_data)

    def predict(self, x_data):
        return self.exp.network.predict(x_data)

    def save(self, path):
        self.exp.save(path)

    def load(self, path):
        self.exp.load(path)
