import theanets
import numpy as np

__author__ = 'Kern'


class TheanetsANNs:
    def __init__(self, x_dim, y_dim, hidden_size, s_id):
        self.serialize_id = s_id
        self.exp = theanets.Experiment(theanets.Regressor, layers=(x_dim, hidden_size, y_dim))

    def train(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])

        if len(y_data.shape) == 1:
            y_matrix = np.matrix(y_data).T
        else:
            y_matrix = y_data.values

        return self.exp.train((x_data, y_matrix))

    def predict(self, x_data):
        return self.exp.network.predict(x_data)

    def save(self, path):
        self.exp.save(path)

    def load(self, path):
        self.exp.load(path)