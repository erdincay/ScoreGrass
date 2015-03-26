import theanets

from src.learning.evaluation.CrossValidation import train_test_split


__author__ = 'Kern'


class TheanetsANNs:
    def __init__(self, x_dim, y_dim, hidden_size):
        self.exp = theanets.Experiment(theanets.Regressor, layers=(x_dim, hidden_size, y_dim))

    def train(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        x_train, x_test, y_train, y_test = train_test_split(0.3)(x_data, y_data)
        return self.exp.train((x_train, y_train), (x_test, y_test))

    def predict(self, x_data):
        return self.exp.network.predict(x_data)

    def save(self, path):
        self.exp.save(path)

    def load(self, path):
        self.exp.load(path)