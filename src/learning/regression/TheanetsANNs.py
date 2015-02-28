import theanets
from ..preprocess.CrossValidation import train_test_split

__author__ = 'Kern'


class TheanetsANNs:
    def __init__(self):
        self.exp = theanets.Experiment(theanets.Regressor)

    def train(self, x_data, y_data):
        assert(x_data.shape[0] == y_data.shape[0])
        x_train, x_test, y_train, y_test = train_test_split(0.3)(x_data, y_data)
        return self.exp.train((x_train, y_train), (x_test, y_test))

    def predict(self, x_data):
        return self.exp.network.predict(x_data)