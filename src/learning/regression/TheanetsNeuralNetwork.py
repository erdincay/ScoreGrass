import theanets
from ..preprocessing.CrossValidation import train_test_split

__author__ = 'Kern'


class TheanetsNeuralNetwork:
    def __init__(self):
        self.exp = theanets.Experiment(theanets.Regressor)

    def train(self, x_data, y_data):
        x_train, x_test, y_train, y_test = train_test_split(0.3)(x_data, y_data)
        return self.exp.train((x_train, y_train), (x_test, y_test))

    def predict(self, in_data):
        return self.exp.network.predict(in_data)