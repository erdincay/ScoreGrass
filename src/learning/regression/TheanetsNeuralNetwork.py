from sklearn import cross_validation
import theanets

__author__ = 'Kern'


class TheanetsNeuralNetwork:
    def __init__(self):
        self.exp = theanets.Experiment(theanets.Regressor)

    def train(self, x_data, y_data):
        x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_data, y_data,
                                                                             test_size=0.3, random_state=0)
        return self.exp.train((x_train, y_train), (x_test, y_test))

    def predict(self, in_data):
        return self.exp.network.predict(in_data)