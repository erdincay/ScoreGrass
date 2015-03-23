from sklearn.svm import SVR

from src.learning.evaluation.CrossValidation import train_test_split


__author__ = 'Kern'


class ScikitSvm:
    def __init__(self, knr):
        self.svr = SVR(kernel=knr)

    def training(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        x_train, x_test, y_train, y_test = train_test_split(0.3)(x_data, y_data)
        self.svr.fit(x_train, y_train)
        return self.svr.score(x_test, y_test)

    def predict(self, x_data):
        return self.svr.predict(x_data)