from sklearn.svm import SVR
from ..preprocessing.CrossValidation import train_test_split


__author__ = 'Kern'


class SklearnSVM:
    def __init__(self, kernel):
        self.svr = SVR(kernel)

    def training(self, x_data, y_data):
        x_train, x_test, y_train, y_test = train_test_split(0.3)(x_data, y_data)
        self.svr.fit(x_train, y_train)
        self.svr.score(x_test, y_test)

    def predict(self, in_data):
        return self.svr.predict(in_data)