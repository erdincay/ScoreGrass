__author__ = 'Kern'

from sklearn.svm import SVR


class SklearnSVM:
    def __init__(self, kernel):
        self.svr = SVR(kernel)

    def training(self, in_data, output):
        self.svr.fit(in_data, output)

    def predict(self, in_data):
        return self.svr.predict(in_data)