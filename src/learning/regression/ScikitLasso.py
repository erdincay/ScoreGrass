from sklearn.linear_model import Lasso

__author__ = 'Kern'


class Lasso:
    def __init__(self):
        self.clf = Lasso(alpha=0.1)

    def train(self, x_data, y_data):
        self.clf.fit(x_data, y_data)

    def predict(self, x_data):
        return self.clf.predict(x_data)