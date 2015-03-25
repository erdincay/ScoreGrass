from sklearn.linear_model import Ridge

__author__ = 'Kern'


class Ridge:
    def __init__(self):
        self.clf = Ridge(alpha=0.1)

    def train(self, x_data, y_data):
        self.clf.fit(x_data, y_data)

    def predict(self, x_data):
        return self.clf.predict(x_data)