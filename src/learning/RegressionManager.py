__author__ = 'Kern'


class RegressionManager:
    def __init__(self, model_list):
        self.model_list = model_list

    def train(self, x_data, y_data):
        for m in self.model_list:
            m.train(x_data, y_data)

    def predict(self, x_data):
        return [m.predict(x_data) for m in self.model_list]