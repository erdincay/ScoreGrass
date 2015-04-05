from sklearn.externals import joblib

__author__ = 'Kern'


class ScikitModel:
    def __init__(self, model, s_id):
        self.serialize_id = s_id
        self.model = model

    def train(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])
        self.model.fit(x_data, y_data)

    def predict(self, x_data):
        return self.model.predict(x_data)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)