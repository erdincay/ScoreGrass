import theanets

__author__ = 'Kern'


class TheanetsANNs:
    def __init__(self, x_dim, y_dim, hidden_size, s_id):
        self.serialize_id = s_id
        self.exp = theanets.Experiment(theanets.Regressor, layers=(x_dim, hidden_size, y_dim))

    def train(self, x_train, y_train, x_test, y_test):
        assert (x_train.shape[0] == y_train.shape[0])
        assert (x_test.shape[0] == y_test.shape[0])
        assert (x_train.shape[1] == x_test.shape[1])
        assert (y_train.shape[1] == y_test.shape[1])

        return self.exp.train((x_train, y_train), (x_test, y_test))

    def predict(self, x_data):
        return self.exp.network.predict(x_data)

    def save(self, path):
        self.exp.save(path)

    def load(self, path):
        self.exp.load(path)