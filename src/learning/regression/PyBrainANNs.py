import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer
from sklearn.externals import joblib


__author__ = 'Kern'


class PyBrainANNs:
    def __init__(self, x_dim, y_dim, hidden_size, s_id):
        self.serialize_id = s_id
        self.net = FeedForwardNetwork()

        in_layer = LinearLayer(x_dim)
        hidden_layer = SigmoidLayer(hidden_size)
        out_layer = LinearLayer(y_dim)
        self.net.addInputModule(in_layer)
        self.net.addModule(hidden_layer)
        self.net.addOutputModule(out_layer)

        in_to_hidden = FullConnection(in_layer, hidden_layer)
        hidden_to_out = FullConnection(hidden_layer, out_layer)
        self.net.addConnection(in_to_hidden)
        self.net.addConnection(hidden_to_out)

        self.net.sortModules()

    def train(self, x_data, y_data):
        assert (x_data.shape[0] == y_data.shape[0])

        if len(y_data.shape) == 1:
            y_matrix = np.matrix(y_data).T
        else:
            y_matrix = y_data.values

        assert (x_data.shape[1] == self.net.indim)
        assert (y_matrix.shape[1] == self.net.outdim)

        train_data_set = SupervisedDataSet(self.net.indim, self.net.outdim)
        train_data_set.setField("input", x_data)
        train_data_set.setField("target", y_matrix)

        trainer = BackpropTrainer(self.net, train_data_set)
        trainer.train()

    def predict(self, x_data):
        return [self.net.activate(sample) for sample in x_data]

    def save(self, path):
        joblib.dump(self.net, path)

    def load(self, path):
        self.net = joblib.load(path)