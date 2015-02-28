from pybrain.datasets.sequential import SequentialDataSet
from pybrain.structure.connections.full import FullConnection
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer

from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer

from src.learning.preprocess.CrossValidation import data_set_split


__author__ = 'Kern'


class PyBrainANNs:
    def __init__(self, x_dim, y_dim, hidden_size):
        self.data_set = SequentialDataSet(x_dim, y_dim)

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
        self.data_set.setField("input", x_data)
        self.data_set.setField("target", y_data)

        train_ds, test_ds = data_set_split(0.3)(self.data_set)

        trainer = BackpropTrainer(self.net, train_ds)
        trainer.train()

        return test_ds

    def predict(self, x_data):
        return self.net.activate(x_data)

    def predict_set(self, x_datas):
        return self.net.activateOnDataset(x_datas)