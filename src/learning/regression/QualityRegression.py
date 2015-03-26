from src.learning.regression import ScikitBayesianRidge
from src.learning.regression import PyBrainANNs
from src.learning.regression import ScikitSvm
from src.learning.regression import TheanetsANNs
from src.learning.regression import ScikitMultiTaskLasso

__author__ = 'Kern'


def register_regression_quality(x_dim, y_dim):
    return [
        ScikitBayesianRidge.BayesianRidge(),
        ScikitMultiTaskLasso.MultiTaskLasso(),
        ScikitSvm.ScikitSvm('rbf'),
        TheanetsANNs.TheanetsANNs(x_dim, y_dim, x_dim / 2),
        PyBrainANNs.PyBrainANNs(x_dim, y_dim, x_dim / 2),
    ]

def train_color(learn_models, x_data, y_data):
    for m in learn_models:
        m.train(x_data, y_data)