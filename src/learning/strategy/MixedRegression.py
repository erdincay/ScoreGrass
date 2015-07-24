import math

from src.learning.strategy.RegressionManager import RegressionManager, combine_name, deserialize_json
from src.learning.regression import ScikitMultiTaskLasso
from src.learning.regression import TheanetsANNs
from src.learning.regression import PyBrainANNs


__author__ = 'Kern'


def _register_regression(x_dim, y_dim):
    return [
        ScikitMultiTaskLasso.ScikitMultiTaskLasso(
            combine_name(MixedRegression.__name__, ScikitMultiTaskLasso.ScikitMultiTaskLasso.__name__)),

        TheanetsANNs.TheanetsANNs(x_dim, y_dim, math.floor(x_dim / 2),
                                  combine_name(MixedRegression.__name__, TheanetsANNs.TheanetsANNs.__name__)),

        PyBrainANNs.PyBrainANNs(x_dim, y_dim, math.floor(x_dim / 2),
                                combine_name(MixedRegression.__name__, PyBrainANNs.PyBrainANNs.__name__))
    ]


class MixedRegression(RegressionManager):
    def __init__(self, x_dim, y_dim):
        super().__init__(_register_regression(x_dim, y_dim), x_dim, y_dim)

    @classmethod
    def deserialize_regression(cls, path):
        x_dim, y_dim = deserialize_json(path, cls.__name__)
        return cls(x_dim, y_dim).load(path)