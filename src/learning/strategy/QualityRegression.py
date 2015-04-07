import math

from src.learning.regression import ScikitBayesianRidge
from src.learning.regression import PyBrainANNs
from src.learning.regression import ScikitSvm
from src.learning.regression import TheanetsANNs
from src.learning.strategy.RegressionManager import RegressionManager, combine_name, deserialize_json


__author__ = 'Kern'


def _register_regression(x_dim, y_dim):
    return [
        ScikitBayesianRidge.ScikitBayesianRidge(combine_name(QualityRegression.__name__, ScikitBayesianRidge.ScikitBayesianRidge.__name__)),
        ScikitSvm.ScikitSvm('rbf', combine_name(QualityRegression.__name__, ScikitSvm.ScikitSvm.__name__)),
        TheanetsANNs.TheanetsANNs(x_dim, y_dim, math.floor(x_dim / 2), combine_name(QualityRegression.__name__, TheanetsANNs.TheanetsANNs.__name__)),
        PyBrainANNs.PyBrainANNs(x_dim, y_dim, math.floor(x_dim / 2), combine_name(QualityRegression.__name__, PyBrainANNs.PyBrainANNs.__name__))
    ]


class QualityRegression(RegressionManager):
    def __init__(self, x_dim, y_dim):
        super().__init__(_register_regression(x_dim, y_dim))

    @classmethod
    def deserialize_regression(cls, path):
        x_dim, y_dim = deserialize_json(path, cls.__name__)
        return cls(x_dim, y_dim).load(path)
