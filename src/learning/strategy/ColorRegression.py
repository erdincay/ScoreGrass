from src.learning.regression import ScikitRidge
from src.learning.regression import ScikitLasso
from src.learning.strategy.RegressionManager import RegressionManager, combine_name, deserialize_json

__author__ = 'Kern'


def _register_regression():
    return [
        ScikitLasso.ScikitLasso(combine_name(ColorRegression.__name__, ScikitLasso.ScikitLasso.__name__)),
        ScikitRidge.ScikitRidge(combine_name(ColorRegression.__name__, ScikitRidge.ScikitRidge.__name__))
    ]


class ColorRegression(RegressionManager):
    def __init__(self, x_dim, y_dim):
        super().__init__(_register_regression(), x_dim, y_dim)

    @classmethod
    def deserialize_regression(cls, path):
        x_dim, y_dim = deserialize_json(path, cls.__name__)
        return cls(x_dim, y_dim).load(path)
