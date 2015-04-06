from src.learning.regression import ScikitRidge
from src.learning.regression import ScikitLasso
from src.learning.strategy.RegressionManager import RegressionManager, combine_name

__author__ = 'Kern'


def _register_regression():
    return [
        ScikitLasso.ScikitLasso(combine_name(ColorRegression.__name__, ScikitLasso.ScikitLasso.__name__)),
        ScikitRidge.ScikitRidge(combine_name(ColorRegression.__name__, ScikitRidge.ScikitRidge.__name__))
    ]


class ColorRegression(RegressionManager):
    def __init__(self, serialize_path):
        super().__init__(_register_regression(), serialize_path)