from src.learning.regression import ScikitRidge
from src.learning.regression import ScikitLasso
from src.learning.strategy.RegressionManager import RegressionManager, ModelType, combine_name

__author__ = 'Kern'


def _register_regression():
    return [
        (ModelType.train_only, ScikitLasso.ScikitLasso(combine_name(ColorRegression.__name__, ScikitLasso.ScikitLasso.__name__))),
        (ModelType.train_only, ScikitRidge.ScikitRidge(combine_name(ColorRegression.__name__, ScikitRidge.ScikitRidge.__name__))),
    ]


class ColorRegression(RegressionManager):
    def __init__(self, serialize_path):
        super().__init__(_register_regression(), serialize_path)