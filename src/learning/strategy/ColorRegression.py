from src.learning.strategy import RegressionManager
from src.learning.regression import ScikitRidge
from src.learning.regression import ScikitLasso
from src.learning.strategy.RegressionManager import ModelType, combine_name

__author__ = 'Kern'


def _register_regression():
    return [
        (ModelType.train_only, ScikitLasso.Lasso(combine_name(ColorRegression.__name__, ScikitLasso.Lasso.__name__))),
        (ModelType.train_only, ScikitRidge.Ridge(combine_name(ColorRegression.__name__, ScikitRidge.Ridge.__name__))),
    ]


class ColorRegression(RegressionManager):
    def __init__(self):
        super().__init__(_register_regression)