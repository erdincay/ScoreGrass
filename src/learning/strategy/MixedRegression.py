from src.learning.strategy import RegressionManager
from src.learning.regression import ScikitMultiTaskLasso
from src.learning.regression import TheanetsANNs
from src.learning.regression import PyBrainANNs
from src.learning.strategy.RegressionManager import ModelType, serialize_extension, combine_name

__author__ = 'Kern'


def _register_regression(x_dim, y_dim):
    return [
        (ModelType.train_only, ScikitMultiTaskLasso.MultiTaskLasso(combine_name(MixedRegression.__name__, ScikitMultiTaskLasso.MultiTaskLasso.__name__))),
        (ModelType.train_test, TheanetsANNs.TheanetsANNs(x_dim, y_dim, x_dim / 2, combine_name(MixedRegression.__name__, TheanetsANNs.TheanetsANNs.__name__))),
        (ModelType.train_only, PyBrainANNs.PyBrainANNs(x_dim, y_dim, x_dim / 2, combine_name(MixedRegression.__name__, PyBrainANNs.PyBrainANNs.__name__)))
    ]


class MixedRegression(RegressionManager):
    def __init__(self, x_dim, y_dim):
        super().__init__(_register_regression(x_dim, y_dim))