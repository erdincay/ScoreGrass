from src.learning.strategy import RegressionManager
from src.learning.regression import ScikitBayesianRidge
from src.learning.regression import PyBrainANNs
from src.learning.regression import ScikitSvm
from src.learning.regression import TheanetsANNs
from src.learning.regression import ScikitMultiTaskLasso
from src.learning.strategy.RegressionManager import ModelType, combine_name

__author__ = 'Kern'


def _register_regression(x_dim, y_dim):
    return [
        (ModelType.train_only, ScikitBayesianRidge.BayesianRidge(combine_name(QualityRegression.__name__, ScikitBayesianRidge.BayesianRidge.__name__))),
        (ModelType.train_only, ScikitMultiTaskLasso.MultiTaskLasso(combine_name(QualityRegression.__name__, ScikitMultiTaskLasso.MultiTaskLasso.__name__))),
        (ModelType.train_only, ScikitSvm.ScikitSvm('rbf', combine_name(QualityRegression.__name__, ScikitSvm.ScikitSvm.__name__))),
        (ModelType.train_test, TheanetsANNs.TheanetsANNs(x_dim, y_dim, x_dim / 2, combine_name(QualityRegression.__name__, TheanetsANNs.TheanetsANNs.__name__))),
        (ModelType.train_only, PyBrainANNs.PyBrainANNs(x_dim, y_dim, x_dim / 2, combine_name(QualityRegression.__name__, PyBrainANNs.PyBrainANNs.__name__)))
    ]


class QualityRegression(RegressionManager):
    def __init__(self, x_dim, y_dim):
        super().__init__(_register_regression(x_dim, y_dim))